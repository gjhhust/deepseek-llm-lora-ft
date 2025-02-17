# app.py
from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

app = Flask(__name__)


class DeepSeekChatEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self._init_generation_config()
        
    def _init_generation_config(self):
        self.model.generation_config = GenerationConfig.from_pretrained(self.model_path)
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.max_new_tokens = 512  # 控制最大生成长度
        # self.model.generation_config.do_sample = True
        # self.model.generation_config.temperature = 0.7
        # self.model.generation_config.top_p = 0.9

    def create_prompt(self, system_prompt, history, new_input):
        """构建符合DeepSeek格式的对话提示"""
        messages = [{"role": "system", "content": system_prompt}]
        messages += history
        messages.append({"role": "user", "content": new_input})
        
        template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] }}\n\n"
            "{% set messages = messages[1:] %}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "User: {{ message['content'] }}\n\n"
            "Assistant:"
            "{% else %}"
            "{{ message['content'] }}"  # 直接拼接助理回复
            "{% endif %}"
            "{% endfor %}"
        )
        
        return self.tokenizer.apply_chat_template(
            messages,
            chat_template=template,
            add_generation_prompt=True,
            tokenize=False
        )

    def generate_response(self, system_prompt, history, new_input):
        try:
            # 构建完整提示
            full_prompt = self.create_prompt(system_prompt, history, new_input)
            
            # Tokenize输入
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt", 
                return_attention_mask=True
            ).to(self.model.device)
            
            # 生成响应
            outputs = self.model.generate(
                **inputs,
                generation_config=self.model.generation_config
            )
            
            # 解码并清理结果
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:], 
                skip_special_tokens=True
            )
            
            # 移除可能重复的Assistant前缀
            response = response.split("User:")[0].strip()
            
            return response
        
        except Exception as e:
            print(f"生成错误: {str(e)}")
            return ""

model_path = "/root/deepseek-llm-7B-chat-lora-ft/train/output/chai-epoch_10_base_2/merge_model"  # 修改为实际模型路径
model_path2 = "/root/model/deepseek-ai/deepseek-llm-7b-base"
    
# 全局变量保持模型和聊天状态
chat_engine = None

current_chat1 = {
    "system_message": "",
    "history": []
}
current_chat2 = {
    "system_message": "",
    "history": []
}

def initialize_model1(model_path1):
    global chat_engine
    chat_engine = DeepSeekChatEngine(model_path1)

def initialize_model2(model_path2):
    global chat_engine
    chat_engine = DeepSeekChatEngine(model_path2)
    
def generate_response2(prompt):
    global current_chat2 
    response = chat_engine.generate_response(
            system_prompt=current_chat2["system_message"],
            history=current_chat2["history"],
            new_input=prompt
        )
    current_chat2["history"].append({"role": "user", "content": prompt})
    current_chat2["history"].append({"role": "assistant", "content": response})
    print("\n\n", current_chat2["history"])
    return response

def generate_response1(prompt):
    global current_chat1 
    response = chat_engine.generate_response(
            system_prompt=current_chat1["system_message"],
            history=current_chat1["history"],
            new_input=prompt
        )
    current_chat1["history"].append({"role": "user", "content": prompt})
    current_chat1["history"].append({"role": "assistant", "content": response})
    print("\n\n", current_chat1["history"])
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat1', methods=['POST'])
def chat_endpoint1():
    data = request.json
    if chat_engine is None or chat_engine.model_path != model_path:
        initialize_model1(model_path)
    response1 = generate_response1(data['message'])
    return jsonify({"response1": response1})

@app.route('/chat2', methods=['POST'])
def chat_endpoint2():
    data = request.json
    if chat_engine is None or chat_engine.model_path != model_path2:
        initialize_model2(model_path2)
    response2 = generate_response2(data['message'])
    return jsonify({"response2": response2})

@app.route('/new_chat', methods=['POST'])
def new_chat():
    data = request.json
    current_chat1["history"] = [{"role": "assistant", "content": data['init_action']}]
    current_chat2["history"] = [{"role": "assistant", "content": data['init_action']}]
    return jsonify({"status": "success"})

@app.route('/set_role', methods=['POST'])
def set_role():
    data = request.json
    current_chat1["system_message"] = data['system_message']
    current_chat1["history"] = [{"role": "assistant", "content": data['init_action']}]
    
    current_chat2["system_message"] = data['system_message']
    current_chat2["history"] = [{"role": "assistant", "content": data['init_action']}]
    return jsonify({"status": "success"})

if __name__ == '__main__':
    
    # initialize_models(model_path, model_path2)
    current_chat1["system_message"] = "You are a helpful assistant."  # 默认系统消息
    current_chat1["init_action"] = "我有什么可以帮你的."  # 默认系统消息
    current_chat2["system_message"] = "You are a helpful assistant."  # 默认系统消息
    current_chat2["init_action"] = "我有什么可以帮你的."  # 默认系统消息
    app.run(host='0.0.0.0', port=5000, debug=True)