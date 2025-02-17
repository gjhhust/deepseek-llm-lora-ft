# app.py
from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

app = Flask(__name__)

# 全局变量保持模型和聊天状态
model = None
tokenizer = None
current_chat = {
    "system_message": "",
    "history": []
}

def initialize_model(model_path):
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                               torch_dtype=torch.bfloat16,
                                               device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

def generate_response(prompt):
    global model, tokenizer, current_chat
    
    # 构建完整消息历史
    messages = [{"role": "system", "content": current_chat["system_message"]}]
    messages += current_chat["history"]
    messages.append({"role": "user", "content": prompt})
    print("\n*****************history****************")
    print(current_chat["history"])
    print("*********************************\n")
    print(prompt)
    # 应用模板
    deepseek_template = (
        "{{ bos_token }}"
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] }}\n\n"
        "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "User: {{ message['content'] }}\n\nAssistant:"
        "{% else %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
    )
    tokenizer.chat_template = deepseek_template

    # 生成响应
    input_tensor = tokenizer.apply_chat_template(messages, 
                                               add_generation_prompt=True,
                                               return_tensors="pt").to(model.device)
    outputs = model.generate(input_tensor, max_new_tokens=2048)
    response = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    
    # 更新历史
    current_chat["history"].extend([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ])
    
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    response = generate_response(data['message'])
    return jsonify({"response": response})

@app.route('/new_chat', methods=['POST'])
def new_chat():
    data = request.json
    current_chat["history"] = [{"role": "assistant", "content": data['init_action']}]
    return jsonify({"status": "success"})

@app.route('/set_role', methods=['POST'])
def set_role():
    data = request.json
    current_chat["system_message"] = data['system_message']
    current_chat["history"] = [{"role": "assistant", "content": data['init_action']}]
        
    return jsonify({"status": "success"})

if __name__ == '__main__':
    model_path = "/root/deepseek-llm-7B-chat-lora-ft/train/chai-epoch_10_base_2/merge_model"  # 修改为实际模型路径
    initialize_model(model_path)
    current_chat["system_message"] = "You are a helpful assistant."  # 默认系统消息
    current_chat["init_action"] = "我有什么可以帮你的."  # 默认系统消息
    app.run(host='0.0.0.0', port=5000, debug=True)