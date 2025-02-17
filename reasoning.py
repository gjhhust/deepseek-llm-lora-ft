import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig

### infiniAI的megrez-3b-omni测试效果
def megrez_inference(model_path, image_path):
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
        )
        .eval()
        .cuda()
    )

    # Chay with text
    text_messages = [
        {
            "role": "user",
            "content": {
                "text": "请问你叫什么？"
            }
        }
    ]

    # Chat with text and image
    image_messages = [
        {
            "role": "user",
            "content": {
                "text": "请你描述下图像",
                "image": image_path,
            },
        },
    ]

    MAX_NEW_TOKENS = 500

    # text
    text_response = model.chat(
        text_messages,
        sampling=False,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.8,
    )

    image_response = model.chat(
        image_messages,
        sampling=False,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.8,
    )

    print(text_response)
    print(image_response)


def yi_chat_model_reasoning(model_path: str, prompt: str):
    """
    单论对话的回复
    :param model_path: 模型下载地址
    :param prompt: 需要询问的问题
    :return: 回复的话
    """

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto', device_map="auto")
    model.eval()

    messages = [
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    return response

def yi_base_model_reasoning(model_path: str, prompt: str):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto",device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=256)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def qwen_model_inference(model_path: str,prompt: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    messages = [
        {"role": "system", "content": "你是千问，一个很有用的助手"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def llama_inference(model_path: str, prompt: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,device_map="cuda:0")

    # 将输入文本转换为模型的输入格式
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 推理过程
    with torch.no_grad():
        # 生成输出，调整参数以控制生成长度
        output = model.generate(
            inputs['input_ids'],
            max_length=2048,  # 设置最大生成长度
            num_return_sequences=1,  # 生成一个序列
            no_repeat_ngram_size=2,  # 防止重复的n-gram
            top_p=0.95,  # nucleus sampling
            temperature=0.7  # 控制输出的随机性
        )

    # 解码并打印生成的文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def deepseek_model_inference(model_path: str, prompt: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    deepseek_template = (
        "{{ bos_token }}"  # 添加前缀的bos_token
        "{% if messages[0]['role'] == 'system' %}"  # 处理系统消息
        "{{ messages[0]['content'] }}\n\n"  # 系统消息格式
        "{% set messages = messages[1:] %}"  # 移除已处理的系统消息
        "{% endif %}"
        "{% for message in messages %}"  # 遍历剩余消息
        "{% if message['role'] == 'user' %}"
        "User: {{ message['content'] }}\n\nAssistant:"  # 用户消息格式
        "{% else %}"
        "{{ message['content'] }}"  # 助理消息直接拼接内容
        "{% endif %}"
        "{% endfor %}"
    )
    tokenizer.chat_template = deepseek_template
    
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=2048)

    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    return result

# 多轮对话推理
def deepseek_multi_conversation_inference(model_path: str, prompt: str, chat_history: list, max_new_tokens=2048):
    """
    chat_history:[{"role": "user", "content": ……},{"role": "assistant", "content": ……}]
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    messages = chat_history.copy()
    messages.append({"role": "user", "content": prompt})
    chat_history.append({"role": "user", "content": prompt})
    
    deepseek_template = (
        "{{ bos_token }}"  # 添加前缀的bos_token
        "{% if messages[0]['role'] == 'system' %}"  # 处理系统消息
        "{{ messages[0]['content'] }}\n\n"  # 系统消息格式
        "{% set messages = messages[1:] %}"  # 移除已处理的系统消息
        "{% endif %}"
        "{% for message in messages %}"  # 遍历剩余消息
        "{% if message['role'] == 'user' %}"
        "User: {{ message['content'] }}\n\nAssistant:"  # 用户消息格式
        "{% else %}"
        "{{ message['content'] }}"  # 助理消息直接拼接内容
        "{% endif %}"
        "{% endfor %}"
    )
    
    tokenizer.chat_template = deepseek_template

    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=max_new_tokens)

    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    chat_history.append({"role": "assistant", "content": result})
    return result,chat_history

# 多轮对话推理
def deepseek_multi_conversation_inference_orige(model_path: str, prompt: str, chat_history: list, max_new_tokens=2048):
    """
    chat_history:[{"role": "user", "content": ……},{"role": "assistant", "content": ……}]
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    messages = chat_history.copy()
    messages.append({"role": "user", "content": prompt})
    chat_history.append({"role": "user", "content": prompt})
    
    deepseek_template = (
        "{{ bos_token }}"  # 添加前缀的bos_token
        "{% if messages[0]['role'] == 'system' %}"  # 处理系统消息
        "{{ messages[0]['content'] }}\n\n"  # 系统消息格式
        "{% set messages = messages[1:] %}"  # 移除已处理的系统消息
        "{% endif %}"
        "{% for message in messages %}"  # 遍历剩余消息
        "{% if message['role'] == 'user' %}"
        "User: {{ message['content'] }}\n\nAssistant:"  # 用户消息格式
        "{% else %}"
        "{{ message['content'] }}"  # 助理消息直接拼接内容
        "{% endif %}"
        "{% endfor %}"
    )
    
    tokenizer.chat_template = deepseek_template

    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=max_new_tokens)

    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    chat_history.append({"role": "assistant", "content": result})
    return result,chat_history

if __name__ == "__main__":
    model_path = "/root/model/deepseek-ai/deepseek-llm-7b-base"
    merge_path = "/root/deepseek-llm-7B-chat-lora-ft/train/chai-epoch_10_base_2/merge_model"

    inputs = """*戴上耳塞，开始看电影"""
    prompt = f"""你现在扮演的是布鲁克，一个布鲁克（空姐）的角色： 她是纽约一位才华横溢的词曲作者。她善于捕捉人们的感受。角色背景设定： 您乘坐的是商务舱。布鲁克绕过每个座位，检查每个人的安全带。当她经过你身边时，你对她微笑，她也回以微笑。她回到空乘室，给每个人发了一份菜单，让大家点饮料和食物。她推着一车零食走过来，给每个人都提供了一份。她注意到了你，再次微笑。她免费给了你一杯饮料，并小声说--别告诉别人，这是给你的。 [她是你朋友的表妹，但你们俩以前都不知道这件事。］详细设定： 事实：她是布鲁克。今年 29 岁。她是西雅图的一名空姐。她的星座是双子座。她善于捕捉人们的感受。她热爱运动和时尚。她是你朋友的表妹。她在飞行中给你特殊待遇。她很专业。无论您有什么需求，她都会尽力满足： 她是纽约一位才华横溢的作曲家。她善于捕捉人们的感受： 布鲁克（空姐）是纽约一位才华横溢的词曲作者。她善于捕捉人们的感受。\n{inputs}"""
    # print(deepseek_model_inference(merge_path,inputs))
    
    
    print("*****************************merge_model response******************************************************")
    chat_history=[]
    result, chat_history=deepseek_multi_conversation_inference(model_path,prompt,chat_history)
    print(result)

    inputs = """是的，这基本上是我每次飞行时的首选电影"""
    result, chat_history = deepseek_multi_conversation_inference(model_path, inputs, chat_history)
    print(result)

    inputs = """*屏幕上弹出的场面，脸红了"""
    result, chat_history = deepseek_multi_conversation_inference(model_path, inputs, chat_history)
    print(result)

    inputs = """我旁边有个空座位，如果你想留下来看一会儿的话"""
    result, chat_history = deepseek_multi_conversation_inference(model_path, inputs, chat_history)
    print(result)
    
    # print("\n\n\n*****************************orige response******************************************************")
    # chat_history=[]
    # result, chat_history=deepseek_multi_conversation_inference(model_path,inputs,chat_history)
    # print(result)

    # inputs = """可是我是个社恐，做不到怎么办？"""
    # result, chat_history = deepseek_multi_conversation_inference(model_path, inputs, chat_history)
    # print(result)

    # inputs = """我在数学方面总是比他落后很多，我尝试了很多方法提高，但还是觉得力不从心。"""
    # result, chat_history = deepseek_multi_conversation_inference(model_path, inputs, chat_history)
    # print(result)


