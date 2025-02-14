import json
import jsonlines
import random # 导入 random 模块

def process_raw_data(input_file, output_file):
    cleaned_data = []
    with jsonlines.open(input_file, 'r') as reader:
        for session_data in reader:
            cleaned_conversation = process_session(session_data)
            if cleaned_conversation: # 确保对话不为空才添加
                cleaned_data.append({"conversation": cleaned_conversation})

    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(cleaned_data)

def process_session(session_data):
    conversation_turns = []
    session_id = session_data.get("session", "unknown_session")
    bot_prompt_data = session_data.get("bot_info", {}).get("prompt", {})
    bot_prompt_str = bot_prompt_data.get("prompt", '{}') # 获取 prompt 字符串
    try:
        bot_prompt = json.loads(bot_prompt_str) # 解析 prompt 字符串为 JSON
        system_prompt = construct_system_prompt(bot_prompt, session_data.get("bot_info", {}).get("bot_name", "Bot")) # 构建系统 prompt
    except json.JSONDecodeError:
        print(f"警告: Session {session_id} 的 prompt JSON 解析失败，跳过此会话.")
        return None # 如果 prompt 解析失败，则跳过整个会话
    except Exception as e:
        print(f"处理 Session {session_id} 的 prompt 时发生错误: {e}, 跳过此会话.")
        return None

    conversation_history = session_data.get("conversation", [])
    if not conversation_history:
        return None # 如果对话历史为空，则跳过

    negative_responses = [] # 用于存储负样本回复
    turn_count = 0 #  对话轮次计数器

    # 确保对话以用户开始
    if conversation_history and conversation_history[0]['role'] == 'bot':
        conversation_history = conversation_history[1:] # 移除 Bot 的第一句 Greeting，如果存在

    if not conversation_history or conversation_history[0]['role'] != 'user':
         return None # 确保对话以用户开始

    for i in range(0, len(conversation_history) - 1, 2): # 每次迭代处理用户和 Bot 消息对
        user_message = conversation_history[i]
        bot_message = conversation_history[i+1] if i + 1 < len(conversation_history) else None

        if user_message['role'] == 'user' and bot_message and bot_message['role'] == 'bot':
            metadata = bot_message.get('metadata', {})
            if metadata.get('is_accept') == True and metadata.get('is_reset') == False and metadata.get('gpt_type') == 50:
                user_content = user_message['content']
                bot_content = bot_message['content']
                star_rating = metadata.get('star') # 获取星级评分
                msg_type = metadata.get('msg_type') # 获取消息类型
                history_count = metadata.get('history_count') # 获取历史对话轮数
                turn_count += 1 # 递增对话轮次计数器

                # 计算质量评分 (使用更复杂的评分函数)
                quality_score = calculate_quality_score_v2(star_rating, history_count, msg_type, turn_count)

                # 提取负样本回复 (在被接受的回答之前的被拒绝回答)
                current_negative_responses = []
                for j in range(i + 1, len(conversation_history)): # 查找当前 Bot 回复之后的消息
                    if conversation_history[j]['role'] == 'bot' and conversation_history[j] != bot_message: # 排除当前接受的 Bot 消息
                        meta = conversation_history[j].get('metadata', {})
                        if meta.get('is_accept') == False or meta.get('is_reset') == True: # 如果是不接受或重置的消息
                            current_negative_responses.append(conversation_history[j]['content'])
                        else:
                            break # 遇到第一个被接受的消息或者用户消息就停止，因为负样本只应该在当前接受消息之前

                conversation_turn = {}
                if conversation_turns: # 如果不是第一轮对话，则不添加 system
                    conversation_turn = {"input": user_content, "output": bot_content, "negative_response": current_negative_responses, "quality": quality_score}
                else: # 首次对话添加 system prompt
                    conversation_turn = {"system": system_prompt, "input": user_content, "output": bot_content, "negative_response": current_negative_responses, "quality": quality_score}

                conversation_turns.append(conversation_turn)
                negative_responses.extend(current_negative_responses) # 累积负样本，虽然当前例子未使用

    return conversation_turns

def construct_system_prompt(bot_prompt, bot_name): # 系统 prompt 构建函数 (与之前版本相同，此处省略)
    fact = bot_prompt.get("Fact", "角色背景信息缺失")
    head = bot_prompt.get("Head", "角色设定信息缺失")
    brief = bot_prompt.get("Brief", "角色简要描述信息缺失")
    original_background = bot_prompt.get("OriginalBackground", "欢迎来到与角色的对话")

    system_prompt = f"你现在扮演的是{bot_name}，一个{head}。\n角色背景设定: {original_background}。\n详细设定: Fact: {fact}, Head: {head}, Brief: {brief}。"
    return system_prompt


def calculate_quality_score_v2(star_rating, history_count, msg_type, turn_count):
    """
    更复杂的质量评分函数，综合考虑多个因素。

    :param star_rating: 星级评分 (int 或 None)
    :param history_count: 历史对话轮数 (int 或 None)
    :param msg_type: 消息类型 (string 或 None)
    :param turn_count: 当前对话在session中的轮数 (int)
    :return: quality score (float, 0.0 - 1.0)
    """

    # 权重 (可以根据需要调整)
    weight_star = 0.6      # 星级评分的权重 (最重要)
    weight_history = 0.1   # 历史对话轮数的权重
    weight_msg_type = 0.15  # 消息类型的权重
    weight_turn_count = 0.15 # 对话轮数的权重
    default_score = 0.5     # 默认质量分


    base_score = default_score # 默认基础分

    # 1. 星级评分影响
    if isinstance(star_rating, int) and 1 <= star_rating <= 4:
        base_score = star_rating / 4.0  # 归一化星级评分 (0.25 - 1.0)
    #  没有星级评分时，保持默认基础分


    # 2. 历史对话轮数影响 (history_count)
    history_bonus = 0
    if isinstance(history_count, int):
        history_bonus = min(history_count / 20.0, 0.2) # 历史对话轮数越多，略微增加分数, 但设置上限为 0.2
    base_score += history_bonus * weight_history


    # 3. 消息类型影响 (msg_type)
    msg_type_bonus = 0
    if msg_type == "normal":
        msg_type_bonus = 0.1  # "normal" 类型消息略微加分
    elif msg_type == "greeting":
        msg_type_bonus = -0.05 # "greeting" 消息略微减分 (可以根据实际情况调整)
    elif msg_type == "push":
        msg_type_bonus = -0.1 # "push" 消息减分更多 (可以根据实际情况调整)
    base_score += msg_type_bonus * weight_msg_type


    # 4. 对话轮次影响 (turn_count)
    turn_bonus = min(turn_count / 30.0, 0.15) # 对话轮数越多，略微增加分数, 但设置上限 0.15
    base_score += turn_bonus * weight_turn_count


    # 最终得分裁剪到 0-1 范围
    final_score = max(0.0, min(base_score * weight_star + (1-weight_star-weight_history-weight_msg_type-weight_turn_count)*default_score , 1.0)) # 保证得分在 0-1 之间, 并为剩余权重分配默认分

    return final_score


input_file_path = '/root/dataset/chaiting_data/session-02.jsonl' # 替换为您的输入文件路径
output_file_path = 'cleaned_session-02_complex_quality.jsonl' # 替换为您的输出文件路径
process_raw_data(input_file_path, output_file_path)
print(f"数据清洗完成，包含更复杂的质量评分，结果已保存到: {output_file_path}")

# # 加载清洗后的数据并随机展示几个对话示例
# def display_random_conversations(cleaned_file_path, num_conversations=2): #  默认展示 2 个对话，方便对比
#     cleaned_conversations = []
#     with jsonlines.open(cleaned_file_path, 'r') as reader:
#         for conversation_data in reader:
#             cleaned_conversations.append(conversation_data)

#     if not cleaned_conversations:
#         print("清洗后的对话数据为空，无法展示。")
#         return

#     num_to_display = min(num_conversations, len(cleaned_conversations))
#     random_indices = random.sample(range(len(cleaned_conversations)), num_to_display)

#     print(f"\n--- 随机展示 {num_to_display} 个包含负样本对比的可视化对话示例 ---") #  修改标题
#     for index in random_indices:
#         conversation = cleaned_conversations[index]['conversation']
#         print(f"\n--- 对话示例 {index + 1} ---")
#         for turn in conversation:
#             if "system" in turn:
#                 print(f"System: {turn['system']}")
#             print(f"User Input: {turn['input']}") #  更明确的标签

#             #  可视化对比展示正负样本
#             if turn.get("negative_response"):
#                 if turn["negative_response"]: # 确保负样本列表非空
#                     print(f"  {'Accepted Output:':<20} {turn['output']}") #  左对齐，更美观
#                     print(f"  {'Rejected Outputs:':<20}") #  左对齐
#                     for i, negative_response in enumerate(turn["negative_response"]):
#                         print(f"  {'':<20} {i+1}. {negative_response}") #  二级缩进对齐
#                     print(f"  {'Quality Score:':<20} {turn['quality']:.3f}") #  质量评分也左对齐
#             else: #  如果没有负样本，则按原格式展示
#                 print(f"  {'Output:':<20} {turn['output']}") # 左对齐 Output 标签
#                 print(f"  {'Quality Score:':<20} {turn['quality']:.3f}") # 质量评分也左对齐

#             print("-" * 40) #  更长的分隔线


def display_random_conversations(cleaned_file_path, num_conversations=2): 
    cleaned_conversations = []
    with jsonlines.open(cleaned_file_path, 'r') as reader:
        for conversation_data in reader:
            cleaned_conversations.append(conversation_data)

    if not cleaned_conversations:
        print("清洗后的对话数据为空，无法展示。")
        return

    #  过滤出包含负样本的对话
    negative_sample_conversations = [
        conv_data for conv_data in cleaned_conversations
        if any(turn.get("negative_response") and turn["negative_response"] for turn in conv_data['conversation']) # 检查任何一个 turn 是否有非空负样本列表
    ]

    if not negative_sample_conversations:
        print("清洗后的对话数据中没有包含负样本的对话，无法展示。") #  更明确的提示
        return

    num_to_display = min(num_conversations, len(negative_sample_conversations))
    random_indices = random.sample(range(len(negative_sample_conversations)), num_to_display)

    print(f"\n--- 随机展示 {num_to_display} 个 **包含负样本** 的可视化对话示例 ---") # 修改标题，强调负样本
    for index in random_indices:
        conversation = negative_sample_conversations[index]['conversation'] # 从过滤后的列表中取对话
        print(f"\n--- 对话示例 {index + 1} ---")
        for turn in conversation:
            if "system" in turn:
                print(f"System: {turn['system']}")
            print(f"User Input: {turn['input']}")

            if turn.get("negative_response"): #  这里仍然需要判断是否有负样本，因为可能只有部分 turn 有
                if turn["negative_response"]:
                    print(f"  {'Accepted Output:':<20} {turn['output']}")
                    print(f"  {'Rejected Outputs:':<20}")
                    for i, negative_response in enumerate(turn["negative_response"]):
                        print(f"  {'':<20} {i+1}. {negative_response}")
                    print(f"  {'Quality Score:':<20} {turn['quality']:.3f}")
            else: # 处理没有负样本的 turn (虽然在这个专门展示负样本的版本中，应该不会出现这种情况，但为了代码完整性保留)
                print(f"  {'Output:':<20} {turn['output']}")
                print(f"  {'Quality Score:':<20} {turn['quality']:.3f}")

            print("-" * 40)


cleaned_output_file_path = 'cleaned_session-02_complex_quality.jsonl' #  确保使用和清洗后保存的文件名一致
display_random_conversations(cleaned_output_file_path)
