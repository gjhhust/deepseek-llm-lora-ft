"""
deepseek微调思路整理(自己的代码)
1、加载模型+分词器
2、处理数据集
3、设置lora参数
4、设置训练参数
5、设置SwanLab可视化工具
6、设置训练器参数+训练
7、保存模型
"""

### 1、加载模型+分词器
from transformers import AutoTokenizer,AutoModelForCausalLM,DataCollatorForSeq2Seq
import torch

# 加载模型
model_path = "/root/model/deepseek-ai/deepseek-llm-7b-base"
model_kwargs = {
        "torch_dtype": torch.float16,
        "use_cache": True,
        "device_map":"auto"
    }
model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

print("模型：",model)
print("分词器：",tokenizer)

# import pdb;pdb.set_trace()
### 2、处理数据集
import pandas as pd
from datasets import Dataset

data_path = "/root/cleaned_session-02_complex_quality.jsonl"
data = pd.read_json(data_path, lines=True)
train_ds = Dataset.from_pandas(data)
print(train_ds)

def process_data(data, tokenizer, max_seq_length):
    input_ids, attention_mask, labels = [], [], []

    conversations = data["conversation"]
    for i,conv in enumerate(conversations):

        if "instruction" in conv:
            instruction_text = conv['instruction']
        else:
            instruction_text = ""
        human_text = conv["input"]
        assistant_text = conv["output"]

        input_text = f"{tokenizer.bos_token}{instruction_text}\n\nUser:{human_text}\n\nAssistant:"

        input_tokenizer = tokenizer(
            input_text,
            add_special_tokens=False,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        output_tokenizer = tokenizer(
            assistant_text,
            add_special_tokens=False,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        input_ids += (
                input_tokenizer["input_ids"] + output_tokenizer["input_ids"] + [tokenizer.eos_token_id]
        )
        attention_mask += input_tokenizer["attention_mask"] + output_tokenizer["attention_mask"] + [1]
        labels += ([-100] * len(input_tokenizer["input_ids"]) + output_tokenizer["input_ids"] + [tokenizer.eos_token_id]
                   )

    if len(input_ids) > max_seq_length:  # 做一个截断
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels = labels[:max_seq_length]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

train_dataset = train_ds.map(process_data,
                             fn_kwargs={"tokenizer": tokenizer, "max_seq_length": tokenizer.model_max_length},
                             remove_columns=train_ds.column_names)

# 数据整理
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

### 3、设置lora参数
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=['up_proj', 'gate_proj', 'q_proj', 'o_proj', 'down_proj', 'v_proj', 'k_proj'],
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False  # 训练模式
    )

### 4、设置训练参数
from transformers import TrainingArguments

# 输出地址
output_dir="./output/deepseek-mutil-test"
# 配置训练参数
train_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=1,
    num_train_epochs=3,
    save_steps=5000,
    learning_rate=2e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to=None,
    seed=42,
    optim="adamw_torch",
    fp16=True,
    bf16=False,
    remove_unused_columns=False,
)

### 5、设置可视化工具
from swanlab.integration.transformers import SwanLabCallback
import os

os.environ["SWANLAB_API_HOST"] = "https://swanlab.115.zone/api"
os.environ["SWANLAB_WEB_HOST"] = "https://swanlab.115.zone"
swanlab_config = {
        "dataset": data_path,
        "peft":"lora"
    }
swanlab_callback = SwanLabCallback(
    project="deepseek-finetune-test",
    experiment_name="first-test",
    description="微调多轮对话",
    workspace=None,
    config=swanlab_config,
)

### 6、设置训练器参数+训练
from peft import get_peft_model
from transformers import Trainer

# 用于确保模型的词嵌入层参与训练
model.enable_input_require_grads()
# 应用 PEFT 配置到模型
model = get_peft_model(model,lora_config)
model.print_trainable_parameters()

# 配置训练器
trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[swanlab_callback],
        )
# 启动训练
trainer.train()

### 7、保存模型
from os.path import join

final_save_path = join(output_dir)
trainer.save_model(final_save_path)
