import os, sys
import os
import transformers
import torch
import datasets
import time
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    LlamaTokenizer,
    LlamaForCausalLM, BitsAndBytesConfig
)
import argparse
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)  # 训练进度和结果的上报目标
parser.add_argument("--data_path", type=str, default="None")  # 微调数据集目录
parser.add_argument("--output_path", type=str, default="None")  # 模型检查点输出目录
parser.add_argument("--save_path", type=str, default="None")  # 模型最终保存目录
parser.add_argument("--model_path", type=str, default="None")  # llama2的模型目录
parser.add_argument("--val_size", type=float, default=0.3)  # 验证集比例
parser.add_argument("--num_train_epochs", type=int, default=1)  # 验证集比例
args = parser.parse_args()

"""基本训练超参设置"""
batch_size = 128  # 批次大小
micro_batch_size = 16  # 每个GPU上的训练批量大小。
gradient_accumulation_steps = batch_size // micro_batch_size  # 梯度累积步数
lr = 3e-4  # 学习率
val_set_size = args.val_size  # 训练集大小
max_length = 1024  # 最大输入token长度
val_size = args.val_size

# 同时设置了这两个参数，训练将在任一条件首先满足时停止。
num_train_epochs  = args.num_train_epochs  # 遍历整个训练数据集的次数

"""lora参数设置"""
lora_r = 8
lora_alpha = 32  # 推荐r参数的四倍
lora_dropout = 0.05

"""文件路径及运行环境"""
data_path = args.data_path
out_dir = args.output_path
save_path = args.save_path
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))  # 获取当前的多机环境
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}  # 如果当前不止有一张卡，则修改 device_map
    gradient_accumulation_steps = gradient_accumulation_steps // world_size  # 需要配合多卡，切分梯度累积步数，batch会呗拆分到多张卡上

# 量化配置，加载模型
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = LlamaForCausalLM.from_pretrained(
    args.model_path,
    quantization_config=bnb_config,
    device_map=device_map,
)
model = prepare_model_for_kbit_training(model)  # 应用量化方案

tokenizer = LlamaTokenizer.from_pretrained(args.model_path, add_eos_token=True)  # 加载分词器
tokenizer.pad_token_id = 0  # 为了区分EOStoken

config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=['q_proj','k_proj','v_proj','o_proj'],  # 微调哪些线性层
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()  # 打印模型的参数

""" 提示词模板 """
prompt_template = {
    'prompt_input':"Below is an instruction that describes a task,"
                   "paired with an input that provides further context. "
                   "Write a response that appropriately completes the request."
                   "\n\n"
                   "### Instruction:"
                   "\n{instruction}"
                   "\n\n"
                   "### Input:"
                   "\n{input}"
                   "\n\n"
                   "### Response:\n",
    'prompt_not_input':"Below is an instruction that describes a task"
             "Write a response that appropriately completes the request."
             "\n\n"
             "### Instruction:"
             "\n{instruction}"
             "\n\n"
             "### Response:\n",
}

def generate_and_tokenize_prompt(data_item):
    """
    用于将数据生成提示词，并进行编码，数据集中，有些数据不带input，所以需要区分提示词
    :param data_item: 每一条数据
    :return: 一个字典，数据包括：input_ids，labels，attention_mask
    input_ids与labels的差别仅仅在于labels的提示词编码恒为-100，表示不计算损失
    """
    if data_item['input']:
        user_prompt = prompt_template['prompt_input'].format(instruction=data_item['instruction'],input=data_item['input'])
    else:
        user_prompt = prompt_template['prompt_not_input'].format(instruction=data_item['instruction'])
    len_user_prompt_tokens = len(tokenizer(user_prompt,truncation=True,max_length=max_length)["input_ids"]) - 1
    full_tokens = tokenizer(
        user_prompt + data_item["output"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
                  + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }

def filter_empty_rows(row):
    """
    规范化数据，去掉指令和输出为空和none的数据，否则会有编码错误
    :param row: 每条数据
    :return:
    """
    return (row['instruction'] is not None
            and row['instruction'] != ''
            and row['output'] is not None
            and row['output'] != '' )

dataset = datasets.load_dataset('csv', data_files=data_path)  # 读入数据
train_test_split = dataset['train'].train_test_split(test_size=val_size)  # 切分数据

train_dataset = train_test_split['train']
train_dataset = train_dataset.filter(filter_empty_rows)

test_dataset = train_test_split['test']
test_dataset = test_dataset.filter(filter_empty_rows)

cols = ["instruction", "input", "output"]
train_data = train_dataset.shuffle().map(generate_and_tokenize_prompt, remove_columns=cols)
val_data = test_dataset.shuffle().map(generate_and_tokenize_prompt, remove_columns=cols)

# 训练器
args=TrainingArguments(
    per_device_train_batch_size=micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    learning_rate=lr,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=32,
    output_dir=out_dir,
    save_total_limit=10,
    ddp_find_unused_parameters=False if ddp else None,
    report_to="wandb" if args.wandb else [],
    ignore_data_skip=False,
)  # 定义训练参数

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False

start_time = time.time()
trainer.train()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time}秒")

model.save_pretrained(save_path)