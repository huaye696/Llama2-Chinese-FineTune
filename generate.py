import sys
import torch
from peft import PeftModel
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument("--llm_path", type=str, default=False)  # 训练进度和结果的上报目标
parser.add_argument("--lora_path", type=str, default=False)  # 微调数据集目录
args = parser.parse_args()
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
model = LlamaForCausalLM.from_pretrained(
    args.llm_path,
    load_in_8bit=True,
    use_cache=False,
    device_map=device
)
model = PeftModel.from_pretrained(
    model,
    args.lora_path,
    torch_dtype=torch.float16,
)
model.eval()

generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4, # beam search
)
tokenizer = LlamaTokenizer.from_pretrained(args.llm_path)
tokenizer.pad_token_id = 0  # 为了区分EOStoken

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

def generate(input):
    prompt = generate_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        for generation_output in model.stream_generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            repetition_penalty=float(2.0),
        ):
            outputs = tokenizer.batch_decode(generation_output)
            show_text = "\n--------------------------------------------\n".join(
                [output.split("### Response:")[1].strip().replace('�','')+" ▌" for output in outputs]
            )
            yield show_text
        return outputs[0].split("### Response:")[1].strip().replace('�','')


while True:  # 创建一个无限循环
    user_input = input("Input you question：")  # 获取用户输入
    print(generate(user_input))