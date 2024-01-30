import sys
import torch
from peft import PeftModel
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument("--llm_path", type=str, default="../BigFile/model/Llama_Model")  # 训练进度和结果的上报目标
parser.add_argument("--lora_path", type=str, default="../BigFile/save/model-save-2GPU")  # 微调数据集目录
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
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
    max_new_tokens=256, # max_length=max_new_tokens+input_sequence
    min_new_tokens=1, # min_length=min_new_tokens+input_sequence,
    do_sample=True
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

def generate():
    while True:  # 创建一个无限循环
        user_input = input("Input you question：\n")
        prompt = generate_prompt(user_input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                repetition_penalty=1.3,
            )
            output = generation_output.sequences[0]
            output = tokenizer.decode(output).split("### Response:")[1].strip()
            print(output)

generate()