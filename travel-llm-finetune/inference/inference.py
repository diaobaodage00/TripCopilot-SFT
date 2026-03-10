import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "outputs/qwen_travel_lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

def generate_response(instruction, max_new_tokens=512):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    user_input = "帮我规划一个新加坡3日旅行，预算3000元，喜欢美食和城市观光，不要太赶。"
    result = generate_response(user_input)
    print(result)