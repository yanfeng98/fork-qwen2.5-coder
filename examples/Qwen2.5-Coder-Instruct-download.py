from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer.save_pretrained(f"./Qwen2-5-Coder-0-5B-Instruct")
model.save_pretrained(f"./Qwen2-5-Coder-0-5B-Instruct")

prompt = "write a quick sort algorithm."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
