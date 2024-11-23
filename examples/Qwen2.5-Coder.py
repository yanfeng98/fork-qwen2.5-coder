from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-Coder-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

input_text = "#write a quick sort algorithm"
model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024, do_sample=False)[0]
output_text = tokenizer.decode(generated_ids[len(model_inputs.input_ids[0]):], skip_special_tokens=True)

print(f"Prompt: {input_text}\n\nGenerated text: {output_text}")
