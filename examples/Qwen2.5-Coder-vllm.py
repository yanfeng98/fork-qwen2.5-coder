from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_name = "Qwen/Qwen2.5-Coder-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

eos_token_ids = [151659, 151660, 151661, 151662, 151663, 151664, 151645, 151643]
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=1024, stop_token_ids=eos_token_ids)

llm = LLM(model=model_name)

prompt = "#write a quick sort algorithm.\ndef quick_sort("

outputs = llm.generate([prompt], sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print(f"Prompt:\n\n{prompt}\n\nGenerated text:\n\n{generated_text}")