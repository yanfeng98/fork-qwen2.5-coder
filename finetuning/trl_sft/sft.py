"""
# regular:
python sft.py \
    --model_name_or_path="../model/Meta-Llama-3___1-8B-Instruct" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --max_seq_length=2048 \
    --bf16=True \
    --output_dir="full_alpaca_gpt4_data_zh" \
    --logging_steps=20 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --gradient_checkpointing

# peft:
python sft.py \
    --model_name_or_path="../model/Meta-Llama-3___1-8B-Instruct" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --max_seq_length=2048 \
    --bf16=True \
    --output_dir="lora_alpaca_gpt4_data_zh" \
    --logging_steps=20 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16

# deepspeed
accelerate launch --config_file=accelerate_configs/deepspeed_zero{1,2,3}.yaml --num_processes {NUM_GPUS} path_to_your_script.py --all_arguments_of_the_script

accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml --num_processes 8 sft.py \
    --model_name_or_path="../model/Meta-Llama-3___1-8B-Instruct" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --max_seq_length=2048 \
    --bf16=True \
    --output_dir="full_alpaca_gpt4_data_zh" \
    --logging_steps=20 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --gradient_checkpointing
"""

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    SFTConfig,
    SFTScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

def formatting_prompts_func(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        response = examples["output"][i]

        if len(input_text) >= 1:
            text = f'''
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
### Instruction:
{instruction}
            
### Input:
{input_text}
            
### Response:
{response}
'''
        else:
            text = f'''
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
### Instruction:
{instruction}
            
### Response:
{response}
'''
        output_text.append(text.strip())

    return output_text


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name)
    dataset = load_dataset("json", data_files="./alpaca_gpt4_data_zh.json")
    print(dataset)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        # eval_dataset=dataset[script_args.dataset_test_split],
        formatting_func=formatting_prompts_func,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)