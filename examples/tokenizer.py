from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-0.5B"

tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)

endoftext: dict[str, str|int] = {'token': '<|endoftext|>', 'id': 151643, 'description': 'end of text/sequence'}
im_start: dict[str, str|int] = {'token': '<|im_start|>', 'id': 151644, 'description': 'im start'}
im_end: dict[str, str|int] = {'token': '<|im_end|>', 'id': 151645, 'description': 'im end'}

fim_prefix: dict[str, str|int] = {'token': '<|fim_prefix|>', 'id': 151659, 'description': 'FIM prefix'}
fim_middle: dict[str, str|int] = {'token': '<|fim_middle|>', 'id': 151660, 'description': 'FIM middle'}
fim_suffix: dict[str, str|int] = {'token': '<|fim_suffix|>', 'id': 151661, 'description': 'FIM suffix'}
fim_pad: dict[str, str|int] = {'token': '<|fim_pad|>', 'id': 151662, 'description': 'FIM pad'}
repo_name: dict[str, str|int] = {'token': '<|repo_name|>', 'id': 151663, 'description': 'repository name'}
file_sep: dict[str, str|int] = {'token': '<|file_sep|>', 'id': 151664, 'description': 'file separator'}

# token: <|endoftext|>, id: 151643, input_id: 151643, decode_s: <|endoftext|>
# token: <|im_start|>, id: 151644, input_id: 151644, decode_s: <|im_start|>
# token: <|im_end|>, id: 151645, input_id: 151645, decode_s: <|im_end|>
# token: <|fim_prefix|>, id: 151659, input_id: 151659, decode_s: <|fim_prefix|>
# token: <|fim_middle|>, id: 151660, input_id: 151660, decode_s: <|fim_middle|>
# token: <|fim_suffix|>, id: 151661, input_id: 151661, decode_s: <|fim_suffix|>
# token: <|fim_pad|>, id: 151662, input_id: 151662, decode_s: <|fim_pad|>
# token: <|repo_name|>, id: 151663, input_id: 151663, decode_s: <|repo_name|>
# token: <|file_sep|>, id: 151664, input_id: 151664, decode_s: <|file_sep|>
special_tokens: list[dict[str, str|int]] = [
    endoftext,
    im_start,
    im_end,
    fim_prefix,
    fim_middle,
    fim_suffix,
    fim_pad,
    repo_name,
    file_sep
]

input_texts = [token['token'] for token in special_tokens]

for token in special_tokens:
    token_s: str = token['token']
    token_id: int = token['id']
    print(f"token: {token_s}, id: {token_id}, input_id: {tokenizer(token_s).input_ids[0]}, decode_s: {tokenizer.decode(token_id)}")
