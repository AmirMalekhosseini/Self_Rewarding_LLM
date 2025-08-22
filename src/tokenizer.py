from transformers import AutoTokenizer

def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name, use_fast=False)

    # Set the chat template manually
    tok.chat_template = """<|begin_of_text|>
{%- for message in messages %}
{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
{{- message['content'] + '<|eot_id|>' }}
{%- endfor %}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
"""
    tok.eos_token = "<|eot_id|>"
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    return tok

def to_chat(tokenizer, system, user, assistant=None):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    return tokenizer.apply_chat_template(messages, tokenize=False)

