import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MAX_NEW_TOKENS = 128
SYSTEM_PROMPT = "You are a helpful assistant."


def configure_generation(model, tokenizer):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = []

    for token_id in [im_end_id, tokenizer.eos_token_id]:
        if token_id is not None and token_id not in eos_ids:
            eos_ids.append(token_id)

    if not eos_ids:
        raise ValueError("No valid eos_token_id found.")

    model.generation_config.do_sample = False
    model.generation_config.eos_token_id = eos_ids
    model.generation_config.pad_token_id = tokenizer.eos_token_id or eos_ids[-1]
    model.generation_config.repetition_penalty = 1.05
    model.generation_config.max_new_tokens = MAX_NEW_TOKENS


def chat_loop(model, tokenizer, device):
    model.eval()
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    i = 0
    while True:
        question = input("User:\n").strip()
        if not question:
            print("输入为空，退出。")
            break

        print()
        history.append({"role": "user", "content": question})

        input_text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([input_text], return_tensors="pt").to(device)

        if model_inputs.input_ids.size(1) > 32000:
            print("超过模型上下文长度，退出。")
            break

        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=model.generation_config.pad_token_id,
                repetition_penalty=model.generation_config.repetition_penalty,
                use_cache=True,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print("Assistant:\n")
        print(response)
        print("--------------------\n")
        history.append({"role": "assistant", "content": response})

        #测试两轮对话后退出
        i += 1
        if i >= 10:
            print("测试完成，退出。")
            break


def t1():
    print("测试LoRA适配器...")
    device = "cuda"
    adapter_path = Path(__file__).resolve().parent / "models" / "Qwen2.5-0.5B-SFT-falsesample-r16"
    adapter_config_path = adapter_path / "adapter_config.json"

    with adapter_config_path.open("r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    base_model_path = adapter_config["base_model_name_or_path"]
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)

    configure_generation(model, tokenizer)
    chat_loop(model, tokenizer, device)


def t2():
    print("测试合并后的模型...")
    device = "cuda"
    model_path = Path(__file__).resolve().parent / "models" / "Qwen2.5-0.5B-SFT-merged"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    configure_generation(model, tokenizer)
    chat_loop(model, tokenizer, device)


def t3():
    print("测试基础模型...")
    device = "cuda"
    model_path = r"C:\Users\lihaodong\.cache\modelscope\hub\models\Qwen\Qwen2___5-0___5B"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    configure_generation(model, tokenizer)
    chat_loop(model, tokenizer, device)


if __name__ == "__main__":
    t1()  # LoRA adapter
    # t2()  # merged model
    # t3()  # base model
