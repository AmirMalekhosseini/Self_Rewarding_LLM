from peft import LoraConfig, get_peft_model, TaskType
def add_lora(model, r=16, alpha=32, dropout=0.05):
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
