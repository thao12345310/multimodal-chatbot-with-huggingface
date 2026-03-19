import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig   # TRL 0.24: use SFTConfig, not TrainingArguments
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto"  # dtype= (not torch_dtype=)
)
model.enable_input_require_grads()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files={
    "train": "data/train.jsonl",
    "validation": "data/valid.jsonl",
})

# TRL 0.24: SFTConfig replaces TrainingArguments.
# dataset_text_field → SFTConfig  (removed from SFTTrainer)
# max_seq_length     → max_length in SFTConfig  (renamed)
args = SFTConfig(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),   # False on MPS (Apple Silicon)
    logging_steps=20,
    eval_strategy="steps", eval_steps=200,
    save_steps=200, load_best_model_at_end=True,
    report_to="none",
    dataset_text_field="text",        # lives in SFTConfig, not SFTTrainer
    max_length=2048,                  # renamed: max_seq_length → max_length
)

# TRL 0.24: SFTTrainer only accepts: model, args, datasets, processing_class, etc.
# NO dataset_text_field / max_seq_length here — those are in SFTConfig above.
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,       # replaces tokenizer=
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()
trainer.save_model("./outputs/final")
print("Training complete! Model saved to ./outputs/final")