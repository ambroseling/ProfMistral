#https://github.com/brevdev/notebooks/blob/main/mistral-finetune-own-data.ipynb
import torch
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import Accelerator
from datetime import datetime
#loading dataset

accelerator = Accelerator(device_placement=True,mixed_precision="fp16",gradient_accumulation_steps=1)

model_path = "/home/tiny_ling/projects/my_mistral/Mistral-7B-v0.1"

train_dataset = load_dataset('json',data_files = 'train_data.jsonl',split="train")
val_dataset = load_dataset('json',data_files = 'val_data.jsonl',split="train")

max_length = 512

def format_prompt(example):
    return f"This is what the lecturer said: {example['text']}"

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16 
)

model = AutoModelForCausalLM.from_pretrained(model_path,quantization_config=bnb_config,device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side="left",add_eos_token=True,add_bos_token=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_prompt(prompt):
    result = tokenizer(format_prompt(prompt),truncation=True,max_length=max_length,padding="max_length")
    result['labels'] = result['input_ids'].copy()
    return result

tokenized_train_dataset = train_dataset.map(tokenize_prompt)
tokenized_val_dataset = val_dataset.map(tokenize_prompt)



# Init an eval tokenizer that doesn't add padding or eos token
eval_prompt = "What is APS105? "
eval_tokenizer = AutoTokenizer.from_pretrained(model_path,add_bos_token=True)
model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")
model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(model.generate(**model_input,max_new_tokens=256,repetition_penalty=1.15)[0]))

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
config = LoraConfig(r=32,lora_alpha=64,target_modules=[
"q_proj",
"k_proj",
"v_proj",
"o_proj",
'gate_proj',
'up_proj',
'down_proj',
'lm_head',
],
bias="none",
lora_dropout = 0.05,
task_type = "CAUSAL_LM",)

model = get_peft_model(model,config)
total_params = 0; trainable_params = 0
for param in model.parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
print(f"Number of trainable parameters: {total_params} | Number of trainable parameters: {trainable_params} | % of trainable: {(trainable_params/total_params)*100}%")
print(model)

model = accelerator.prepare_model(model)
project = "prof-finetune"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=1500,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=25,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=25,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()