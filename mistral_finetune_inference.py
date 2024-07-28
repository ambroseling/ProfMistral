from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "/home/tiny_ling/projects/my_mistral/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
)

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
ft_model = PeftModel.from_pretrained(base_model, "/home/tiny_ling/projects/my_mistral/mistral-prof-finetune/checkpoint-1500")

# Init an eval tokenizer that doesn't add padding or eos token
eval_prompt = "Come on, there we go. All right, welcome back to, all right, welcome back to Computer Fundamentals. Thank you for joining me again today."
model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")
ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(ft_model.generate(**model_input,max_new_tokens=256,repetition_penalty=1.15)[0]))
