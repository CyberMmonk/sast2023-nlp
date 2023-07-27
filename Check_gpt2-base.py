from peft import PeftModelForSeq2SeqLM
from transformers import AutoTokenizer, GenerationConfig
from gpt2.modeling_gpt2 import GPT2LMHeadModel
# you have to fill [MASK] in gpt2/modeling_gpt2.py first

model = GPT2LMHeadModel.from_pretrained("model/gpt2-base").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("model/gpt2-base")

generation_config = GenerationConfig(max_new_tokens=32, use_cache=True, num_beams=1, temperature=0.7, pad_token_id=tokenizer.eos_token_id)

context = ""


inputs = tokenizer("Hello?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, generation_config=generation_config)
response = tokenizer.batch_decode(outputs)[0]
print("gpt2-base: ", response)

# Write a + b problem in python.