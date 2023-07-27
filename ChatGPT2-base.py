from peft import PeftModelForSeq2SeqLM
from transformers import AutoTokenizer, GPT2LMHeadModel, GenerationConfig

# you have to train your own chatgpt2-base by running Supervised_Fine-Tuning.py

model = GPT2LMHeadModel.from_pretrained("model/gpt2-base")
model = PeftModelForSeq2SeqLM.from_pretrained(model, "model/chatgpt2-base").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("model/gpt2-base")

generation_config = GenerationConfig(max_new_tokens=128, use_cache=True, num_beams=1, temperature=0.7, pad_token_id=tokenizer.eos_token_id)

context = ""


while True:
    human_input = input("Human: ")
    context += "Human: " + human_input + "\n" + "chatgpt2-base: "
    inputs = tokenizer(human_input, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, generation_config=generation_config)
    response = tokenizer.batch_decode(outputs)[0]
    print("chatgpt2-base: ", response)
    context += response + "\n"

# Write a + b problem in python.