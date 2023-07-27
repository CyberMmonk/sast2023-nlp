from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("model/chatglm2-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("model/chatglm2-6b-int4", trust_remote_code=True).half().cuda()
model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)

response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
