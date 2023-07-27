import torch
from transformers import AutoTokenizer, BertForMaskedLM
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("model/bert-base-chinese")
model = BertForMaskedLM.from_pretrained("model/bert-base-chinese", trust_remote_code=True).eval()

def get_embedding(word: str):
    embedding = model.bert.embeddings.word_embeddings(tokenizer.encode(word, return_tensors="pt"))
    return embedding.mean(dim=list(range(embedding.dim() - 1)))


def get_embedding_by_id(idx: int):
    embedding = model.bert.embeddings.word_embeddings(torch.tensor(idx))
    return embedding.mean(dim=list(range(embedding.dim() - 1)))


def cal_similarity(x, y):
    similarity = (x * y).sum() / ((y * y).sum() ** 0.5) / ((x * x).sum() ** 0.5)
    return similarity

with torch.no_grad():
    a = get_embedding("欧盟")
    b = get_embedding("英国")
    c = get_embedding("法国")
    d = get_embedding("德国")

    choices_str = ["俄罗斯", "欧盟", "波兰", "美国"]
    choices_embeddings = [get_embedding(choice_str) for choice_str in choices_str]

    x = a - b - c - d

    similarities = [cal_similarity(x, answers_embedding) for answers_embedding in choices_embeddings]

    max_similarity = max(similarities)
    max_similarity_str = choices_str[similarities.index(max_similarity)]

    print(list(zip(choices_str, similarities)))

    print(max_similarity_str , max_similarity)

