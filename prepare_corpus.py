from sentence_transformers import SentenceTransformer
import numpy as np
import json

# 加载预训练的句子嵌入模型
model = SentenceTransformer('AI-ModelScope/bge-small-zh-v1.5')

# 读取JSON文档库
with open('poetry_corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# 提取所有诗歌内容
all_poems = []
for category in corpus.values():
    for poem in category:
        all_poems.append(poem['content'])

# 向量化文档库中的每个诗歌内容
embeddings = model.encode(all_poems)

# 保存嵌入向量
np.save('poetry_embeddings.npy', embeddings)

print(f"文档向量化完成，共处理了 {len(all_poems)} 首诗，嵌入向量已保存至 poetry_embeddings.npy")