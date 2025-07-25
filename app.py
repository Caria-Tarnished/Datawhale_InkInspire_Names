import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import random
import re
import json

def get_models():
    tokenizer = AutoTokenizer.from_pretrained('IEITYuan/Yuan2-2B-Mars-hf/', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('IEITYuan/Yuan2-2B-Mars-hf/', trust_remote_code=True).to('cuda')
    embedding_model = SentenceTransformer('AI-ModelScope/bge-small-zh-v1.5').to('cuda')
    document_embeddings = np.load('poetry_embeddings.npy')
    with open('poetry_corpus.json', 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    return tokenizer, model, embedding_model, document_embeddings, corpus

def retrieve_relevant_documents(query, embedding_model, document_embeddings, corpus, top_k=10):
    query_embedding = embedding_model.encode([query])
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(document_embeddings)
    _, indices = index.search(query_embedding, top_k)
    
    all_poems = []
    for category in corpus.values():
        all_poems.extend(category)
    
    relevant_docs = [all_poems[i] for i in indices[0] if i < len(all_poems)]
    return relevant_docs

def extract_name_suggestions(docs):
    suggestions = []
    for doc in docs:
        if '适合作名字的字词' in doc:
            suggestions.extend(doc['适合作名字的字词'].split('、'))
    return list(set(suggestions))

def generate_content(tokenizer, model, prompt, max_retries=2):
    for _ in range(max_retries):
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
            outputs = model.generate(inputs["input_ids"], max_length=1024, num_return_sequences=1, temperature=0.7)
            content = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return content
        except Exception as e:
            print(f"Error in generate_content: {e}")
    return None

def remove_eod(text):
    return re.sub(r'<eo.*?>', '', text).strip()

def generate_names(tokenizer, model, surname, gender, characteristics, style, relevant_poems, name_suggestions):
    meaningful_words = set()
    for poem in relevant_poems:
        words = re.findall(r'[\u4e00-\u9fa5]{1,2}', poem['content'])
        meaningful_words.update(words)
    
    prompt = f"""为姓{surname}的{gender}创作2个富有诗意的名字。名字应体现：{characteristics}。诗词风格：{style}。
要求：每个名字2个字，来自以下参考内容。

参考：
{' '.join([poem['content'] for poem in relevant_poems[:2]])}
{', '.join(list(meaningful_words)[:5])}
{', '.join(name_suggestions[:3])}

请以"姓名1：[名]，姓名2：[名]"的格式给出："""

    response = generate_content(tokenizer, model, prompt)
    if not response:
        return []
    names = re.findall(r'姓名\d：([\u4e00-\u9fa5]{2})', response)
    return names[:2]

def generate_meaning(tokenizer, model, surname, name, gender, characteristics, style):
    prompt = f"""解释{surname}{name}这个{gender}名字的含义。体现：{characteristics}。风格：{style}。
要求：简洁、有创意，不超过20字。

请给出名字含义："""

    response = generate_content(tokenizer, model, prompt)
    return remove_eod(response) if response else "未能生成含义"

def app(surname, gender, characteristics, style):
    tokenizer, model, embedding_model, document_embeddings, corpus = get_models()
    
    query = f"{gender} {characteristics} {style}"
    relevant_poems = retrieve_relevant_documents(query, embedding_model, document_embeddings, corpus)
    name_suggestions = extract_name_suggestions(relevant_poems)
    
    names = generate_names(tokenizer, model, surname, gender, characteristics, style, relevant_poems, name_suggestions)
    
    meanings = []
    for name in names:
        full_name = surname + name
        meaning = generate_meaning(tokenizer, model, surname, name, gender, characteristics, style)
        meanings.append((full_name, meaning))
    
    if names:
        referenced_poems = [
            f"{poem['title']}·{poem['author']}：{poem['content']}" for poem in relevant_poems[:2]
        ]
        return (
            ', '.join([surname + name for name in names]),
            meanings,
            '\n\n'.join(referenced_poems),
            ", ".join(name_suggestions[:5])
        )
    else:
        return ("名字生成失败，请重试。", [], "", "")

iface = gr.Interface(
    fn=app,
    inputs=[
        gr.Textbox(label="请输入宝宝的姓氏"),
        gr.Radio(["男孩", "女孩"], label="宝宝的性别"),
        gr.Textbox(label="您希望名字体现哪些品质或意境？"),
        gr.Textbox(label="您偏好的诗词意境"),
    ],
    outputs=[
        gr.Textbox(label="为您生成的名字"),
        gr.JSON(label="名字含义"),
        gr.Textbox(label="参考诗句原文"),
        gr.Textbox(label="建议的名字字词")
    ],
    title="📜 文墨启名",
    description="通过《诗经》、《楚辞》、《唐诗三百首》、《宋词》等古典诗词，为宝宝取一个有文化底蕴的名字。",
    article="""
    注：本工具仅供参考，最终取名请结合您个人喜好和文化背景。
    """
)

if __name__ == "__main__":
    iface.launch()