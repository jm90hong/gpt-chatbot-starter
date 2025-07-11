import openai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# OpenAI 클라이언트 생성
client = openai.OpenAI(api_key="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")

# 로드
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.bin")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def search_context(question, top_k=3):
    q_embedding = model.encode([question])
    distances, indices = index.search(np.array(q_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def ask_gpt(context, question):
    prompt = f"""
다음 문서를 참고하여 질문에 답해주세요:

문서 내용:
{context}

질문:
{question}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# 콘솔 입력 루프
if __name__ == "__main__":
    print("📄 사내 문서 기반 GPT 챗봇 (종료하려면 'exit' 입력)\n")
    while True:
        question = input("질문 > ")
        if question.strip().lower() in ["exit", "quit"]:
            break
        context = "\n".join(search_context(question))
        answer = ask_gpt(context, question)
        print(f"\n�� 답변: {answer}\n")
