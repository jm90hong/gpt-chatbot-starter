import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

def extract_text_from_pdf(pdf_path):
    """PDF 파일에서 텍스트를 추출합니다."""
    if not os.path.exists(pdf_path):
        print(f"❌ PDF 파일 '{pdf_path}'을 찾을 수 없습니다.")
        return None
        
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def split_text(text, max_length=500):
    """텍스트를 의미 있는 청크로 분할합니다."""
    sentences = text.split('\n')
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_length:
            chunk += sentence + " "
        else:
            if chunk.strip():
                chunks.append(chunk.strip())
            chunk = sentence + " "
    if chunk.strip():
        chunks.append(chunk.strip())
    return chunks

def build_faiss_index(pdf_path):
    """PDF 파일을 처리하여 FAISS 인덱스를 생성합니다."""
    print("📄 PDF 문서를 벡터화하는 중...")
    
    # 텍스트 추출
    text = extract_text_from_pdf(pdf_path)
    if text is None:
        return
    
    # 텍스트 분할
    chunks = split_text(text)
    print(f"✅ {len(chunks)}개 청크로 분할 완료")
    
    # 벡터 임베딩 생성
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    
    # FAISS 인덱스 생성
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    # 파일 저장
    faiss.write_index(index, "faiss_index.bin")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print("✅ 문서 벡터화 및 저장 완료!")

if __name__ == "__main__":
    build_faiss_index("document.pdf")
