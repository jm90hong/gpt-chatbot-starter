# 📄 사내 문서 기반 GPT 챗봇

이 프로젝트는 PDF 문서를 기반으로 한 지능형 챗봇 시스템입니다. 문서의 내용을 벡터화하여 저장하고, 사용자의 질문에 대해 관련된 문서 내용을 참고하여 답변을 제공합니다.

## 🚀 주요 기능

- **PDF 문서 처리**: PyMuPDF를 사용하여 PDF 파일에서 텍스트 추출
- **텍스트 청킹**: 긴 문서를 의미 있는 단위로 분할
- **벡터 임베딩**: Sentence Transformers를 사용하여 텍스트를 벡터로 변환
- **FAISS 검색**: 고속 벡터 검색을 통한 관련 문서 찾기
- **GPT-4 통합**: OpenAI GPT-4를 사용한 자연어 답변 생성

## 📋 요구사항

```bash
pip install openai sentence-transformers faiss-cpu PyMuPDF numpy tf-keras
```

## 🛠️ 설치 및 설정

1. **의존성 설치**:
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenAI API 키 설정**:
   `chatbot.py` 파일에서 `openai_api_key` 부분을 실제 API 키로 변경하세요.

3. **문서 처리**:
   ```bash
   python embbeding.py
   ```

4. **챗봇 실행**:
   ```bash
   python chatbot.py
   ```

## 📁 파일 구조

```
gpt-ai/
├── README.md           # 프로젝트 설명서
├── chatbot.py          # 메인 챗봇 애플리케이션
├── embbeding.py        # 문서 벡터화 처리
├── document.pdf        # 처리할 PDF 문서
├── faiss_index.bin     # FAISS 벡터 인덱스
├── chunks.pkl          # 텍스트 청크 저장
└── requirements.txt    # 의존성 목록
```

## 🔧 사용법

### 1. 문서 벡터화
먼저 PDF 문서를 벡터화하여 검색 가능한 형태로 변환합니다:

```bash
python embbeding.py
```

### 2. 챗봇 실행
벡터화가 완료되면 챗봇을 실행할 수 있습니다:

```bash
python chatbot.py
```

### 3. 질문하기
챗봇이 실행되면 문서 내용에 대한 질문을 할 수 있습니다:

```
질문 > 회사 정책에 대해 알려주세요
💬 답변: [GPT-4가 생성한 답변]

질문 > exit
```

## ⚙️ 설정 옵션

### 텍스트 청킹 설정
`embbeding.py`에서 `max_length` 매개변수를 조정하여 청크 크기를 변경할 수 있습니다:

```python
def split_text(text, max_length=500):  # 기본값: 500자
```

### 검색 결과 수 조정
`chatbot.py`에서 `top_k` 매개변수를 조정하여 검색 결과 수를 변경할 수 있습니다:

```python
def search_context(question, top_k=3):  # 기본값: 3개
```

## 🔍 작동 원리

1. **문서 처리**: PDF 파일에서 텍스트를 추출하고 의미 있는 청크로 분할
2. **벡터화**: 각 청크를 Sentence Transformers 모델을 사용하여 벡터로 변환
3. **인덱싱**: FAISS를 사용하여 벡터를 고속 검색 가능한 인덱스로 저장
4. **질문 처리**: 사용자 질문을 벡터로 변환하고 유사한 문서 청크 검색
5. **답변 생성**: 검색된 관련 문서를 컨텍스트로 사용하여 GPT-4가 답변 생성

## 🛡️ 보안 주의사항

- OpenAI API 키를 코드에 직접 하드코딩하지 마세요
- 환경 변수나 별도 설정 파일을 사용하여 API 키를 관리하세요
- 민감한 문서를 처리할 때는 적절한 보안 조치를 취하세요

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

버그 리포트나 기능 제안은 이슈를 통해 제출해 주세요.
