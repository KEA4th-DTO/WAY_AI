import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 회귀 스플리터
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 0
)

# 임베딩 모델(허킹페이스)
model_name='jhgan/ko-sbert-nli'
model_kwargs={'device':'cpu'}
encode_kwargs={'normalize_embeddings':True}

hf = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
)

# 블로그 글 텍스트파일 읽기
with open('/Users/jangminseong/Downloads/zindex_all_content.txt') as f:
  text = f.read()

# chromadb 버전 0.4.8 고정해두고 쓰자. 버전때문에 골치 좀 아팠다.
texts = text_splitter.split_text(text)
split = text_splitter.create_documents(texts)
db = Chroma.from_documents(split, hf)

# 벡터 디비의 결과와 함께 gpt에 전달.
qa = RetrievalQA.from_chain_type(
                    llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0),
                    chain_type = 'stuff',
                    retriever = db.as_retriever(
                                    search_type = 'mmr',
                                    search_kwargs = {'fetch_k':3}
                    ),
                    return_source_documents=False
)

query='이 글의 분위기와 핵심 주제를 각각 세 단어로 정리해줘'

result = qa(query)
print(result['result'])