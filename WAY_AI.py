import os
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import torchvision.transforms as transforms
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, Form, HTTPException
import pymongo

app = FastAPI()

gpt_key = os.environ.get('GPT-KEY')
mongo_client = os.environ.get("mongoURL")

def fetch_s3_object(url):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch S3 object")
    return response.raw

def CNN(image_stream):
    image_raw = Image.open(image_stream)
    image = image_raw.resize((550, 550))  # 이미지 크기 조정 예시
    try:
        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # input channel: 1, output channel: 32
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # input channel: 32, output channel: 64
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
                self.fc1 = nn.Linear(128 * 16 * 16, 128)  # Fully connected layer, 평면화 후 크기는 64 * 25 * 25
                self.fc2 = nn.Linear(128, 10)  # Output layer

            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = self.pool(torch.relu(self.conv3(x)))
                x = x.view(-1, 128 * 16 * 16)  # 평면화, 크기는 64 * 25 * 25
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        def preprocess_image(image_path):
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
            image = Image.open(image_path)     
            image = transform(image).unsqueeze(0)  
            return image

        # 모델 인스턴스 생성
        model = CNN()
        # 저장된 가중치 로드
        model.load_state_dict(torch.load('WAY_CNN_model_weights.pth'))
        # 모델을 평가 모드로 설정
        model.eval()
        # contents = await image.read()
        input_image = preprocess_image(image)
        # 추론 수행
        with torch.no_grad():
            output = model(input_image)
        _, predicted_class = torch.max(output, 1)
        
        classes = ['Chungbuk', 'Chungnam', 'Gangwon', 'Gyeonggi', 'Gyeongbuk', 'Gyeongnam', 'Jeonbuk', 'Jeonnam', 'Seoul']
        return classes[predicted_class.item()]
    
    except Exception as e:
        return str(e)

def NLP(post_stream):
    try:
        post_text = post_stream.read()
        
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
        
        # chromadb 버전 0.4.8 고정해두고 쓰자. 버전때문에 골치 좀 아팠다.
        texts = text_splitter.split_text(post_text)
        split = text_splitter.create_documents(texts)
        db = Chroma.from_documents(split, hf)
        
        # 벡터 디비의 결과와 함께 gpt에 전달.
        qa = RetrievalQA.from_chain_type(
                    llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0, openai_api_key=gpt_key),
                    chain_type = 'stuff',
                    retriever = db.as_retriever(
                                    search_type = 'mmr',
                                    search_kwargs = {'fetch_k':3}
                    ),
                    return_source_documents=False
        )

        query = '저장된 글의 분위기와 핵심 주제를 글을 세 단어로 정리해줘'

        result = qa(query)
        result_ = result["result"]
        values = result_.split(", ")
        return values
      
    except Exception as e:
        return {"error": str(e)}

def vectorization(list, region):
    plain_vector = list + [region]
    combined = ' '.join(plain_vector)
    
    vectorizer = TfidfVectorizer()
    
    tfidf_matrix = vectorizer.fit_transform([combined])
    tfidf_vector = tfidf_matrix.toarray()[0]
    
    return tfidf_vector

def mongo_insert(id, vector):
    client = pymongo.MongoClient(mongo_client)
    
    db = client["way"]
    collection = db["user_vector"]
    
    document = {
        "user_id": id,
        "vector_value": vector
    }
    
    collection.insert_one(document)

@app.post("/ai-service/user_tag")
def way_ai(user_id: int = Form(...), image_url: str = Form(...), text_url: str = Form(...)):
    
    text_stream = fetch_s3_object(text_url)
    way_tag = NLP(text_stream)
    
    image_stream = fetch_s3_object(image_url)
    region_tag = CNN(image_stream)
    
    vex = vectorization(way_tag, region_tag)
    
    mongo_insert(user_id, vex)

    return way_tag