import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/postNLP")
async def upload_file(file: UploadFile = File(...)):
    try:
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
        contents = await file.read()
        
        # 텍스트로 디코딩하여 처리
        text = contents.decode("utf-8")
        
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

        query='이 글의 분위기와 핵심 주제를 고려하여 글을 세 단어로 정리해줘'

        result = qa(query)
        
        return {"GPT Result": result}
      
    except Exception as e:
        return {"error": str(e)}
    
    
@app.post("/mymapCNN")
async def process_image(image: UploadFile = File(...)):
    
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

    contents = await image.read()

    input_image = preprocess_image(contents)
    classes = ['Chungbuk', 'Chungnam', 'Gangwon', 'Gyeonggi', 'Gyeongbuk', 'Gyeongnam', 'Jeonbuk', 'Jeonnam', 'Seoul']

    # 추론 수행
    with torch.no_grad():
        output = model(input_image)
        
    _, predicted_class = torch.max(output, 1)
    
    return classes[predicted_class.item()]