import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn

# 모델 클래스 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.fc1 = nn.Linear(128 * 16 * 16, 128)  
        self.fc2 = nn.Linear(128, 10)  

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)  
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
model.load_state_dict(torch.load('model_weights.pth'))

# 모델을 평가 모드로 설정
model.eval()

image_path = 'CNNTData/Chungcheongbuk/6.jpg'

input_image = preprocess_image(image_path)
classes = ['Chungbuk', 'Chungnam', 'Gangwon', 'Gyeonggi', 'Gyeongbuk', 'Gyeongnam', 'Jeonbuk', 'Jeonnam', 'Seoul']
# 추론 수행
with torch.no_grad():
    output = model(input_image)

_, predicted_class = torch.max(output, 1)
print("Predicted Class:", classes[predicted_class.item()])
