import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 전처리
trans = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
]) 

trainset = torchvision.datasets.ImageFolder(root = "CNNData", transform = trans)
testset = torchvision.datasets.ImageFolder(root = "CNNTData", transform = trans)

# 데이터 로더 설정
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=32, shuffle=False)

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # input channel: 1, output channel: 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # input channel: 32, output channel: 64
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.fc1 = nn.Linear(64 * 25 * 25, 128)  # Fully connected layer, 평면화 후 크기는 64 * 25 * 25
        self.fc2 = nn.Linear(128, 10)  # Output layer

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 25 * 25)  # 평면화, 크기는 64 * 25 * 25
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 초기화 및 GPU로 이동
model = CNN().to(device)

# 손실 함수와 최적화 기준 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 훈련
num_epochs = 3
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward 연산: 예측값 계산
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward 연산 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

print("훈련 완료!")

# 모델 평가
model.eval()  # 평가 모드로 설정
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'테스트 데이터의 정확도: {100 * correct / total}%')