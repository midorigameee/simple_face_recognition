import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms, datasets
import os


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# Hyper parameters
NUM_EPOCHS = 100
IMAGE_SIZE = 32
BATCH_SIZE = 10
DATASET_DIR = "training_data"


"""
32*32のRGB画像に対応しているCNN

input(3, 32, 32)
conv1(3, 6, 5)   => (6, 28, 28)
pool1(2, 2)      => (6, 14, 14)
conv2(6, 16, 5)  => (16, 10, 10)
pool2(2, 2)      => (16, 5, 5)

16 * 5 * 5 = 400 => 120
120 => 84
84 => 4

この構造はLeNetと呼ばれるCNNの初期モデル
"""
class CNN(nn.Module):
    def __init__(self, num_class):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # 取り込んだデータに施す処理を指定
    data_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

    # データセットの読み込み
    full_dataset = datasets.ImageFolder(root=DATASET_DIR,
                                        transform=data_transform)

    # 訓練用とテスト用に分割
    train_size = int(0.8 * len(full_dataset))  
    test_size = len(full_dataset) - train_size  
    train_dataset, test_dataset = torch.utils.data.random_split(
                                        dataset=full_dataset,
                                        lengths=[train_size, test_size])

    # DataLoader化
    train_loader = torch.utils.data.DataLoader(
                                        dataset=train_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                                        dataset=test_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

    # モデル、損失関数、最適化関数の定義
    num_class = len(os.listdir(DATASET_DIR))    # class数を取得
    model = CNN(num_class).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # モデルのトレーニング
    total_step = len(train_loader)
    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 5 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))

    # モデルのテスト
    model.eval()  # ネットワークを推論モードに切り替える
    with torch.no_grad():   # 推論中は勾配の保存を止める
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy: {:.3f} %' .format(100 * correct / total))

    # モデルの保存
    model_name = "model_" + DATASET_DIR + ".ckpt"
    torch.save(model.state_dict(), 'model.ckpt')


if __name__ == "__main__":
    main()