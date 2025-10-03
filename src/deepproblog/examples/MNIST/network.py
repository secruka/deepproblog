import torch.nn as nn

#2つの**畳み込み層（nn.Conv2d）とプーリング層（nn.MaxPool2d）**を交互に重ねています。
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        return x

#特徴ベクトルを受け取り、どの数字（0〜9）かを分類する部分
class MNIST_Classifier(nn.Module):
    def __init__(self, n=10, with_softmax=True):
        super(MNIST_Classifier, self).__init__()
        self.with_softmax = with_softmax
        if with_softmax:
            self.softmax = nn.Softmax(1)
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n),
        )

    def forward(self, x):
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x.squeeze(0)

# 上記のMNIST_CNN（特徴抽出器）とMNIST_Classifier（分類器）を統合した、完全なニューラルネットワークモデルです。
class MNIST_Net(nn.Module):
    def __init__(self, n=10, with_softmax=True, size=16 * 4 * 4):
        super(MNIST_Net, self).__init__()
        self.with_softmax = with_softmax
        self.size = size
        if with_softmax:
            if n == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(1)
        #特徴抽出
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        #分類
        self.classifier = nn.Sequential(
            nn.Linear(size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x
