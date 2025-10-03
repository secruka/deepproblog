# このモデルのいいところは、CNN,言語翻訳機能、論理部としてしっかりと機能が分かれているところ
# だからこそこれらを上手く変えれば、作りたいモデルが作れるのではないだろうか。

from json import dumps

import torch
#各ディレクトリに詳細を説明したものをコメントしている
#変換器　ミニバッチ形式にデータを変換
from deepproblog.dataset import DataLoader
#推理エンジン
from deepproblog.engines import ApproximateEngine, ExactEngine
#結果を評価ツール
from deepproblog.evaluate import get_confusion_matrix
#変換器　画像データを、足し算タスク用のクエリ形式に変換
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition
#CNN
from deepproblog.examples.MNIST.network import MNIST_Net
#CNN 単体の機能というより色々な情報を統合をする
from deepproblog.model import Model
#変換器 CNNをdeepproblogで扱える形に変換する
from deepproblog.network import Network
#CNN* 作られたモデルを学習させる
from deepproblog.train import train_model


method = "exact" #厳密な推論エンジン
N = 1 #桁数

name = "addition_{}_{}".format(method, N)

train_set = addition(N, "train") #refer to init.py
test_set = addition(N, "test")

network = MNIST_Net()

pretrain = 0 
#if there is pretrained model, we have to load
if pretrain is not None and pretrain > 0: 
    network.load_state_dict(torch.load("models/pretrained/all_{}.pth".format(pretrain)))
#we have to see arity to know what is happening inside
net = Network(network, "mnist_net", batching=True)
# we use Adam as the optimizer and learning rate is 10 ** -3
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

#defining model for deepproblog
model = Model("models/addition.pl", [net])
if method == "exact":
    model.set_engine(ExactEngine(model), cache=True) #we use this for now
elif method == "geometric_mean":
    model.set_engine(
        ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False)
    )

#実際の画像データを紐付け、学習、評価をする一連
model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

#making data loader for sufficient data supply
#false means there is no shuffle
loader = DataLoader(train_set, 2, False)

#実際の学習 1= epochs profile= ? log_iter = 100回のイテレーション（バッチ処理）ごとに進捗を報告するという意味
train = train_model(model, loader, 1, log_iter=100, profile=0)
#saving the model
model.save_state("snapshot/" + name + ".pth")
#showing the result to the logfile
train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.comment(
    "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
)
train.logger.write_to_file("log/" + name)


# まずadd_tensor_sourceで、モデルが画像データの場所を把握できる住所録を準備します。
# 次にDataLoaderで、学習データをモデルに供給するためのベルトコンベアを用意します。
# 最後にtrain_modelという工場長が、住所録とベルトコンベアを使って、実際にモデルの学習（生産）を開始します。