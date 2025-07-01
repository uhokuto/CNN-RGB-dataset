import os
import glob
import cv2
import torch
import torchvision
import torch.nn.functional as f
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
from create_dataset import img2tensor
from learning import cnn_train, cnn_test,output_graph
from model import MyNet # このあと自分で定義するmodel.pyからのネットワーククラス


# https://qiita.com/harutine/items/972cc5ff7868d6dec27b#3-pytorch%E3%81%A7%E3%81%AEcnn%E5%AE%9F%E8%A3%85

# 計算環境が、CUDA(GPU)か、CPUか
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device: ' + device)

# 学習・テスト結果の保存用辞書
history_train = {
    'train_loss': [],   # 損失関数の値
    'train_acc': [],    # 正解率
}

history_test = {
    'test_loss': [],    # 損失関数の値
    'test_acc': [],     # 正解率
}
img_w=128
img_h=128
# ネットワークを構築（ : torch.nn.Module は型アノテーション）
# 変数netに構築するMyNet()は「ネットワークの実装」で定義します
net : torch.nn.Module = MyNet(img_w,img_h)
net = net.to(device) # GPUあるいはCPUに合わせて再構成

# データローダー・データ数を取得
#（load_dataは「学習データ・テストデータのロードの実装（データローダー）」の章で定義します）
#train_dir = args.trainDir # 学習用画像があるディレクトリパス
#test_dir = args.testDir   # テスト用画像があるディレクトリパス

cat_dir = 'C:/Users/uhoku/Dropbox/Python/CNN/catdog/img/0/'
dog_dir = 'C:/Users/uhoku/Dropbox/Python/CNN/catdog/img/1/'
img_w=128
img_h=128
data_cat, label_cat = img2tensor(cat_dir,0,img_w,img_h)
data_dog, label_dog = img2tensor(dog_dir,1,img_w,img_h)
data_tensor=torch.cat((data_cat,data_dog),0)
label_tensor=torch.cat((label_cat,label_dog),0)

data_cat.size()

# 画像データ・正解ラベルのペアをデータにセットする
dataset = torch.utils.data.TensorDataset(data_tensor, label_tensor)

train_data_size = int(len(dataset) * 0.8)
test_data_size = len(dataset)-train_data_size
train_data, val_data = torch.utils.data.random_split(dataset, [train_data_size, test_data_size])
  
    
# セットしたデータをバッチサイズごとの配列に入れる。
train_loaders = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
test_loaders = torch.utils.data.DataLoader(val_data, batch_size=10, shuffle=True)

#train_loaders, train_data_size = load_data(args.trainDir)
#test_loaders, test_data_size = load_data(args.testDir)

# オプティマイザを設定
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)

epoch = 15

# 学習・テストを実行
for e in range(epoch):
    # 以下2つの関数は「学習の実装」「テストの実装」の章で定義します
    cnn_train(net, device, train_loaders, train_data_size, optimizer, e, history_train)
    cnn_test(net, device, test_loaders, test_data_size, e, epoch, history_test)

# 学習済みパラメータを保存
torch.save(net.state_dict(), 'params_cnn.pth')

# 結果を出力（「結果出力の実装」の章で定義します）
output_graph(epoch, history_train, history_test)

