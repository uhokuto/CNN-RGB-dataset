import os
import glob
import cv2
import torch
import torchvision
import torch.nn.functional as f
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
from create_dataset import img2tensor
#from learning import cnn_train, cnn_test,output_graph
from model import MyNet # このあと自分で定義するmodel.pyからのネットワーククラス
import torch.nn as nn

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
model : torch.nn.Module = MyNet(img_w,img_h)
model = model.to(device) # GPUあるいはCPUに合わせて再構成

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
train_length=len(train_data)
val_length = len(val_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=10, shuffle=True)


optimizer=torch.optim.Adagrad(model.parameters(), lr=0.0001, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,500], gamma=0.5)
criterion = nn.CrossEntropyLoss()
history = {"train_loss": [], "val_loss": [], "labels":[],"train_loss_byBatch":[]}
num_epochs=30
Y_train = []
pred_train =[]
min_loss=10000000000
predicts=[]
trues = []
n_epoch=0
for epoch in range(num_epochs):
  model.train()# nn.moduleを継承したクラスからオブジェクトを生成した場合、訓練と予測はコードをそれぞれ記述
             # する必要はなく　train(), eval()　で自動的にに切り替わる
  
  train_loss_acc=0
  for i, (x, labels) in enumerate(train_loader):
    
    #x=torch.t(x)
    output = model(x)
    # 分類問題の場合output は、行方向がバッチサイズ、列方向がクラス数であるような2次元配列
    # これに対して、labelsは、バッチサイズ分の1次元配列。分類問題でのニューラルネットワークの
    # 教師ラベルレイヤは、one hot なので、本来はoutputと同じクラス数分の次元数をもつはずだか
    # これは、nn.CrossEntropyLoss()のなかで自動的にデータ形式の変換をやっている模様
    # また、criterion での損失計算結果 loss は、バッチサイズ分足し合わせたスカラーになる。
    loss = criterion(output, labels)   
    history["train_loss_byBatch"].append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss_acc+=loss 
    if num_epochs-1 == epoch:
      Y_train+=labels.view(1,-1).to('cpu').detach().numpy().copy().tolist()
      pred_train+=output.view(1,-1).to('cpu').detach().numpy().copy().tolist()

  train_loss=train_loss_acc/train_length

  #if 損失が閾値以上ならepochをリセット
  print(f'Epoch: {epoch+1}, loss: {train_loss: 0.4f}')
  history["train_loss"].append(train_loss)

  if train_loss<min_loss:
        min_loss=train_loss
        torch.save(model, 'best_model.pth')

  
  scheduler.step()

  correct = 0
  model.eval()

  val_loss_acc=0  
  with torch.no_grad():
    for i, (x, labels) in enumerate(val_loader):
      
      output = model(x)
      loss = criterion(output,labels)
      val_loss_acc+=loss

      if num_epochs-1 == epoch:    
        pred = output.argmax(1)
        predicts+=pred.detach().numpy().copy().tolist()
        trues+=labels.detach().numpy().copy().tolist()
        # 正解数をカウント　
        correct += pred.eq(labels.view_as(pred)).sum().item()    
  val_loss = val_loss_acc/val_length
  print(f'Epoch: {epoch+1}, val_loss: {val_loss : 0.4f}')
  history["val_loss"].append(val_loss)
  '''
  if n_epoch >10:
      break
  else:
    if val_loss < train_loss :
      n_epoch +=1
    else:
      n_epoch =0
  '''

train_loss_tensor = torch.stack(history["train_loss"])
train_loss_np = train_loss_tensor.detach().numpy().copy()
val_loss_tensor = torch.stack(history["val_loss"])
val_loss_np = val_loss_tensor.detach().numpy().copy()
plt.plot(train_loss_np[5:])
plt.plot(val_loss_np[5:])
plt.show()


'''
train_loss_batch = torch.stack(history["train_loss_byBatch"])
train_loss_batchnp = train_loss_batch.detach().numpy().copy()

plt.plot(train_loss_batchnp)
plt.show()
'''



print(f"Accuracy: {100*correct/len(val_data)}% ({correct}/{len(val_data)})")


