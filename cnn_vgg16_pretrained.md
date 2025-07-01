# CNN vgg16 pretrained モデル
ローカル版[cnn_augmentation_vgg16main.py](./CNN_pytorch_fineTuning/cnn_augmentation_vgg16main.py)
colab版[cnn_augmentation_vgg16main.ipynb](./CNN_pytorch_fineTuning/cnn_augmentation_vgg16main.ipynb)
学習済CNN(vgg16)をロードして、更新が必要なパラメータだけを取り出し、optimizerに渡すプログラム。また、出力レイヤの構成のみ変更する。ファインチューニングの方法も以下に記載した。これを応用すると、必要なレイヤのパラメータだけを更新したり、畳み込み層を含めたあらゆるレイヤを任意に上書きして構成変更することができる。

元ネタは
https://venoda.hatenablog.com/entry/2020/10/14/071440#33-Dataset%E3%81%AE%E4%BD%9C%E6%88%90

### RGBデータのテンソルへの変換
上記のメインプログラムは、RGB画像データを読み込んでaugmentation, Dataloaderに変換するためのクラス[get_image_tensor.py](./CNN_pytorch_fineTuning/get_image_tensor.py)を呼び出す。このプログラムの詳細は、[3. RGB画像からのpytorch tensor作成](./create_RGBtensor.md)を参照。colab版では、get_image_tensor.pyは、上記のipynbに一体化している。ただし、get_image_tensor.pyは、サブプログラム[augmentation_colab.py](./CNN_pytorch_fineTuning/augmentation_colab.py) [config.py](./CNN_pytorch_fineTuning/config.py) [np.py](./CNN_pytorch_fineTuning/np.py)　を呼び出すため、colaboratoryのサイドウィンドウにこの3つのプログラムをアップロードすること。colaboratoryから外部モジュールを実行する方法は以下を参照。
https://qiita.com/76r6qo698/items/6279035a16d1548c8ff8





```python
from torchvision import models

use_pretrained = True

# 学習済cnn(vgg16)をロード
model = models.vgg16(pretrained=use_pretrained)

vggのレイヤ構成を表示
print(model)

以下の構文で上記のレイヤを指定して取り出すことができる classifierは全結合層で、[6]は出力レイヤを意味する
print('変更前 : ', model.classifier[6])
レイヤの上書き
model.classifier[6] = nn.Linear(in_features=4096, out_features=2)
print('変更後 : ', model.classifier[6])

#####
list(model.classifier[6].named_parameters())
for name, param in model.named_parameters():
    print('name : ', name)
params_to_update = []

# 学習させるパラメータ名
update_param_names = ['classifier.6.weight', 'classifier.6.bias']

# モデルのパラメータを逐次読み出す。
for name, param in model.named_parameters():
    
　勾配降下が必要なパラメータ以外は凍結
    if name in update_param_names:
        param.requires_grad = True
        params_to_update.append(param)
        print('name : ', name)
    else:
        param.requires_grad = False

# params_to_updateの中身を確認
print('--------------------')
print(params_to_update)
#####


catdog_dir = 'C:/Users/uhoku/Dropbox/Python/CNN/catdog/img/'

シフトや、回転などfor文で繰り返す回数を指定。この数分、画像データを水増しできる
no_of_augment =2
dataset = image_feature(catdog_dir,device,no_of_augment,img_w,img_h)

画像データ・正解ラベルのペアをデータにセットする
TensorDatasetはtensor をデータセット型に変換するメソッド。上記のimage_featureでは、これに代えてclass image_feature(Dataset):でカスタマイズしたdataset作成メソッドを使う。このようにカスタマイズしたデータセットを使いたい場合は、TensorDatasetのかわりにDatasetメソッドを継承して使う
#dataset = torch.utils.data.TensorDataset(data_tensor, label_tensor)

train_data_size = int(len(dataset) * 0.8)
test_data_size = len(dataset)-train_data_size
train_data, val_data = torch.utils.data.random_split(dataset, [train_data_size, test_data_size])
train_length=len(train_data)
val_length = len(val_data)
    
セットしたデータをバッチサイズごとの配列に入れる。
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=10, shuffle=True)



最適化の引数に、更新対象のパラメータのリスト型を指定する
なお、例えば nn モデルパラメータをスクラッチで学習する場合は、この引数をmodel.parameters()とする。ので、vgg16でファインチューニングする場合は、同様に、上記のレイヤの上書きまでコーディングして#####    ######は除外して、以下の引数に
model.named_parameters():とすればよいはず。

ファインチューニング
#optimizer=torch.optim.Adagrad(model.parameters(), lr=0.0001, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
#
転移学習
optimizer=torch.optim.Adagrad(params_to_update, lr=0.0001, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)


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
  

train_loss_tensor = torch.stack(history["train_loss"])
train_loss_np = train_loss_tensor.detach().numpy().copy()
val_loss_tensor = torch.stack(history["val_loss"])
val_loss_np = val_loss_tensor.detach().numpy().copy()
plt.plot(train_loss_np[5:])
plt.plot(val_loss_np[5:])
plt.show()


print(f"Accuracy: {100*correct/len(val_data)}% ({correct}/{len(val_data)})")

```