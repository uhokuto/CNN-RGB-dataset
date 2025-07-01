import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision
from PIL import Image
 
# 学習済みモデルの取得
model = models.vgg16(pretrained=True)
#for param in model.parameters():
#    param.requires_grad = False

 
# 全結合層の変更(最終層の出力を2にする)
model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 2),
)


# GPUを使う
device = torch.device('cuda')
model = model.to(device)
 
# バッチサイズ
batchsize = 8
 
# 学習データ
traindata = torchvision.datasets.ImageFolder(root='C:/Users/uhoku/Dropbox/Python/bag_of_visual_words/catdog/img/',transform=transforms.Compose([transforms.Resize((224,224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]))
trainloader = torch.utils.data.DataLoader(dataset=traindata, batch_size=batchsize, shuffle=True)
 
# テストデータ
testdata = torchvision.datasets.ImageFolder(root='C:/Users/uhoku/Dropbox/Python/bag_of_visual_words/catdog/test_img/',transform=transforms.Compose([transforms.Resize((224,224), interpolation=Image.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]))
testloader = torch.utils.data.DataLoader(dataset=testdata, batch_size=batchsize, shuffle=True)
 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
 
# 50エポック学習
for epoch in range(2):
    running_loss = 0.0
    correct_num = 0
    total_num = 0
    for i, (data, target) in enumerate(trainloader):
        print(data, target)
        print(type(target[0]))
        inputs, labels = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        predicted = torch.max(outputs.data, 1)[1]
        correct_num_temp = (predicted==labels).sum()
        correct_num += correct_num_temp.item()
        total_num += batchsize
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
 
    # 経過の出力
    print('epoch:%d loss: %.3f acc: %.3f' %
          (epoch + 1, running_loss / 100, correct_num*100/total_num))
 
# モデルの保存
torch.save(model.state_dict(), 'sample.pt')
 
# テストデータ精度出力
model.eval()
correct_num = 0
total_num = 0
for i, (data, target) in enumerate(testloader):
    inputs, labels = data.to(device), target.to(device)
    outputs = model(inputs)
    predicted = torch.max(outputs.data, 1)[1]
    correct_num_temp = (predicted == labels).sum()
    correct_num += correct_num_temp.item()
    total_num += batchsize
 
print('test acc: %.3f' % (correct_num * 100 / total_num))