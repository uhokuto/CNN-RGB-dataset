import os
import glob
import cv2
import torch
import torchvision
import torch.nn.functional as f
from torchvision.transforms import functional as TF






def img2tensor(dir_path,label,img_w,img_h):
    """画像データを読み込み、データセットを作成する。
    画像のファイル名は、末尾を「...ans〇.jpg」（〇に正解ラベルの数字を入れる）として用意すること。 

    Returns:
        DataLoader: 画像データ配列と正解ラベル配列がバッチごとにペアになったイテレータ
        data_size: データ数
    """

    # 画像ファイル名を全て取得
    img_paths = os.path.join(dir_path, "*.jpg")
    img_path_list = glob.glob(img_paths)

    # 画像データ・正解ラベル格納用配列
    data = []
    labels = []

    # 各画像データ・正解ラベルを格納する
    for img_path in img_path_list:
        # 画像読み込み・(3, height, width)に転置・正規化
        img=cv2.imread(img_path)
        img = cv2.resize(img, dsize=(img_w,img_h))
        img = TF.to_tensor(img)

        # 画像をdataにセット
        data.append(img.detach().numpy()) # 配列にappendするため、一度ndarray型へ

        # 正解ラベルをlabelsにセット
        
        labels.append(label)

    # PyTorchで扱うため、tensor型にする
    data = torch.tensor(data)
    labels = torch.tensor(labels)
    
    return data,labels
    
    
    