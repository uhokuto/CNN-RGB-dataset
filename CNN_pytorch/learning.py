
import os
import argparse
import glob
import cv2
import torch
import torchvision
import torch.nn.functional as f
from torchvision.transforms import functional as TF
from model import MyNet # 
import matplotlib.pyplot as plt
import torch.nn as nn

def cnn_train(net, device, loaders, data_size, optimizer, e, history):
    """CNNによる学習を実行する。
    net.parameters()に各conv, fcのウェイト・バイアスが格納される。
    """

    loss = None         # 損失関数の結果
    loss_sum = 0        # 損失関数の値（エポック合計）
    train_correct_counter = 0   # 正解数カウント

    # 学習開始（再開）
    net.train(True) # 引数は省略可能
    criterion = nn.CrossEntropyLoss()
    for i, (data, labels) in enumerate(loaders):
        # GPUあるいはCPU用に再構成
        data = data.to(device)      # バッチサイズの画像データtensor
        labels = labels.to(device)  # バッチサイズの正解ラベルtensor

        # 学習
        optimizer.zero_grad()   # 前回までの誤差逆伝播の勾配をリセット
        output = net(data)      # 推論を実施（順伝播による出力）

        loss = criterion(output, labels)
        #loss = f.nll_loss(output, labels)   # 交差エントロピーによる損失計算（バッチ平均値）
        loss_sum += loss.item() * data.size()[0] # バッチ合計値に直して加算

        loss.backward()         # 誤差逆伝播
        optimizer.step()        # パラメータ更新

        train_pred = output.argmax(dim=1, keepdim=True) # 0 ~ 9のインデックス番号がそのまま推論結果
        train_correct_counter += train_pred.eq(labels.view_as(train_pred)).sum().item() # 推論と答えを比較し、正解数を加算

        # 進捗を出力（8バッチ分ごと）
        if i % 8 == 0:
            print('Training log: epoch_{} ({} / {}). Loss: {}'.format(e+1, (i+1)*loaders.batch_size, data_size, loss.item()))

    # エポック全体の平均の損失関数、正解率を格納
    ave_loss = loss_sum / data_size
    ave_accuracy = train_correct_counter / data_size
    history['train_loss'].append(ave_loss)
    history['train_acc'].append(ave_accuracy)
    print(f"Train Loss: {ave_loss} , Accuracy: {ave_accuracy}")

    return
    
def cnn_test(net, device, loaders, data_size, e, epoch, history):
    """
    学習したパラメータでテストを実施する。
    """
    # 学習のストップ
    net.eval() # または　net.train(False)でもいい
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0                # 損失関数の値（数値のみ）
    test_correct_counter = 0    # 正解数カウント
    data_num = 0                # 最終エポックでの出力画像用ナンバー

    with torch.no_grad():
        for data, labels in loaders:
            # GPUあるいはCPU用に再構成
            data = data.to(device)      # バッチサイズの画像データtensor
            labels = labels.to(device)  # バッチサイズの正解ラベルtensor

            output = net(data)  # 推論を実施（順伝播による出力）
            loss_sum = loss = criterion(output, labels)
            #loss_sum += f.nll_loss(output, labels, reduction='sum').item() # 損失計算　バッチ内の合計値を加算

            test_pred = output.argmax(dim=1, keepdim=True) # 0 ~ 9のインデックス番号がそのまま推論結果
            test_correct_counter += test_pred.eq(labels.view_as(test_pred)).sum().item() # 推論と答えを比較し、正解数を加算

            # 最終エポックのみNG画像を出力
            if e == epoch - 1:
                last_epoch_NG_output(data, test_pred, labels, data_num)
                data_num += loaders.batch_size
    
    # テスト全体の平均の損失関数、正解率を格納
    ave_loss = loss_sum / data_size
    ave_accuracy = test_correct_counter / data_size
    history['test_loss'].append(ave_loss)
    history['test_acc'].append(ave_accuracy)
    print(f'Test Loss: {ave_loss} , Accuracy: {ave_accuracy}\n')

    return

def last_epoch_NG_output(data, test_pred, target, counter):
    """
    不正解した画像を出力する。
    ファイル名：「データ番号-pre-推論結果-ans-正解ラベル.jpg」
    """
    # フォルダがなければ作る
    dir_path = "./NG_photo_CNN"
    os.makedirs(dir_path, exist_ok=True)

    for i, img in enumerate(data):
        pred_num = test_pred[i].item()  # 推論結果
        ans = target[i].item()          # 正解ラベル

        # 推論結果と正解ラベルを比較して不正解なら画像保存
        if pred_num != ans:
            # ファイル名設定
            data_num = str(counter+i).zfill(5)
            img_name = f"{data_num}-pre-{pred_num}-ans-{ans}.jpg"
            fname = os.path.join(dir_path, img_name)
            
            # 画像保存
            torchvision.utils.save_image(img, fname)

    return

def output_graph(epoch, history_train, history_test):
    os.makedirs("./CNNLearningResult", exist_ok=True)

    # 各エポックの損失関数グラフ
    plt.figure()
    plt.plot(range(1, epoch+1), history_train['train_loss'], label='train_loss', marker='.')
    plt.plot(range(1, epoch+1), history_test['test_loss'], label='test_loss', marker='.')
    plt.xlabel('epoch')
    plt.legend() # 凡例
    plt.savefig('./CNNLearningResult/loss_cnn.png')

    # 各エポックの正解率グラフ
    plt.figure()
    plt.plot(range(1, epoch+1), history_train['train_acc'], label='train_acc', marker='.')
    plt.plot(range(1, epoch+1), history_test['test_acc'], label='test_acc', marker='.')
    plt.xlabel('epoch')
    plt.legend() # 凡例
    plt.show()
    plt.savefig('./CNNLearningResult/acc_cnn.png')

    return