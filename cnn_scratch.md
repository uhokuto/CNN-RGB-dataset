# CNNのスクラッチ学習
メインプログラム
ローカル版　[cnn_augmentation_main.py](./CNN_pytorch/cnn_augmentation_main.py)
colab版　[cnn_augmentation_main.ipynb](./CNN_pytorch/cnn_augmentation_main.ipynb)

[2. pytorch版CNN vgg16 pretrained](cnn_vgg16_pretrained.md)との共通点・違いは以下の通り。なお、pretrainedに比べてはるかにレイヤがシンプルなので、ローカルPC(cpu)で十分速い。
なお、colabでGPUを使う場合、DataLoaderからのデータをGPU空間に投げ込む必要があるので、x = x.to(device)  labels = labels.to(device)のようにコーディングしている

1. pretrained modelをロードする代わりに、[model.py](./CNN_pytorch/model.py)のclass MyNet(torch.nn.Module):を使う。model.pyにはモデルのレイヤ構成やノードのデザインをスクラッチで書いている

2. RGB画像を読み込んでテンソルに変換するモジュールは、以下の通りでcnn_vgg16_pretrainedと共通（テスト確認済）
メイン：[get_image_tensor.py](./CNN_pytorch_fineTuning/get_image_tensor.py)　（colab版は、このプログラムを一体化した）

サブプログラム：[augmentation_colab.py](./CNN_pytorch_fineTuning/augmentation_colab.py) [config.py](./CNN_pytorch_fineTuning/config.py) [np.py](./CNN_pytorch_fineTuning/np.py)


3. colab版の実行
上記メインプログラムをアップロードするとともに、[model.py](./CNN_pytorch/model.py)[augmentation_colab.py](./CNN_pytorch_fineTuning/augmentation_colab.py) [config.py](./CNN_pytorch_fineTuning/config.py) [np.py](./CNN_pytorch_fineTuning/np.py)の3つをサイドウィンドウにアップロードする
