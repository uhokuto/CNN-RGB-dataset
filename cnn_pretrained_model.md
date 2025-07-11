# CNNの代表的な特徴抽出バックボーン(pretrained model)
バックボーンとは、CNNでの特徴量抽出のためのレイヤを意味する。従って、後続に接続するレイヤが１つの全結合NNならば、クラス分類。並行して複数の畳み込みが接続される場合は、物体検出となる。

CNNの代表的なバックボーンには，たとえば以下のようなものがある。実装記事があるものはリンクを貼ってある：
[6大バックボーン](https://ai-kenkyujo.com/artificial-intelligence/ai-architecture-02/
)

[AlexNet](https://www.koi.mashykom.com/pytorch.html)
VGGNet
GoogleNet
[ResNet](https://www.nogawanogawa.com/entry/resnet_pytorch)
DenseNet
SENet(Squeeze-and-Exitation Networks)
[PSPNet](https://venoda.hatenablog.com/entry/2024/05/16/194504)

ResNetは非常に深いバックボーンで、学習が困難だがこれを解決するための残差接続という方法を用いている。詳しくは
[1](https://medium.com/@siddheshb008/resnet-architecture-explained-47309ea9283d)
[2](https://aismiley.co.jp/ai_news/title-resnet-cnn-microsoft-research/)
[3](https://cvml-expertguide.net/2020/04/13/cnn/#:~:text=%E5%88%9D%E6%9C%9F%E3%81%AB%E3%82%88%E3%81%8F%E7%94%A8%E3%81%84%E3%82%89%E3%82%8C,%E3%82%88%E3%81%86%E3%81%AB%E3%81%AA%E3%81%A3%E3%81%A6%E3%81%84%E3%82%8B%EF%BC%8E)

こちらは各バックボーンの進化の歴史と詳細をまとめたサーベイ
https://qiita.com/yu4u/items/7e93c454c9410c4b5427

その他参考記事　ImageNet（ILSVRC2012）データセット
https://starpentagon.net/analytics/imagenet_ilsvrc2012_dataset/

