# RGB画像からのtorch tensorの作成
フォルダーに保存したRGB画像を自動的に読み込んで、イメージデータの読み込み(Image.open()) → augmentation → resize → tensor型への変換→ DataSet型(Data loader)への変換をすべて連続的に処理する。

メインプログラム：[get_image_tensor.py](./CNN_pytorch_fineTuning/get_image_tensor.py)
サブプログラム：[augmentation.py](./CNN_pytorch_fineTuning/augmentation.py) [config.py](./CNN_pytorch_fineTuning/config.py) [np.py](./CNN_pytorch_fineTuning/np.py)

なお、colaboratoryで実行する場合、メインプログラムは、CNNメインプログラムの.ipynbと一体化している（google driveをmountするように修正している）。また、augmentation.pyは[augmentation_colab.py](./CNN_pytorch_fineTuning/augmentation_colab.py)を使うこと。ローカル環境ではopenCVがcuda対応しているので、cuda上でaugmentationするようにコーディングしているが、colabではcvが
cuda対応していないので、該当部分をcpu対応に修正している。

## メインプログラム get_image_tensor.py
pytorchのtorch.utils.data.Datasetを継承、オーバーライドして、torchvision.datasets.ImageFolderと同じような
処理を実行し、それにcv2ベースのaugmentation(affine etc.)処理を追加したもの。具体的には以下の処理を行う

- 画像は教師ラベル別フォルダ（ラベルは整数）に入っているとする（ここは修正可能）
- 教師ラベル別フォルダの親フォルダをパスに指定すると、サブフォルダ名を教師ラベル、サブフォルダ中の画像をテンソル
　とするデータセットを作成する。

1. Datasetクラス
class image_feature(Dataset):torch.utils.data.Datasetを継承したクラス。
Datasetを継承したクラスでは　以下の__init__  __getitem__ いずれかにデータセット生成処理を記述すると、コンストラクタを生成すると同時にデータセットが生成される（改めてメソッドを呼び出す必要がない）。（インスタンスを生成しなくても、いきなり全データセットがtraindataに入ってくるのが特徴。）以下torch.utils.data.Datasetの骨格。


以下、dataset, dataloader, compose, transformなど主なクラスの継承、オーバーライド方法が書かれている
https://hacks.deeplearning.jp/pytorch%E3%81%AEdataloader/
https://pystyle.info/pytorch-how-to-create-custom-dataset-class/

```python
class image_feature(Dataset):

    def __init__(self, img_dir,cuda_enabled,augment):ここに処理を書くことでクラス呼び出し時にデータセットが生成される
　　　　　　　　　　　　　　　　　　　　　　（__getitem__もクラス呼び出し時に実行されるので、そちらに処理を書いてもいい）

    
    def __getitem__(self, index):  生成したデータセット集合から特定のデータをindexで取り出すための内部メソッド。通常のインデクシングだと img[index] , label[index] というように個別に指定する必要があるが、__getitem__ （ここではimgとlabelのペア）に対して、まとめてインデクシングできる。つまり、traindata[index] とすると該当のイメージテンソルとクラスラベルがいっぺんに取り出される。
        return img, label
        
    def __len__(self):
        return len(self.images) # 単にデータセットの数を返すもの
```


2. コンストラクタ
augment = 1 1ならaugmentあり、0ならなし


3. メインプログラムコーディングの詳細
if __main___ 記述を書くと単体で動かせる。特に　__getitem__　からの戻り値のデータを確認すること。

```python
def __init__(self, img_dir,cuda_enabled,augment):
     画像ファイルのパス一覧を取得する。この処理は以下を参考にした。サブフォルダにクラスラベルをつけると、片っ端から画像とクラスラベルを作成する。なお、クラスラベル名は以下の*でパス名から文字列処理で取得している
　　　https://ohke.hateblo.jp/entry/2019/12/28/230000
        img_dir = Path(img_dir)
        dir_list = img_dir.glob('*/*')        
        self.img_paths = [ p for p in dir_list if p.suffix in image_feature.IMG_EXTENSIONS]
        
        path = self.img_paths
        self.labels =[]
        self.images =[]
        self.transform=Compose([transforms.Resize((224,224), interpolation=Image.BICUBIC), transforms.ToTensor(), \
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        def build_dataset(label, img):
            img=Image.fromarray(img) # cv2ベースのaugmentationはnumpy配列型に対して処理する。これに対しいて、torch transform(augmentationに相当）は、PIL Imageデータ型に対して処理するので、変換している。
            img = self.transform(img)# このメソッド（上記）でテンソル型に変換。最後に必ずtorch.tensor型に変換が必要
            self.labels.append(label)
            self.images.append(img)
            
            
        
        for i in range(len(path)):
            image = Image.open(path[i])        
            img = np.asarray(image)       
            label = int(str(path[i]).split("\\")[-2])*
            build_dataset(label,img)
            if augment ==0:
                continue
            aug = augmentation.aug(img,cuda_enabled) #これ以降がcv2ベースのaugmentation. torchベースのaugmentationは、transforms なのでこれに書き換えられるかもしれない。
            参考サイトは　https://pystyle.info/pytorch-list-of-transforms/

            img_h_flip = aug.horizontal_flip()
            build_dataset(label,img_h_flip)
            for j in range(1):
                shear_range =0.2
                img_affine = aug.affine(shear_range)
                build_dataset(label,img_affine)
                shift_ratio = 0.3
                img_shift = aug.horizontal_shift(shift_ratio)
                build_dataset(label,img_shift)
                crop_rate=0.7
                img_crops = aug.random_crop(crop_rate)
                build_dataset(label,img_crops)
            
        self.labels=torch.tensor(self.labels, dtype=torch.long)  クラスラベルデータは、torch.longでテンソルにしないと、nn.の勾配計算でエラーになる！

## augmentation　　　　
1）アフィン変換　引数shear_rangeには画像をゆがめる割合を入力する
　　cv2のアフィン変換の記述およびアフィン変換の原理は以下が詳しい（このコーディングの元ネタ）
　　https://qiita.com/koshian2/items/c133e2e10c261b8646bf
　　アフィン変換すると、画像上に黒い余白ができるのでcropしていることに注意
　　
def affine(self,shear_range):
    
        self.w =self.img.shape[1]
        self.h=self.img.shape[0]
        self.shear_range=shear_range
        
        
        h = self.img.shape[0] // 2　画像センターの座標を算出
        w = self.img.shape[1] // 2
        randoms = np.random.uniform(1.0-shear_range, 1.0+shear_range, (3,2)).astype(np.float32) #32ビット変数にする
        coefs = np.array([[-1,-1],[1,-1],[1,1]], np.float32)　
        centers = np.array([[h,w]], np.float32)
        origin = centers + centers * coefs　画像上にセンターを中心とした相対座標[-1,-1],[1,-1],[1,1]の3角形を作る
        dest = centers + centers * coefs * randoms　上記3角形をshear_rangeからの乱数randomsに従ってゆがめる（アフィン変換）
        affine_matrix = cv2.getAffineTransform(origin, dest)　アフィン変換行列を生成（アフィン変換は、変換行列と変換対象の画像の積で計算するが、どのように変換したいのかを3角形origin -> destの変換でまず表しておいて、それを実現するためのアフィン変換行列を逆算する便利なメソッド）
        h_margin=int(h*shear_range)
        w_margin=int(w*shear_range)
        image_affine= cv2.warpAffine(self.img, affine_matrix, (self.img.shape[1],self.img.shape[0]))　画像を入力してアフィン変換
        img_crop = image_affine[h_margin:-h_margin,w_margin:-w_margin,:]　アフィン変換でできあがる黒い余白は　最大h*shear_range　w*shear_rangeなのでcropして余白を取る
        
        return img_crop

2) 水平シフト　アフィン変換を用いて水平移動する
    def horizontal_shift(self,shear_range):
    
3) 左右逆転
　　参考　https://www.kumilog.net/entry/numpy-data-augmentation
        
    def horizontal_flip(self,):
        image = self.img[:, ::-1, :]
        return image
    
4）上下逆転
    def vertical_flip(self,):
        image = image[::-1, :, :]
        return image

5）crop（切り抜き）
    def random_crop(self, crop_rate):

6）リサイズ　これはCNN入力データセット作成時にどちらにしろ統一したリサイズ
　　　になるので意味なし
    
    def random_resize(self, scale_ratio):