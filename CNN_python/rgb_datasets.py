import shutil
import glob
import numpy as np
from skimage import io, color,transform
from PIL import Image

class rgb:
     # -- 各層の初期化 --

    def __init__(self,label_list,dataDir_list,img_h,img_w):
    
        labels=[]
        img_data=[]
        img_list=[]
        for label,dataDir in zip(label_list,dataDir_list):
        #ワイルドカードでdataディレクトリ内の全ファイル名を取得してリスト化
            files = glob.glob(dataDir + "*")
            for file in files:  
                labels.append(label)
                #img = io.imread(file)
                img = Image.open(file)
                #resizeの引数はwidth, height イメージ画像のデータもこの並びになっている
                img_resize = img.resize((img_w,img_h))
                img_data.append(img_resize)
                #np.arrayに変換すると、height,width,ch になる
                array_img=np.asarray(img_resize)
                trans_img=array_img.transpose(2,0,1)
                img_one_dim = np.ravel(trans_img)
                img_list.append(img_one_dim)
                
        self.correct =np.array(labels)
        self.input_data = np.asfarray(img_list)
        self.images = img_data
        self.n_data = len(self.correct)
    
    def normalize(self,no_of_class):
        # -- 入力データの標準化 --
        
        ave_input = np.mean(self.input_data,axis=1)
        std_input = np.std(self.input_data,axis=1)
        self.input_data = (self.input_data - ave_input.reshape(self.n_data,-1)) / std_input.reshape(self.n_data,-1)
        
        # -- 正解をone-hot表現に --
        self.correct_data = np.zeros((self.n_data,no_of_class))
        for i in range(self.n_data):
    
            self.correct_data[i, self.correct[i]] = 1.0
            
            
        return self.correct_data,self.input_data
        

    
    def create_train_test(self,split_rate):
        r = split_rate*10
    
        

        # -- 訓練データとテストデータ --
        #シャッフルしていないが問題なし　メインプログラムでバッチを作る際にランダムサンプリングしているので
        index = np.arange(self.n_data)
        index_train = index[index%r != 0]
        index_test = index[index%r == 0]

        self.input_train = self.input_data[index_train, :]  # 訓練 入力
        self.correct_train = self.correct_data[index_train, :]  # 訓練 正解
        self.input_test = self.input_data[index_test, :]  # テスト 入力
        self.correct_test = self.correct_data[index_test, :]  # テスト 正解
        
        return self.input_train,self.correct_train,self.input_test,self.correct_test




