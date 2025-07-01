import shutil
import glob
import numpy as np
from skimage import io, color,transform
from PIL import Image
import augmentation
import cv2

class rgb:
	 # -- 各層の初期化 --

	def __init__(self,label_list,dataDir_list,img_h,img_w):
	
		labels=[]
		img_data=[]
		img_list=[]
		
		def build_dataset (label,img):
		
			labels.append(label)
			img_resize = cv2.resize(img,(img_w,img_h))
			img_data.append(img_resize)
			array_img=np.asarray(img_resize)
			trans_img=array_img.transpose(2,0,1)
			img_one_dim = np.ravel(trans_img)
			img_list.append(img_one_dim)
				
		
		for label,dataDir in zip(label_list,dataDir_list):
		#ワイルドカードでdataディレクトリ内の全ファイル名を取得してリスト化
			files = glob.glob(dataDir + "*")
			for file in files:	
				
				#img = io.imread(file)
				#img = Image.open(file)
				img = cv2.imread(file)[:,:,::-1]
				build_dataset(label,img)				
				
				aug = augmentation.aug(img)
				img_h_flip = aug.horizontal_flip()
				build_dataset(label,img_h_flip)
				
				gen_no=2
				for i in range(gen_no):
					shear_range =0.2
					img_affine = aug.affine(shear_range)
					build_dataset(label,img_affine)
					
					shift_ratio = 0.3
					img_shift = aug.horizontal_shift(shift_ratio)
					build_dataset(label,img_shift)
				
					crop_rate=0.7
					img_crops = aug.random_crop(crop_rate)
					build_dataset(label,img_crops)
				
				
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




