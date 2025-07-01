
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
import random
import local_features


class bovw:
	 # -- 各層の初期化 --

	def __init__(self,label_list,dataDir_list,img_w,img_h,split_rate,codebook_size):

		self.codebook_size=codebook_size
		self.correct, self.data_descriptors,self.images = local_features.img2descriptors(label_list,dataDir_list,img_w,img_h)
		
		self.n_data = len(self.correct)
	
		r = split_rate*10
		index=np.arange(self.n_data)		
		random.shuffle(index)
		self.index_train = index[index%r != 0]
		self.index_test = index[index%r == 0]

		train_descriptors = np.vstack(self.data_descriptors[self.index_train])
		#kmeansでHog特徴量ベクトルcodebook_size数のクラスにクラスタリング
		self.kmeans = MiniBatchKMeans(n_clusters=self.codebook_size, batch_size=100, random_state=0)
		self.kmeans.fit(train_descriptors.astype(float))
		
		

	def create_train_test(self):
	
			
		self.correct_data = np.zeros((self.n_data,2))
		for i in range(self.n_data):
    
			self.correct_data[i, self.correct[i]] = 1.0
		
		def get_feature(descriptors):
	
			#featureは、ヒストグラムの横軸。コードブックの種類数分がヒストグラムのbin数になるのでまずは、全てのヒストグラムを０に初期化
			feature = np.zeros(self.codebook_size)	
			for index in self.kmeans.predict(descriptors.astype(float)):
				#コードブック番号に該当するbinの頻度を１つ足す
				feature[index] += 1
			return feature / np.sum(feature)

		histGram_list = np.array([get_feature(descriptors) for descriptors in self.data_descriptors])
			
		
		self.input_train=histGram_list[self.index_train]
		self.correct_train=self.correct_data[self.index_train]
		self.input_test=histGram_list[self.index_test]
		self.correct_test=self.correct_data[self.index_test]
	
		return self.input_train,self.correct_train,self.input_test,self.correct_test
	
	def get_codebook(self,label_list,dataDir_list,img_w,img_h ):
	
		correct,data_descriptors,images = local_features.img2descriptors(label_list,dataDir_list,img_w,img_h)
		
		n_data = len(correct)	
		correct_data = np.zeros((n_data,2))
		for i in range(n_data):
    
			correct_data[i,correct[i]] = 1.0
		
		def get_feature(descriptors):
	
			#featureは、ヒストグラムの横軸。コードブックの種類数分がヒストグラムのbin数になるのでまずは、全てのヒストグラムを０に初期化
			feature = np.zeros(self.codebook_size)	
			for index in self.kmeans.predict(descriptors.astype(float)):
				#コードブック番号に該当するbinの頻度を１つ足す
				feature[index] += 1
			return feature / np.sum(feature)

		bovw_features = np.array([get_feature(descriptors) for descriptors in data_descriptors])
		return bovw_features,correct_data,images
			








