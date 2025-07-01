import numpy as np

class digits:
	 # -- 各層の初期化 --

	def __init__(self,digits_data):
		array_datasets=np.array([data.split(',') for data in digits_data])
		# -- 手書き文字データセットの読み込み --
		#digits_data = datasets.load_digits()
		self.input_data = np.asfarray(array_datasets[:,1:])
		self.correct = np.array(array_datasets[:,0], dtype='int64')
		self.n_data = len(self.correct)
	
	def normalize(self):
		# -- 入力データの標準化 --
		ave_input = np.mean(self.input_data,axis=1)
		std_input = np.std(self.input_data,axis=1)
		self.input_data = (self.input_data - ave_input.reshape(self.n_data,-1)) / std_input.reshape(self.n_data,-1)
		
		# -- 正解をone-hot表現に --
		self.correct_data = np.zeros((self.n_data, 10))
		for i in range(self.n_data):
    
			self.correct_data[i, self.correct[i]] = 1.0
			
			
		return self.correct_data,self.input_data
		

	
	def create_train_test(self,split_rate):
		r = split_rate*10
	
		

		# -- 訓練データとテストデータ --
		index = np.arange(self.n_data)
		index_train = index[index%r != 0]
		index_test = index[index%r == 0]

		self.input_train = self.input_data[index_train, :]  # 訓練 入力
		self.correct_train = self.correct_data[index_train, :]  # 訓練 正解
		self.input_test = self.input_data[index_test, :]  # テスト 入力
		self.correct_test = self.correct_data[index_test, :]  # テスト 正解
		
		return self.input_train,self.correct_train,self.input_test,self.correct_test

	
	