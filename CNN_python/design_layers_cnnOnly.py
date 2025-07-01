import numpy as np
import class_layers
from sklearn.linear_model import SGDClassifier

class design:
	 # -- 各層の初期化 --

	def __init__(self, img_ch, img_h, img_w,output_dim,class_layers=class_layers):
		self.img_ch, self.img_h, self.img_w,self.output_dim=img_ch, img_h, img_w,output_dim
		# n_flt:フィルタ数, flt_h:フィルタ高さ, flt_w:フィルタ幅
		# stride:ストライド幅, pad:パディング幅
		self.n_flt1, self.flt1_h, self.flt1_w, self.stride1, self.pad1 =20, 3, 3, 1, 1
		self.cl_1 = class_layers.ConvLayer(img_ch, img_h, img_w,self.n_flt1, self.flt1_h, self.flt1_w, self.stride1, self.pad1 )
		
		# pool:プーリング領域のサイズ, pad:パディング幅
		self.pool1, self.pool_pad1 = 2,0
		self.pl_1 = class_layers.PoolingLayer(self.cl_1.y_ch, self.cl_1.y_h, self.cl_1.y_w,self.pool1, self.pool_pad1)
		
		self.n_flt2, self.flt2_h, self.flt2_w, self.stride2, self.pad2 =10, 3, 3, 2, 1
		self.cl_2 = class_layers.ConvLayer(self.pl_1.y_ch, self.pl_1.y_h, self.pl_1.y_w, self.n_flt2, self.flt2_h, self.flt2_w, self.stride2, self.pad2 )
				
		# pool:プーリング領域のサイズ, pad:パディング幅
		self.pool2, self.pool_pad2 = 4,0
		self.pl_2 = class_layers.PoolingLayer(self.cl_2.y_ch, self.cl_2.y_h, self.cl_2.y_w, self.pool2, self.pool_pad2)
		
		
		self.n_fc_in = self.pl_2.y_ch * self.pl_2.y_h * self.pl_2.y_w
		
		
		self.ol_1 = class_layers.OutputLayer(self.n_fc_in,self.output_dim)

		#svmレイヤの活性化関数オブジェクト生成
		#self.clf = SGDClassifier(loss='hinge')


# -- 順伝播 --

	def forward_propagation(self,x,batchsize):
		self.n_bt = batchsize
		self.images = x.reshape(self.n_bt, self.img_ch, self.img_h, self.img_w)
		self.cl_1.forward(self.images)
		self.pl_1.forward(self.cl_1.y)
		self.cl_2.forward(self.pl_1.y)
		self.pl_2.forward(self.cl_2.y)
		self.fc_input = self.pl_2.y.reshape(self.n_bt, -1)   		
		self.ol_1.forward(self.fc_input)

# -- 逆伝播 --
	def backpropagation(self,given_output):
		self.t = given_output
		self.ol_1.backward(self.t)
		#self.ml_2.backward(self.ol_1.grad_x)
		#self.ml_1.backward(self.ml_2.grad_x)
		self.grad_img = self.ol_1.grad_x.reshape(self.n_bt, self.pl_2.y_ch, self.pl_2.y_h, self.pl_2.y_w)
		self.pl_2.backward(self.grad_img)
		self.cl_2.backward(self.pl_2.grad_x)
		self.pl_1.backward(self.cl_2.grad_x)
		self.cl_1.backward(self.pl_1.grad_x)

# -- 重みとバイアスの更新 --
	def uppdate_wb(self,eta):
		self.eta = eta
		self.cl_1.update(self.eta)
		self.cl_2.update(self.eta)
		#self.ml_1.update(self.eta)
		#self.ml_2.update(self.eta)
		self.ol_1.update(self.eta)

# -- 誤差を計算 --
	def get_error(self,t):
		return -np.sum(t * np.log(self.ol_1.y + 1e-7)) / self.n_bt # 交差エントロピー誤差

# -- サンプルを順伝播 --

	def forward_sample(self,input, correct, n_sample):
		index_rand = np.arange(len(correct))
		np.random.shuffle(index_rand) 
		index_rand = index_rand[:n_sample]
		x = input[index_rand, :]		
		t = correct[index_rand, :]		
		self.forward_propagation(x,n_sample)
		return x, t

