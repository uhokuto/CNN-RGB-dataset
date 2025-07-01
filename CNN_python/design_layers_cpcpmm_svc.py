import numpy as np
import class_layers
from sklearn.linear_model import SGDClassifier

class design:
	 # -- 各層の初期化 --

	def __init__(self, img_ch, img_h, img_w,output_dim,class_layers=class_layers):
		self.img_ch, self.img_h, self.img_w,self.output_dim=img_ch, img_h, img_w,output_dim
		# n_flt:フィルタ数, flt_h:フィルタ高さ, flt_w:フィルタ幅
		# stride:ストライド幅, pad:パディング幅
		self.n_flt, self.flt_h, self.flt_w, self.stride, self.pad =10, 3, 3, 1, 1
		self.cl_1 = class_layers.ConvLayer(img_ch, img_h, img_w,self.n_flt, self.flt_h, self.flt_w, self.stride, self.pad )
		
		# pool:プーリング領域のサイズ, pad:パディング幅
		self.pool, self.pad = 2,0
		self.pl_1 = class_layers.PoolingLayer(self.cl_1.y_ch, self.cl_1.y_h, self.cl_1.y_w,self.pool, self.pad)
		self.n_fc_in = self.pl_1.y_ch * self.pl_1.y_h * self.pl_1.y_w
		
		self.hidden1_nodes = 100
		self.ml_1 = class_layers.MiddleLayer(self.n_fc_in,self.hidden1_nodes)
		self.hidden2_nodes = 10
		self.ml_2 = class_layers.MiddleLayer(self.hidden1_nodes,self.hidden2_nodes)
		self.ol_1 = class_layers.SVCLayer(self.hidden2_nodes,self.output_dim)
		
		#svmレイヤの活性化関数オブジェクト生成
		#self.clf = SGDClassifier(loss='hinge')


# -- 順伝播 --

	def forward_propagation(self,x,batchsize):
		self.n_bt = batchsize
		self.images = x.reshape(self.n_bt, self.img_ch, self.img_h, self.img_w)
		self.cl_1.forward(self.images)
		self.pl_1.forward(self.cl_1.y)
		self.fc_input = self.pl_1.y.reshape(self.n_bt, -1)   
		self.ml_1.forward(self.fc_input)
		self.ml_2.forward(self.ml_1.y)
		self.ol_1.forward(self.ml_2.y)

# -- 逆伝播 --
	def backpropagation(self,given_output):
		self.t = given_output
		self.ol_1.backward(self.t)
		self.ml_2.backward(self.ol_1.grad_x)
		self.ml_1.backward(self.ml_2.grad_x)
		self.grad_img = self.ml_1.grad_x.reshape(self.n_bt, self.pl_1.y_ch, self.pl_1.y_h, self.pl_1.y_w)
		self.pl_1.backward(self.grad_img)
		self.cl_1.backward(self.pl_1.grad_x)

# -- 重みとバイアスの更新 --
	def uppdate_wb(self,eta):
		self.eta = eta
		self.cl_1.update(self.eta)
		self.ml_1.update(self.eta)
		self.ml_2.update(self.eta)
		self.ol_1.update(self.eta)

# -- 誤差を計算 --
	def get_error(self,t):
		
		#t=np.argmax(t, axis=1)
		#self.clf.partial_fit(self.ol_1.y, t, classes=[0, 1])
		return np.sqrt(np.sum((self.ol_1.svc_clf.decision_function(self.ol_1.y))**2))
		#return np.sum(self.ol_1.svc_clf.decision_function(self.ol_1.y))/n_sample
		#return -np.sum(t * np.log(self.ol_1.y + 1e-7)) / self.n_bt # 交差エントロピー誤差

# -- サンプルを順伝播 --

	def forward_sample(self,input, correct, n_sample):
		index_rand = np.arange(len(correct))
		np.random.shuffle(index_rand) 
		index_rand = index_rand[:n_sample]
		x = input[index_rand, :]		
		t = correct[index_rand, :]		
		self.forward_propagation(x,n_sample)
		return x, t

