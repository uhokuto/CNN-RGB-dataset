import numpy as np
import class_layers

class design:
	 # -- 各層の初期化 --

	def __init__(self,input_dim,output_dim,class_layers=class_layers):
		
		self.output_dim=output_dim
		self.hidden1_nodes = 100
		self.ml_1 = class_layers.MiddleLayer(input_dim,self.hidden1_nodes)
		self.hidden2_nodes = 10
		self.ml_2 = class_layers.MiddleLayer(self.hidden1_nodes,self.hidden2_nodes)
		self.ol_1 = class_layers.OutputLayer(self.hidden2_nodes,self.output_dim)


# -- 順伝播 --

	def forward_propagation(self,x,batch_size):
		self.n_bt=batch_size
		self.ml_1.forward(x)
		self.ml_2.forward(self.ml_1.y)
		self.ol_1.forward(self.ml_2.y)

# -- 逆伝播 --
	def backpropagation(self,given_output):
		self.t = given_output
		self.ol_1.backward(self.t)
		self.ml_2.backward(self.ol_1.grad_x)
		self.ml_1.backward(self.ml_2.grad_x)
		

# -- 重みとバイアスの更新 --
	def uppdate_wb(self,eta):
		self.eta = eta
		
		self.ml_1.update(self.eta)
		self.ml_2.update(self.eta)
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

