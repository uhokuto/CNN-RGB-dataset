import cv2
#import matplotlib.pyplot as plt
import numpy as np

class aug:
	 # -- 各層の初期化 --

	def __init__(self,img,cuda_enabled):
	
		self.cuda_enabled =  cuda_enabled
		
		if self.cuda_enabled==1:
			self.src_mat = cv2.cuda_GpuMat()
			self.dst_mat = cv2.cuda_GpuMat()
			self.src_mat.upload(img)

		self.img = img
		
	def affine(self,shear_range):
	
		self.w =self.img.shape[1]
		self.h=self.img.shape[0]
		self.shear_range=shear_range
		
		
		h = self.img.shape[0] // 2
		w = self.img.shape[1] // 2
		randoms = np.random.uniform(1.0-shear_range, 1.0+shear_range, (3,2)).astype(np.float32) #32ビット変数にする
		coefs = np.array([[-1,-1],[1,-1],[1,1]], np.float32)
		centers = np.array([[h,w]], np.float32)
		origin = centers + centers * coefs
		dest = centers + centers * coefs * randoms
		affine_matrix = cv2.getAffineTransform(origin, dest)
		h_margin=int(h*shear_range)
		w_margin=int(w*shear_range)

		if self.cuda_enabled ==1:
			g_dst = cv2.cuda.warpAffine(self.src_mat, affine_matrix, (self.img.shape[1],self.img.shape[0]))
			image_affine = g_dst.download()
		else:
			image_affine= cv2.warpAffine(self.img, affine_matrix, (self.img.shape[1],self.img.shape[0]))

		img_crop = image_affine[h_margin:-h_margin,w_margin:-w_margin,:]
		#img_resize = cv2.resize(img_crop,(self.w,self.h))
		#print(self.img.shape)
		#print(img_crop.shape)
		
		return img_crop
		
	def horizontal_shift(self,shear_range):
	
		self.w =self.img.shape[1]
		self.h=self.img.shape[0]
		self.shear_range=shear_range
		
		
		h = self.img.shape[0] // 2
		w = self.img.shape[1] // 2
		randoms = np.random.uniform(-1*shear_range, shear_range) #32ビット変数にする
		coefs = np.array([[-1,-1],[1,-1],[1,1]], np.float32)
		centers = np.array([[h,w]], np.float32)
		origin = centers + centers * coefs
		coefs[:,0]*=(1+randoms)
		#print(randoms,w*(1+randoms))
		
		dest = centers + centers * coefs 
		affine_matrix = cv2.getAffineTransform(origin, dest)
		if self.cuda_enabled ==1:
			g_dst = cv2.cuda.warpAffine(self.src_mat, affine_matrix, (self.img.shape[1],self.img.shape[0]))
			image_affine = g_dst.download()
		else:
			image_affine= cv2.warpAffine(self.img, affine_matrix, (self.img.shape[1],self.img.shape[0]))
			
		w_margin=int(w*shear_range)
		
		if randoms <0:
			img_crop = image_affine[:,:-w_margin,:]
		else:
			img_crop = image_affine[:,w_margin:,:]
		#img_resize = cv2.resize(img_crop,(self.w,self.h))
		#print(self.img.shape)
		#print(img_crop.shape)
		
		return img_crop
		
	def horizontal_flip(self,):
		image = self.img[:, ::-1, :]
		return image
	
	def vertical_flip(self,):
		image = image[::-1, :, :]
		return image
		
	def random_crop(self, crop_rate):
		#h, w, _ = self.img.shape

		# 0~(400-224)の間で画像のtop, leftを決める
		#print(int(self.h*(1 - crop_rate)))
		top = np.random.randint(0, int(self.h*(1 - crop_rate)))
		left = np.random.randint(0, int(self.w*(1 - crop_rate)))

		# top, leftから画像のサイズである224を足して、bottomとrightを決める
		bottom = top + int(self.h*crop_rate)
		right = left + int(self.w*crop_rate)

		# 決めたtop, bottom, left, rightを使って画像を抜き出す
		image = self.img[top:bottom, left:right, :]
		return image
	
	def random_resize(self, scale_ratio):
		min_height = int(self.h - (self.h*scale_ratio))
		max_height = int(self.h + (self.h*scale_ratio))
		#print(min_height,max_height,self.h)
		min_width = int(self.w - (self.w*scale_ratio))
		max_width = int(self.w + (self.w*scale_ratio))
		height = self.h + np.random.randint(min_height,max_height)
		#print(height)
		width = self.w + np.random.randint(min_width,max_width)
		image = cv2.resize(self.img, (width, height))
		#image = random_crop(image, (crop_size, crop_size))
		return image