#局所特徴量の種類を引数として、画像1枚あたりの局所特徴量ベクトル集合（つまり行列）を戻り値とする
#局所特徴量は、akaze,sift,hogの3種類
import numpy as np
from skimage import io, color
from skimage.color import rgb2gray
from skimage.feature import hog

import cv2

def get_Akaze_descriptors(img):
	
	akaze = cv2.AKAZE_create()
	#1枚の画像に対するkeypoints分の局所特徴量
	keyPoints, feature_matrix = akaze.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
	
	return feature_matrix
	
def get_sift_descriptors(img):
	

	sift = cv2.xfeatures2d.SIFT_create()
	#1枚の画像に対するkeypoints分の局所特徴量
	keyPoints, feature_matrix = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
	
	return feature_matrix


def get_Hog_descriptors(img):
	orientations = 9
	pixels_per_cell = (8, 8)
	cells_per_block = (3, 3)
	#1枚の画像に対するHog局所特徴量が全部つながったリストになって出力
	feature_vector = hog(rgb2gray(img), orientations, pixels_per_cell, cells_per_block)
	#print('vector',feature_vector.shape)
	#上記のリストを、hog局所特徴量１つを1行になるように行列に変換　9つの方向に9個のセルを動かすので1個のhog局所を構成する
	#特徴量は81
	#orientations 9 x cells_per_block 3 x 3 なのでnp.multiply=81となり、残りの次元数が行数になる
	feature_matrix=feature_vector.reshape(-1, np.multiply(*cells_per_block) * orientations)
	
	return feature_matrix

def feature(local_feature,img):
	if local_feature=='hog':
		feature_matrix = get_Hog_descriptors(img)
	elif local_feature=='sift':		
		feature_matrix =get_sift_descriptors(img)
	elif local_feature=='akaze':
		feature_matrix =get_Akaze_descriptors(img)
	
	return feature_matrix