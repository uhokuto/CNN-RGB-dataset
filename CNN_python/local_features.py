import shutil
import glob
import numpy as np
from skimage import io, color,transform
import execute_local_features
from PIL import Image

def img2descriptors(label_list,dataDir_list,img_w,img_h):

	labels=[]
	data_descriptors=[]
	img_list=[]
	for label,dataDir in zip(label_list,dataDir_list):
		#ワイルドカードでdataディレクトリ内の全ファイル名を取得してリスト化
		files = glob.glob(dataDir + "*")
		for file in files:	
			labels.append(label)
			#img = io.imread(file)
			#img_list.append(img)
			img = Image.open(file)
			#resizeの引数はwidth, height イメージ画像のデータもこの並びになっている
			img_resize = img.resize((img_w,img_h))
			array_img=np.asarray(img_resize)
			img_list.append(array_img)
			#np.arrayに変換すると、height,width,ch になる
			#	array_img=np.asarray(img_resize)
			#戻り値は、1枚の画像から生成されたHog特徴量の行列（各行が1つの局所Hog特徴量）
			#sift akaze hogのいずれかを引数にするとそれに応じた局所特徴量を計算する
			local_feature='sift'
			descriptor=execute_local_features.feature(local_feature,array_img)	
			data_descriptors.append(descriptor)
	return np.array(labels),np.array(data_descriptors),img_list

