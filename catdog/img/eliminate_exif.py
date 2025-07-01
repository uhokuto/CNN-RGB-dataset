import shutil
import glob
import numpy as np
import piexif

def del_exif(dataDir_list):	
	
	img_list=[]
	for dataDir in dataDir_list:
		#ワイルドカードでdataディレクトリ内の全ファイル名を取得してリスト化
		files = glob.glob(dataDir + "*")
		for file in files:	
			piexif.remove(file)			
	
	return 

dogDir = './catdog/img/1/'
catDir = './catdog/img/0/'
dataDir_list=[catDir,dogDir]

del_exif(dataDir_list)
