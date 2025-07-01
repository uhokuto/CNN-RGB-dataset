import augmentation
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

cat = 'C:/Users/Uhokuto/Dropbox/Python/bag_of_visual_words/catdog/img/0/cat.4.jpg'
#image = cv2.imread(cat)[:,:,::-1]
img = Image.open(cat)
#resizeの引数はwidth, height イメージ画像のデータもこの並びになっている
#img_resize = img.resize((img_w,img_h))
image = np.asarray(img)

aug = augmentation.aug(image)
for i in range(20):

	
	shear_range =0.2
	sheared = aug.affine(shear_range)
	plt.clf()
	plt.imshow(sheared)
	plt.show()
	shear_range =0.3
	sheared=aug.horizontal_shift(shear_range)
	plt.clf()
	plt.imshow(sheared)
	plt.show()
	sheared = aug.horizontal_flip()
	plt.clf()
	plt.imshow(sheared)
	plt.show()
	crop_rate=0.7
	sheared = aug.random_crop(crop_rate)
	plt.clf()
	plt.imshow(sheared)
	plt.show()
	scale_ratio=0.2
	sheared = aug.random_resize(scale_ratio)
	plt.clf()
	plt.imshow(sheared)
	plt.show()