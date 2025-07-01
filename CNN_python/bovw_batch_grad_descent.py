
import numpy as np
import matplotlib.pyplot as plt
import design_layers_mm
import bovw_datasets


dogDir = 'C:/Users/Uhokuto/Dropbox/Python/bag_of_visual_words/catdog/img/1/'
catDir = 'C:/Users/Uhokuto/Dropbox/Python/bag_of_visual_words/catdog/img/0/'
dataDir_list=[catDir,dogDir]
label_list=[0,1]

split_rate=0.3
codebook_size=100
img_w=128
img_h=128
bovw_data = bovw_datasets.bovw(label_list,dataDir_list,img_w,img_h,split_rate,codebook_size)

input_train,correct_train,input_test,correct_test = bovw_data.create_train_test()

n_train = input_train.shape[0]  # 訓練データのサンプル数
n_test = input_test.shape[0]  # テストデータのサンプル数



#wb_width = 0.1  # 重みとバイアスの広がり具合
eta = 0.01  # 学習係数
epoch = 5
batch_size = 10

n_sample = 100  # 誤差計測のサンプル数
output_dim =2 # 出力層の次元数


nn1 = design_layers_mm.design(codebook_size,output_dim)
   
# -- 誤差の記録用 --
train_error_x = []
train_error_y = []
test_error_x = []
test_error_y = []

# -- 学習と経過の記録 --
n_batch = n_train // batch_size
for i in range(epoch):

    # -- 誤差の計測 -- 
    x, t = nn1.forward_sample(input_train,correct_train,n_sample)
    error_train = nn1.get_error(t)
    
    x, t = nn1.forward_sample(input_test, correct_test, n_sample) 
    error_test = nn1.get_error(t)
    
    # -- 誤差の記録 -- 
    train_error_x.append(i)
    train_error_y.append(error_train) 
    test_error_x.append(i)
    test_error_y.append(error_test) 
    
    # -- 経過の表示 --
    print("Epoch:" + str(i+1) + "/" + str(epoch),
              "Error_train:" + str(error_train),
              "Error_test:" + str(error_test))
    
    # -- 学習 -- 
    index_rand = np.arange(n_train)
    np.random.shuffle(index_rand)   
    for j in range(n_batch):
        
        mb_index = index_rand[j*batch_size : (j+1)*batch_size]
        x = input_train[mb_index, :]
        t = correct_train[mb_index, :]

        nn1.forward_propagation(x,batch_size)
        nn1.backpropagation(t)        
        nn1.uppdate_wb(eta) 
            
    
# -- 誤差の記録をグラフ表示 -- 
plt.plot(train_error_x, train_error_y, label="Train")
plt.plot(test_error_x, test_error_y, label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")

plt.show()

# -- 正解率の測定 -- 
#x, t = forward_sample(input_train, correct_train, n_train) 
nn1.forward_propagation(input_train,n_train)
count_train = np.sum(np.argmax(nn1.ol_1.y, axis=1) == np.argmax(correct_train, axis=1))
nn1.forward_propagation(input_test,n_test)
count_test = np.sum(np.argmax(nn1.ol_1.y, axis=1) == np.argmax(correct_test, axis=1))
#x, t = forward_sample(input_test, correct_test, n_test) 
#count_test = np.sum(np.argmax(ol_1.y, axis=1) == np.argmax(t, axis=1))

print("Accuracy Train:", str(count_train/n_train*100) + "%",
      "Accuracy Test:", str(count_test/n_test*100) + "%")


dogDir = 'C:/Users/Uhokuto/Dropbox/Python/bag_of_visual_words/catdog/test_img/1/'
catDir = 'C:/Users/Uhokuto/Dropbox/Python/bag_of_visual_words/catdog/test_img/0/'
dataDir_list=[catDir,dogDir]
label_list=[0,1]

input_data,correct,images = bovw_data.get_codebook(label_list,dataDir_list,img_w,img_h)

nn1.forward_propagation(input_data,len(input_data))
predicts = np.argmax(nn1.ol_1.y, axis=1) 
trues = np.argmax(correct, axis=1)
print(correct.shape)

fig=plt.figure()

for i,(image,pred,true) in enumerate(zip(images,predicts,trues)):
	
	ax = fig.add_subplot(5,4,i+1)
	#imshow()の引数はnumpy配列
	ax.imshow(image)
	ax.axis("off")
	# 正解ラベルを取り出す
	if pred ==0:
		prediction ='cat'
	else:
		prediction ='dog'
	if true ==0:
		actual = 'cat'
	else:
		actual = 'dog'
	ax.text(0.4,0.4, 'Correct Label:'+ actual +'\n Predict Value:'+ prediction, size = 8,linespacing = 1)
plt.subplots_adjust(wspace=0.4, hspace=0.6)	
plt.show()



