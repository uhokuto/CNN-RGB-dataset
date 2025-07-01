from pathlib import Path
import augmentation
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import Compose
from PIL import Image
import numpy as np
import cv2
import torch
import random

# __getitem__でiterを明示せずにデータ収集できるか
class image_feature(Dataset):
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

    def __init__(self, img_dir,cuda_enabled,augment):
        # 画像ファイルのパス一覧を取得する。
        img_dir = Path(img_dir)
        dir_list = img_dir.glob('*/*')        
        self.img_paths = [ p for p in dir_list if p.suffix in image_feature.IMG_EXTENSIONS]
        
        path = self.img_paths
        self.labels =[]
        self.images =[]
        self.transform=Compose([transforms.Resize((224,224), interpolation=Image.BICUBIC), transforms.ToTensor(), \
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        def build_dataset(label, img):
            img=Image.fromarray(img)
            img = self.transform(img)
            self.labels.append(label)
            self.images.append(img)
            
            
        
        for i in range(len(path)):
            image = Image.open(path[i])        
            img = np.asarray(image)       
            label = int(str(path[i]).split("\\")[-2])
            build_dataset(label,img)
            if augment ==0:
                continue
            aug = augmentation.aug(img,cuda_enabled)
            img_h_flip = aug.horizontal_flip()
            build_dataset(label,img_h_flip)
            for j in range(1):
                shear_range =0.2
                img_affine = aug.affine(shear_range)
                build_dataset(label,img_affine)
                shift_ratio = 0.3
                img_shift = aug.horizontal_shift(shift_ratio)
                build_dataset(label,img_shift)
                crop_rate=0.7
                img_crops = aug.random_crop(crop_rate)
                build_dataset(label,img_crops)
            
        self.labels=torch.tensor(self.labels, dtype=torch.long)

    def __getitem__(self, index):
        
        img = self.images[index]
        label = self.labels[index]
        
        
        return img, label

    def __len__(self):
        """ディレクトリ内の画像ファイルの数を返す。
        """
        return len(self.images)

if __name__ == '__main__':
    cuda_enabled=cv2.cuda.getCudaEnabledDeviceCount()
    print('Enabled CUDA devices:',cuda_enabled) # 1
    dataset = image_feature("C:/Users/uhoku/Dropbox/Python/bag_of_visual_words/catdog/img/",cuda_enabled,0)
    print(len(dataset))
    print(dataset[0])
    
    im = ToNDarray(dataset[0][0])
    plt.imshow(im)