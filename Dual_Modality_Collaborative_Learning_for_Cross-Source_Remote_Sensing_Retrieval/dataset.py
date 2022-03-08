import torch
import os, glob
import random,csv
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
import ast
import cv2
import matplotlib.pyplot as plt

class dataset(Dataset):
    def __init__(self, root, rootmul,rootpan, resize, mode):
        super(dataset, self).__init__()
        self.root = root
        self.rootmul = rootmul
        self.rootpan = rootpan
        self.resize = resize
        self.mode = mode
        self.name2label = {}
        
        for name in sorted(os.listdir(os.path.join(rootmul))):
            if not os.path.isdir(os.path.join(rootmul, name)):
                continue
            self.name2label[name] = len(self.name2label.keys()) 
#         print(self.name2label)
        self.mul_imgs, self.pan_imgs, self.labels = self.load_csv('train.csv','test.csv','/home/tangxu/SDP/Data/DualSource/image.csv')
#         print(self.pan_imgs[0:5], self.mul_imgs[0:5],self.labels[0:5])
#         if mode == 'all':
#             self.mul_imgs = self.mul_imgs[0:]
#             self.pan_imgs = self.pan_imgs[0:]
#             self.labels = self.labels[0:]
       
#         elif mode == 'train':#60%
#             self.mul_imgs = self.mul_imgs[:int(0.8*len(self.mul_imgs))]
#             self.pan_imgs = self.pan_imgs[:int(0.8*len(self.pan_imgs))]
#             self.labels = self.labels[:int(0.8*len(self.labels))]

#         elif mode =='val': #20%  60%->80%
#             self.mul_imgs =self.mul_imgs[int(0.9375*len(self.mul_imgs)):int(1*len(self.mul_imgs))]
#             self.pan_imgs = self.pan_imgs[int(0.9375*len(self.pan_imgs)):int(1*len(self.pan_imgs))]
#             self.labels = self.labels[int(0.9375*len(self.labels)):int(1*len(self.labels))]
#         elif mode == 'test1':
#             self.mul_imgs =self.mul_imgs[int(0.85*len(self.mul_imgs)):int(0.95*len(self.mul_imgs))]
#             self.pan_imgs = self.pan_imgs[int(0.85*len(self.pan_imgs)):int(0.95*len(self.pan_imgs))]
#             self.labels = self.labels[int(0.85*len(self.labels)):int(0.95*len(self.labels))]

#         elif mode == 'test': #20% 80%->100%
#             self.mul_imgs = self.mul_imgs[int(0.8*len(self.mul_imgs)):]
#             self.pan_imgs = self.pan_imgs[int(0.8*len(self.pan_imgs)):]

#             self.labels = self.labels[int(0.8*len(self.labels)):]
    def sorted_filename(self,root):
        
        full_name = []
        for name in self.name2label.keys():
            path = root + '/' + name #/home/tangxu/SDP/Data/DualSource/Muldata/aquafarm  
            fileslist = os.listdir(path) #'aquafarm92.png'

            sort_num_list = []
            for file in fileslist:        
                sort_num_list.append(int(file.split(name)[1].split('.png')[0])) #去掉前面的字符串和下划线以及后缀，只留下数字并转换为整数方便后面排序
                sort_num_list.sort() #然后再重新排序

            #接着再重新排序读取文件
            sorted_file = []
            for sort_num in sort_num_list:
                for file in fileslist:
                    if str(sort_num) == file.split(name)[1].split('.png')[0]:
                        full_name.append(path + '/' + file)
        return full_name
            
    def load_csv(self, train_filename,test_filename,all_filename):
        if not os.path.exists(os.path.join(self.root, train_filename)):
            images = []
#             print(self.b())
            mulfull_name = self.sorted_filename(self.rootmul)
            panfull_name = self.sorted_filename(self.rootpan)
#             print(self.rootmul)
#             print(panfull_name)
            for i in range(len(panfull_name)):
                images.append(mulfull_name[i] + ' ' + panfull_name[i])
                
#             random.shuffle(images[0:10000])
#             print(images)
            random_number = np.arange(10000)
            random.shuffle(random_number)
            with open(os.path.join(self.root, train_filename), mode = 'w', newline='') as f:
                writer = csv.writer(f)
                for i in tqdm(range(8)):
                    for j in random_number[625:]:
                        mul_name = images[10000*i + j].split(' ',1)[0]
                        pan_name = images[10000*i + j].split(' ',1)[1]
                        name = mul_name.split(os.sep)[-2]
                        label = self.name2label[name]
                        writer.writerow([mul_name, pan_name, label])
            print('write into csv file:', train_filename)
            with open(os.path.join(self.root, test_filename), mode = 'w', newline='') as f:
                writer = csv.writer(f)
                for i in tqdm(range(8)):
                    for j in random_number[0:625]:  
                        mul_name = images[10000*i + j].split(' ',1)[0]
                        pan_name = images[10000*i + j].split(' ',1)[1]
                        name = mul_name.split(os.sep)[-2]
                        label = self.name2label[name]
                        writer.writerow([mul_name, pan_name, label])
            print('write into csv file:', test_filename)
        mul_images, pan_images, labels  = [],[],[] 
        if self.mode == 'train':
            with open(os.path.join(self.root, train_filename)) as f:
                reader = csv.reader(f)
                for row in reader:
                    mul_img, pan_img, label = row
                    label = int(label)
                    mul_images.append(mul_img)
                    pan_images.append(pan_img)
                    labels.append(label)
                assert len(mul_images) == len(labels)
                return mul_images, pan_images, labels
        if self.mode == 'test':
             with open(os.path.join(self.root, test_filename)) as f:
                reader = csv.reader(f)
                for row in reader:
                    mul_img, pan_img, label = row
                    label = int(label)
                    mul_images.append(mul_img)
                    pan_images.append(pan_img)
                    labels.append(label)
                assert len(mul_images) == len(labels)
                return mul_images, pan_images, labels
        if self.mode == 'all':
             with open(os.path.join(self.root, all_filename)) as f:
                reader = csv.reader(f)
                for row in reader:
                    mul_img, pan_img, label = row
                    label = int(label)
                    mul_images.append(mul_img)
                    pan_images.append(pan_img)
                    labels.append(label)
                assert len(mul_images) == len(labels)
                return mul_images, pan_images, labels
    def __len__(self):
        return len(self.mul_imgs)
    def denormalize(self,x_hat):        
#         mean = [0.485, 0.456,0.406]
#         td = [0.229,0.224,0.225]
#         x_hat = (x - mean)/std
#         x = x_hat*std + mean
#         x:[c, h ,w]
#         mean:[3] => [3,1,1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean 
        return x
    def __getitem__(self, idx):
        mul_image,pan_image,label = self.mul_imgs[idx],self.pan_imgs[idx],self.labels[idx]
#         print(mul_image,pan_image,label)
        tf = transforms.Compose([
           lambda x :  cv2.imread(x,cv2.IMREAD_UNCHANGED),
            transforms.ToPILImage(),#.convert('RGB'),#Image.open(x),
            transforms.Resize((self.resize, self.resize)),
#             transforms.RandomRotation(15), #旋转15°
# #             transforms.CenterCrop(self.resize),#中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5,0.5,0.5,0.5],
                                std = [0.5,0.5,0.5,0.5]) #归一化
        ])
        mul_image = tf(mul_image)
        pan_image = tf(pan_image)
        label = torch.tensor(label)
#         plt.imshow(mul_image)
#         plt.show()
        return mul_image, pan_image, label
def main():
    import visdom 
#     import time
#     viz = visdom.Visdom()
    db = dataset('/home/tangxu/SDP/Data/DualSource',
                '/home/tangxu/SDP/Data/DualSource/Muldata',
                '/home/tangxu/SDP/Data/DualSource/Pandata',224,'train')
    x, y, z = next(iter(db))
    print(x.shape)
#     viz.image(x, win = 'sample_x', opts = dict(title = 'sample_x'))
#     viz.image(db.denormalize(x), win = 'sample_x', opts = dict(title = 'sample_x'))
    loader = DataLoader(db, batch_size=32,shuffle=True, num_workers = 8)
    
    for x, y, z in loader:
        print(x.shape, y.shape, z)
    