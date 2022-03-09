import cv2
import numpy as np
import h5py
from matplotlib import pyplot

def extract_data(root,key):
    with h5py.File(root,'r') as f:
        images = f[key][:]
 
    image_num = len(images)
    for  j in range(8):
        str_name = '/home/tangxu/SDP/Data/DualSource/Muldata' + '/' + dir_name[j]
        if os.path.exists(str_name) == False:
            os.makedirs(str_name)
        for i in range(10000):
            img = images[i+10000*j,...].transpose((2, 1, 0))
            
            file = str_name +  '/'+dir_name[j] + str(i)+'.png'
            img = img.astype('uint8')
            cv2.imwrite(file, img)
            
def extract_Pandata(root,key):
    with h5py.File(root,'r') as f:
        images = f[key][:]
 
    image_num = len(images)
    for  j in range(8):
        str_name = '/home/tangxu/SDP/Data/DualSource/Pandata' + '/' + dir_name[j]
        if os.path.exists(str_name) == False:
            os.makedirs(str_name)
        for i in range(10000):
            img = images[i + 10000*j,...].transpose((2, 1, 0))
            img = np.concatenate((img, img, img,img), axis=-1)
            file = str_name +  '/'+dir_name[j] + str(i)+'.png'
            img = img.astype('uint8')
            cv2.imwrite(file, img)