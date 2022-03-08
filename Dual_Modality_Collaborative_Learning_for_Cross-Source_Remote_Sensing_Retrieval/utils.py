import torch
import os, glob
import random,csv
from torch.utils.data import DataLoader,Dataset
from torch.nn.parameter import Parameter
import math
from torchvision import transforms
from PIL import Image
import ast
import cv2
from dataset import dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch.utils.data.sampler import BatchSampler
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda:1')
import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable

class hetero_loss(nn.Module):
	def __init__(self, margin=0.1, dist_type = 'l2'):
		super(hetero_loss, self).__init__()
		self.margin = margin
		self.dist_type = dist_type
		if dist_type == 'l2':
			self.dist = nn.MSELoss(reduction='sum')
		if dist_type == 'cos':
			self.dist = nn.CosineSimilarity(dim=0)
		if dist_type == 'l1':
			self.dist = nn.L1Loss()
	
	def forward(self, feat1, feat2, label1, label2):
		feat_size = feat1.size()[1]
		feat_num = feat1.size()[0]
		label_num =  len(label1.unique())
		feat1 = feat1.chunk(label_num, 0)
		feat2 = feat2.chunk(label_num, 0)
		#loss = Variable(.cuda())
		for i in range(label_num):
			center1 = torch.mean(feat1[i], dim=0)
			center2 = torch.mean(feat2[i], dim=0)
			if self.dist_type == 'l2' or self.dist_type == 'l1':
				if i == 0:
					dist = max(0, self.dist(center1, center2) - self.margin)
				else:
					dist += max(0, self.dist(center1, center2) - self.margin)
			elif self.dist_type == 'cos':
				if i == 0:
					dist = max(0, 1-self.dist(center1, center2) - self.margin)
				else:
					dist += max(0, 1-self.dist(center1, center2) - self.margin)

		return dist
class dictionary_w(nn.Module):
    def __init__(self, in_channels, out_channels,bias = False):
        super(dictionary_w,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.randn(in_channels,out_channels))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.fc = nn.Sequential(
#             nn.Linear(2048*7*7,2048),
#             nn.Linear(2048,1000),
#             nn.ReLU(True),
#             nn.Dropout(),
            nn.Linear(200,8)
        )
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)  
    
    def forward(self, x):
        M_ = torch.sigmoid(torch.mul(self.weight.T,self.weight)).to(device)
        M_1 = (M_ + (1-M_.detach()))*(M_>0.5).float()
        x = torch.mm(x,(M_1 + (M_1 == 0).float()* torch.eye(200).to(device)))
        fc = self.fc(x)
        M_0 = (M_ - M_.detach())*(M_<0.5).float()
        MM = M_1 + M_0
#         y = torch.abs(torch.sum(MM)/(self.in_channels*self.out_channels) - 0.05)# * 10
        y = torch.abs(torch.exp(torch.sum(MM)/(self.in_channels*self.out_channels) - 0.02)-1)*10
#         y = torch.sum(torch.abs(self.weight))
        return x,fc,y
class ap_map(object):
    def __init__(self, mulModel, panModel,sharedModel,dic,testloader,allloader,idx):
        self.mulModel = mulModel
        self.panModel = panModel
        self.sharedModel  = sharedModel
        self.dic = dic
        self.testloader = testloader
        self.all_loader = allloader
#         self.W_m = W_m
#         self.W_p = W_p
# #         self.length = len(self.testloader.dataset)
#         self.modelG2 = modelG2
#         self.modelT = modelT
        self.idx  = idx
        self.feature_p_t, self.feature_m_a,self.feature_m_t,self.feature_p_a,self.label_test,self.label_all = self.load_data()
        print(self.label_test.shape, self.label_all.shape)
        print("M->P:shape of M:",self.feature_m_t.shape, ",shape of P:",self.feature_p_a.shape)
        print("P->M:shape of P:",self.feature_p_t.shape, ",shape of M:",self.feature_m_a.shape)
        
        self.euli_dis_m_p = self.euclidean_dist(self.feature_m_t,self.feature_p_a)
        self.ap_m_p = self.compute_ap(self.euli_dis_m_p)
        print("AP of M->P:",self.ap_m_p)
        self.map_m_p = self.compute_map(self.euli_dis_m_p)
        print("MAP of M->P:",self.map_m_p)
        
        self.euli_dis_p_m = self.euclidean_dist(self.feature_p_t,self.feature_m_a)
        self.ap_p_m = self.compute_ap(self.euli_dis_p_m)
        print("AP of P->M:",self.ap_p_m)
        self.map_p_m = self.compute_map(self.euli_dis_p_m)
        print("MAP of P->M:",self.map_p_m)

    def compute_PR(self,euli_dis):
        rank = np.argsort(euli_dis)
        Precision = []
        Recall = []
        pr = 0
        recall = 0
        for i in tqdm(range(self.K)):
            cnt = 0
            index = 0
            for j in rank[i][0:]:
                index = index + 1
                if self.label_test[i] == self.label_all[j]:
                    cnt = cnt + 1
                pr = cnt/index
                recall = cnt/9725
                Precision.append(pr)
                Recall.append(recall)
        P = np.array(Precision)
        P = P.reshape(self.K,75000)
        P = np.sum(P,axis = 0)/self.K
        R = np.array(Recall)
        R = R.reshape(self.K,75000)
        R = np.sum(R,axis = 0)/self.K
        return P,R,Precision
    def compute_map(self,euli_dis):
        
        rank = np.argsort(euli_dis)
        p = []
        pr = 0
        for i in range(75000):
            cnt = 0
            pre  = 0
            index = 0
            for j in rank[i][0:self.idx]:
                index = index + 1
                if self.label_test[i] == self.label_all[j]:
                    cnt = cnt + 1
                    pre = pre + cnt/index
            precision = pre / (cnt + 0.0000001) #查准率
            p.append(precision)
        precision_ = np.array(p)
        map_ = np.sum(precision_)/len(p)#map
    def euclidean_dist(self,x,y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        # xx经过pow()方法对每单个数据进行二次方操作后，
        #在axis=1 方向（横向，就是第一列向最后一列的方向）加和，
        #此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        # yy会在最后进行转置的操作
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT 
        dist.addmm_(1, -2, x, y.t())
        # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist
    def load_data(self):
        label_t = torch.zeros(1)
        label_t = label_t.long()
        
        label_all = torch.zeros(1)
        label_all = label_all.long()
        
        feature_m_t = torch.empty(1,200)
        feature_p_t = torch.empty(1,200)
        
        feature_m_all = torch.empty(1,200)
        feature_p_all = torch.empty(1,200)
        
        for x,y,z in tqdm(self.testloader):
            x,y = x.to(device),y.to(device)
            label_t = torch.cat((label_t,z))
            with torch.no_grad():
                mul_s,mul_p = self.mulModel(x)[0:2]
                pan_s,pan_p = self.panModel(y)[0:2]
                
                mul_ss ,pan_ss= self.sharedModel(mul_s,pan_s)[0:2]
                mul_fea,pan_fea,loss_l1_x,loss_l1_y,x_fc,y_fc = self.dic(mul_ss,pan_ss)
                feature_m_t = torch.cat([feature_m_t, mul_fea.cpu()], 0)
                feature_p_t = torch.cat([feature_p_t,pan_fea.cpu()], 0)
                
        feature_m_t,feature_p_t = feature_m_t[1:], feature_p_t[1:]
#         print(feature_m.shape, feature_p.shape)
        for x,y,z in tqdm(self.all_loader):
            x,y = x.to(device),y.to(device)
            label_all = torch.cat((label_all,z))
            with torch.no_grad():
                mul_s,mul_p = self.mulModel(x)[0:2]
                pan_s,pan_p = self.panModel(y)[0:2]
                
                mul_ss,pan_ss= self.sharedModel(mul_s,pan_s)[0:2]
                
                mul_fea,pan_fea,loss_l1_x,loss_l1_y,x_fc,y_fc = self.dic(mul_ss,pan_ss)
                feature_m_all = torch.cat([feature_m_all, mul_fea.cpu()], 0)
                feature_p_all = torch.cat([feature_p_all, pan_fea.cpu()], 0)
                
        feature_m_all,feature_p_all = feature_m_all[1:], feature_p_all[1:]
        label_all,label_test = label_all[1:], label_t[1:]
        label_all = label_all.numpy()
        label_test = label_test.numpy()
        return feature_p_t, feature_m_all, feature_m_t, feature_p_all,label_test,label_all
class test_ap_map(object):
    def __init__(self, mulModel,panModel,sharedModel,W_m,W_p,fusion,testloader,idx,K):#,modelW1,modelW2,modelG2,modelT,,sharedModel
        self.mulModel = mulModel
        self.panModel = panModel
        self.sharedModel  = sharedModel
        self.testloader = testloader
        self.fusion = fusion
        self.idx = idx
#         self.modelW1 = modelW1
#         self.modelW2 = modelW2
        self.W_m = W_m
        self.W_p = W_p
    
#         self.modelG2 = modelG2
#         self.modelT = modelT
        self.K  = K
        self.feature_m_test,self.feature_p_test,self.feature_m,self.feature_p,self.label_test,self.label_all = self.load_data()
        print(self.label_test.shape,self.label_all.shape,self.feature_m.shape,self.feature_p.shape)
#         print(self.feature_m.shape)
#         self.labels = self.load_label()
        self.euli_dism_p = self.euclidean_dist(self.feature_m_test,self.feature_p)
        self.euli_disp_m = self.euclidean_dist(self.feature_p_test,self.feature_m)
#         self.M_P = self.compute_PR(self.euli_dism_p)
#         self.P_M = self.compute_PR(self.euli_disp_m)
        self.mapm_p = self.compute_map(self.euli_dism_p)
        self.mapp_m = self.compute_map(self.euli_disp_m)
    
#         self.Precision = Precision
#     def compute_class_Map(self): 
#         P = self.Precision.reshape(self.K,5000)
#         self.label_test
    def compute_PR(self,euli_dis):
        rank = np.argsort(euli_dis)
        Precision = []
        Recall = []
        pr = 0
        recall = 0
        for i in tqdm(range(self.K)):
            cnt = 0
            index = 0
            for j in rank[i][0:]:
                index = index + 1
                if self.label_test[i] == self.label_all[j]:
                    cnt = cnt + 1
                pr = cnt/index
                recall = cnt/625
                Precision.append(pr)
                Recall.append(recall)
        P = np.array(Precision)
        P = P.reshape(self.K,5000)
        P = np.sum(P,axis = 0)/self.K
        R = np.array(Recall)
        R = R.reshape(self.K,5000)
        R = np.sum(R,axis = 0)/self.K
        return P,R,Precision
    def compute_map(self,euli_dis):
        
        rank = np.argsort(euli_dis)
        p = []
        pr = 0
        for i in range(5000):
            cnt = 0
            pre  = 0
            index = 0
            for j in rank[i][0:self.idx]:
                index = index + 1
                if self.label_test[i] == self.label_all[j]:
                    cnt = cnt + 1
                    pre = pre + cnt/index
            precision = pre / (cnt + 0.0000001) #查准率
            p.append(precision)
        precision_ = np.array(p)
        map_ = np.sum(precision_)/len(p)#map

        return precision_, map_
    def euclidean_dist(self,x,y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        # xx经过pow()方法对每单个数据进行二次方操作后，
        #在axis=1 方向（横向，就是第一列向最后一列的方向）加和，
        #此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        # yy会在最后进行转置的操作
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT 
        dist.addmm_(1, -2, x, y.t())
        # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist
    def load_data(self):
        label = torch.zeros(1)
        label = label.long()
        
        feature_m_test = torch.empty(1,200)
        feature_p = torch.empty(1,200)
        feature_m = torch.empty(1,200)
        feature_p_test = torch.empty(1,200)
        
        for x,y,z in tqdm(self.testloader):
            x,y = x.to(device),y.to(device)
            label = torch.cat((label,z))
            with torch.no_grad():
                mul_mid,mul_pp,_ = self.mulModel(x)
                pan_mid,pan_pp,_ = self.panModel(y)
                
                mul_ss,pan_ss= self.sharedModel(mul_mid,pan_mid)[0:2]

                m_P = torch.mm(mul_pp,self.W_m)
                feature_m_test_ = torch.cat([mul_ss, m_P], 1)
                feature_m_ = torch.cat([mul_ss, mul_pp],1)
                
                p_M = torch.mm(pan_pp,self.W_p)
                feature_p_test_ = torch.cat([pan_ss, p_M], 1)
                feature_p_ = torch.cat([pan_ss, pan_pp],1)
                
                feature_mm_test_ = self.fusion(feature_m_test_)
                feature_pp_test_ = self.fusion(feature_p_test_)
                feature_mm = self.fusion(feature_m_)
                feature_pp = self.fusion(feature_p_)
                
                feature_m_test = torch.cat([feature_m_test,feature_mm_test_.cpu()], 0)
                feature_p_test = torch.cat([feature_p_test,feature_pp_test_.cpu()], 0)
                feature_m = torch.cat([feature_m,feature_mm.cpu()], 0)
                feature_p = torch.cat([feature_p,feature_pp.cpu()], 0)
                
        feature_m_test,feature_p_test,feature_m,feature_p = feature_m_test[1:],feature_p_test[1:],feature_m[1:], feature_p[1:]
        
        label_all,label_test = label[1:], label[1:]
        label_all = label_all.numpy()
        label_test = label_test.numpy()
        return feature_m_test,feature_p_test,feature_m, feature_p,label_test,label_all
class Tripinput(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """
    def __init__(self, dataset,resize):
        self.dataset = dataset
        self.resize = resize
#         self.mode = self.dataset.mode
#         self.transform = self.mnist_dataset.transform
        self.labels = self.dataset.labels
        self.mul_data = self.dataset.mul_imgs
        self.pan_data = self.dataset.pan_imgs
        self.label_set = set(np.array(self.labels))#.numpy()
        self.label_to_indices = {label:np.where(self.labels == label)[0] for label in self.label_set}
        
#         self.mul_tripe, self.pan_tripe = 
    def __getitem__(self,index):
        mul_img_a, pan_img_a, label_a = self.mul_data[index],self.pan_data[index], self.labels[index]#.item()
#         return(label_a,self.label_to_indices[label_a])
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label_a])
#         return positive_index
        negative_label = np.random.choice(list(self.label_set - set([label_a])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        mul_img_p = self.mul_data[positive_index]
        pan_img_p = self.pan_data[positive_index]
        mul_img_n = self.mul_data[negative_index]
        pan_img_n = self.pan_data[negative_index]
#         return mul_img_a,pan_img_a,mul_img_p,pan_img_p,mul_img_n,pan_img_n
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
        mul_img_a = tf(mul_img_a)
        mul_img_p = tf(mul_img_p)
        mul_img_n = tf(mul_img_n)
        pan_img_a = tf(pan_img_a)
        pan_img_p = tf(pan_img_p)
        pan_img_n = tf(pan_img_n)
        label_a = torch.tensor(label_a)
        negative_label = torch.tensor(negative_label)
        return (label_a,label_a,negative_label),\
                (mul_img_a,mul_img_p,mul_img_n),\
                (pan_img_a,pan_img_p,pan_img_n)
#                 (mul_img_a,pan_img_p,pan_img_n),\
#                 (pan_img_a,mul_img_p,mul_img_n)
      
    def __len__(self):
        return len(self.dataset)
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
#         loader = DataLoader(dataset)
#         self.labels_list = []
#         for _,__, label in loader:
#             self.labels_list.append(label)
        self.labels = torch.LongTensor(dataset.labels)
        self.labels_set = list(set(self.labels.numpy()))
#         print(self.labels_set)[0,1,2,3,4,5,6,7]
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
#         print(self.label_to_indices.shape) #:对于0——7标签，对应的索引
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
    
class process_feature(object):
    def __init__(self, fpS,fmS,fmP, fpP):
        self.fpS = fpS
        self.fmS = fmS
        self.fmP = fmP
        self.fpP = fpP
        self.aS_M = self.affinity(fpS,fmS)
        self.aM_S = self.affinity(fmS, fpS)
        self.aM = self.affinity(fmP,fmP)
        self.aP = self.affinity(fpP,fpP)
        self.affinity_matrix = self.concate_affinity(self.aM_S,self.aS_M,self.aM,self.aP)
        self.normalize_adj = self.normalize_adj_torch(self.affinity_matrix)
        self.confusion = self.feature_cat()
    def affinity(self,x,y):   
        x = F.normalize(x, p = 2, dim = 1)
        y = F.normalize(y, p = 2, dim = 1)
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        euclidean_dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        A = 1 - 0.5*euclidean_dist
        value, indice = A.topk(4,dim=1,largest=True,sorted=True)
        B = torch.zeros((m,n))
    #     sq = np.argpartition(A, np.argmin(A, axis=0))[:, -4:]
        for i in range(m):
            for j in indice[i]:
                B[i][j] = A[i][j]
        return B
    def normalize_adj_torch(self,mx):
    #     mx = mx.to_dense()           #构建张量
        rowsum = mx.sum(1)           #每行的数加在一起
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()   #输出rowsum ** -1/2
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.         #溢出部分赋值为0
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)         #对角化
        mx = torch.matmul(mx, r_mat_inv_sqrt)
        mx = torch.transpose(mx, 0, 1)                   #转置
        mx = torch.matmul(mx, r_mat_inv_sqrt)
        return mx
    def concate_affinity(self,aM_S,aS_M,aM,aP):
        accord_row_aM_aS = torch.cat((aM,aS_M),0)
        accord_row_aS_aP = torch.cat((aM_S,aP),0)
        accord_col = torch.cat((accord_row_aM_aS,accord_row_aS_aP),1)
        return accord_col
    def feature_cat(self):
        size_zeros0,size_zeros1 = self.fpP.shape[0],self.fpP.shape[1]
        add = torch.zeros(size_zeros0,size_zeros1).to(device)
        fea_fusion_m = torch.cat((self.fmP,self.fmS,add), 1)
        fea_fusion_p = torch.cat((add,self.fpS,self.fpP), 1)
        fusion = torch.cat((fea_fusion_m,fea_fusion_p),0)
        return fusion
def edu_dis(x,y):   
    x = F.normalize(x, p = 2, dim = 1)
    y = F.normalize(y, p = 2, dim = 1)
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    euclidean_dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return euclidean_dist
def similarity(x,y):   
    x = F.normalize(x, p = 2, dim = 1)
    y = F.normalize(y, p = 2, dim = 1)
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    euclidean_dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    A = 1 - 0.5*euclidean_dist
    return A
