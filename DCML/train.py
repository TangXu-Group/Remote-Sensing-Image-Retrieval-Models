import torch
import torch.nn as nn
from dataset import dataset
from torch.optim import lr_scheduler
from model import Bottleneck,ResNet,shared,Transfer#,dictionary_w
from utils import BalancedBatchSampler,Tripinput,dictionary_w,train_ap_map,test_ap_map
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import numpy as np 
import scipy.sparse as sp
from torch.utils.data.sampler import BatchSampler
import os, glob
import random,csv
from PIL import Image
from torch.nn.parameter import Parameter
import math
import ast
import cv2
import torch.nn.functional as F
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
device = torch.device('cuda:0')
import itertools



train_db = dataset('/home/tangxu/SDP/Data/DualSource',
                '/home/tangxu/SDP/Data/DualSource/Muldata',
                '/home/tangxu/SDP/Data/DualSource/Pandata',224,'train')
test_db = dataset('/home/tangxu/SDP/Data/DualSource',
                '/home/tangxu/SDP/Data/DualSource/Muldata',
                '/home/tangxu/SDP/Data/DualSource/Pandata',224,'test')

trip = Tripinput(train_db,224)
# bs = BalancedBatchSampler(train_db,4,4)
# train_loader = DataLoader(train_db,batch_sampler=bs)
train_loader = DataLoader(trip, batch_size = 8,shuffle = True)
test_loader = DataLoader(test_db, batch_size = 32, shuffle = True)

# for i,j,k in train_loader:
#     print(k)
#     break
    
# val_loader = DataLoader(val_db, batch_size = 32, shuffle = False)
len_dataloader = len(train_loader.dataset)
print(len(test_loader.dataset))


import torchvision.models as models
resnet50 = models.resnet50(pretrained=True)
pretrained_dict = resnet50.state_dict()
modelM = ResNet()
modelP = ResNet()
modelS = shared()

model_dict = modelM.state_dict()
models_dict = modelS.state_dict()


pretrainedM_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} #只能对层名一致的层进行“层名：参数”键值对赋值。
# 更新现有的model_dict
model_dict.update(pretrainedM_dict)
# 加载我们真正需要的模型参数state_dict
modelM.load_state_dict(model_dict) # cnn.load_state_dict()方法对cnn初始化，其一个重要参数strict，默认为True，表示预训练模型（model_dict）的层和自己定义的网络结构（cnn）的层严格对应相等（比如层名和维度）。
modelP.load_state_dict(model_dict) # cnn.load_state_dict()方法对cnn初始化，其一个重要参数strict，默认为True，表示预训练模型（model_dict）的层和自己定义的网络结构（cnn）的层严格对应相等（比如层名和维度）。
pretrainedS_dict = {k: v for k, v in pretrained_dict.items() if k in models_dict} #只能对层名一致的层进行“层名：参数”键值对赋值。
# 更新现有的model_dict
models_dict.update(pretrainedS_dict)
# 加载我们真正需要的模型参数state_dict
modelS.load_state_dict(models_dict) # cnn.load_state_dict()方法对cnn初始化，其一个重要参数strict，默认为True，表示预训练模型（model_dict）的层和自己定义的网络结构（cnn）的层严格对应相等（比如层名和维度）。


seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)
# graph  = GraphConv(200,200)
# modelW1 = dictionary_w(600,600)
# modelW2 = dictionary_w(600,600)
modelM.to(device)
modelP.to(device)
modelS.to(device)
Dm = T().to(device)
Dp = T().to(device)
D_CM = T().to(device)
D_CP = T().to(device)
coffi = W().to(device)
W_m = torch.randn(400,400)
W_m = W_m.to(device)
W_m.requires_grad = True
# modelT = Transfer().to(device)

def adjust_learning_rate(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

map_best_acc = 0
ap_best_acc = 0
start_lr = 0.001
optimizer = torch.optim.Adam(itertools.chain(modelM.parameters(), modelP.parameters(),modelS.parameters(),\
                                             Dm.parameters(),Dp.parameters(),D_CM.parameters(),D_CP.parameters(),[W_m],coffi.parameters()
                                            ),
                             lr=start_lr, betas=(0.9, 0.999))

# # scheduler = lr_scheduler.StepLR(optimizer,2,0.1)
losses = []
triplet_loss = nn.TripletMarginLoss(margin=0.3, p=2) 
loss_fc = nn.CrossEntropyLoss().to(device) 
criterion = torch.nn.MSELoss(reduction='mean')

for epoch in tqdm(range(6)):   
    adjust_learning_rate(optimizer,epoch,start_lr)
    print("Epoch:{}  Lr:{:.2E}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    for step, (labels,M,P) in enumerate(train_loader):
        modelM.train()
        modelP.train()
        modelS.train()

        label = torch.cat((labels[0],labels[1],labels[2]))
        label = label.to(device)
        label_t = torch.cat((label,label))
        label_t = label_t.to(device)

        mul_img_a,mul_img_p,mul_img_n = M
        pan_img_a,pan_img_p,pan_img_n = P
        mul_img_a,mul_img_p,mul_img_n = mul_img_a.to(device),mul_img_p.to(device),mul_img_n.to(device)
        pan_img_a,pan_img_p,pan_img_n = pan_img_a.to(device),pan_img_p.to(device),pan_img_n.to(device)

        '''
        the output from the modelP and modelM:
        '''
        mul_a_mid,mul_a_pfea, mul_a_fc = modelM(mul_img_a)
        mul_p_mid,mul_p_pfea, mul_p_fc = modelM(mul_img_p)
        mul_n_mid,mul_n_pfea, mul_n_fc = modelM(mul_img_n)

        mul_p = torch.cat((mul_a_pfea, mul_p_pfea,mul_n_pfea),0)
        mul_fc = torch.cat((mul_a_fc,mul_p_fc,mul_n_fc),0)

        pan_a_mid,pan_a_pfea, pan_a_fc = modelP(pan_img_a)
        pan_p_mid,pan_p_pfea, pan_p_fc  = modelP(pan_img_p)
        pan_n_mid,pan_n_pfea, pan_n_fc  = modelP(pan_img_n)

        pan_p = torch.cat((pan_a_pfea,pan_p_pfea,pan_n_pfea),0)
        pan_fc = torch.cat((pan_a_fc,pan_p_fc,pan_n_fc),0)

        '''
        the output from the modelS:
        '''

        mul_as,pan_as,mul_afc, pan_afc = modelS(mul_a_mid,pan_a_mid)
        mul_ps,pan_ps,mul_pfc, pan_pfc = modelS(mul_p_mid,pan_p_mid)
        mul_ns,pan_ns,mul_nfc, pan_nfc = modelS(mul_n_mid,pan_n_mid)
        mul_ss = torch.cat((mul_as,mul_ps,mul_ns),0)
        mul_sfc = torch.cat((mul_afc,mul_pfc,mul_nfc),0)
        pan_ss = torch.cat((pan_as, pan_ps,pan_ns),0)
        pan_sfc = torch.cat((pan_afc,pan_pfc,pan_nfc),0)

        '''
        loss fucntion

        anchor: anchor input tensor
        positive: positive input tensor
        negative: negative input tensor
        p: the norm degree. Default: 2
        '''
        m_P = torch.mm(mul_p,W_m)
        p_M = torch.mm(pan_p,torch.inverse(W_m))
        m_P_fc = Dm(m_P)
        p_M_fc = Dp(p_M)
        
        M_ct = torch.cat((mul_ss, m_P),1)
        M_cta,M_ctp,M_ctn = M_ct[0:8],M_ct[8:16],M_ct[16:24]
        P_ori = torch.cat((pan_ss,pan_p),1)
        P_oria,P_orip,P_orin = P_ori[0:8],P_ori[8:16],P_ori[16:24]
        
        fusion_P_ori = coffi(P_ori)
        f_P_oria,f_P_orip,f_P_orin = fusion_P_ori[0:8],fusion_P_ori[8:16],fusion_P_ori[16:24]
        fusion_M_ct = coffi(M_ct)
        f_M_cta,f_M_ctp,f_M_ctn = fusion_M_ct[0:8],fusion_M_ct[8:16],fusion_M_ct[16:24]
        
        P_ct = torch.cat((pan_ss,p_M),1)
        P_cta,P_ctp,P_ctn = P_ct[0:8],P_ct[8:16],P_ct[16:24]
        M_ori = torch.cat((mul_ss,mul_p),1)
        M_oria,M_orip,M_orin = M_ori[0:8],M_ori[8:16],M_ori[16:24]
        
        fusion_P_ct = coffi(P_ct)
        f_P_cta,f_P_ctp,f_P_ctn = fusion_P_ct[0:8],fusion_P_ct[8:16],fusion_P_ct[16:24]
        fusion_M_ori = coffi(M_ori)
        f_M_oria,f_M_orip,f_M_orin = fusion_P_ori[0:8],fusion_P_ori[8:16],fusion_P_ori[16:24]
#         mul_at,mul_pt,mul_nt = m_P[0:8],m_P[8:16],m_P[16:24]
#         pan_at,pan_pt,pan_nt = p_M[0:8],p_M[8:16],p_M[16:24]
        m_pred = D_CM(mul_ss)
        p_pred = D_CP(pan_ss)

        loss_cross_trip = triplet_loss(mul_as,pan_ps,pan_ns) +\
                                triplet_loss(pan_as,mul_ps,mul_ns) 
        loss_cross_fc = loss_fc(mul_sfc, label) + loss_fc(pan_sfc,label)
        L_s =  loss_cross_fc+loss_cross_trip
        
        loss_cross_fc1 = loss_fc(m_pred, label) + loss_fc(p_pred,label)
        loss_p_trip = triplet_loss(mul_a_pfea,pan_p_pfea,pan_n_pfea)+\
                            triplet_loss(pan_a_pfea,mul_p_pfea,mul_n_pfea)
        loss_p_fc = loss_fc(mul_fc,label) + loss_fc(pan_fc, label)
        Lp = loss_p_trip + loss_p_fc
        
        loss_m_p_trip = triplet_loss(M_cta,P_orip,P_orin) + triplet_loss(P_oria,M_ctp,M_ctn) +\
                        triplet_loss(P_cta,M_orip,M_orin) + triplet_loss(M_oria,P_ctp,P_ctn) 
        loss_fusion_trip = triplet_loss(f_M_cta,f_P_orip,f_P_orin) + triplet_loss(f_P_oria,f_M_ctp,f_M_ctn) +\
                        triplet_loss(f_P_cta,f_M_orip,f_M_orin) + triplet_loss(f_M_oria,f_P_ctp,f_P_ctn)       
                       
#         loss_m_p_trip = triplet_loss(mul_a_pfea,pan_pt,pan_nt) +\
#                         triplet_loss(pan_a_pfea,mul_pt,mul_nt)
        loss_convert_fc = loss_fc(m_P_fc,label) + loss_fc(p_M_fc,label)
        
        temp = 1
        h_k = F.kl_div(F.log_softmax(mul_sfc/temp, dim=1),F.softmax(pan_sfc/temp, dim=1), reduction='batchmean')+\
               F.kl_div(F.log_softmax(pan_sfc/temp, dim=1),F.softmax(mul_sfc/temp, dim=1), reduction='batchmean')
        h_k_sp =  F.kl_div(F.log_softmax(m_pred/temp, dim=1),F.softmax(p_pred/temp, dim=1), reduction='batchmean')+\
                   F.kl_div(F.log_softmax(p_pred/temp, dim=1),F.softmax(m_pred/temp, dim=1), reduction='batchmean') 
        h_K_all = h_k + h_k_sp
        
        loss =0.5* Lp +L_s + loss_cross_fc1  + h_K_all + loss_m_p_trip + loss_convert_fc + loss_fusion_trip
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        if (step+101) % 100 == 0:
            modelM.eval()
            modelP.eval()
            modelS.eval()

            val_acc = train_ap_map(modelM,modelP,modelS,W_m,W_m.inverse(),coffi,test_loader,625)#,modelW1,modelW2,modelG2,modelT, modelS,
            if (val_acc.map>map_best_acc):
                map_best_acc = val_acc.map
#                 ap_best_acc = val_acc.ap
                state = {'modelM':modelM.state_dict(),
                         'modelP':modelP.state_dict(),
                         'modelS':modelS.state_dict(),
                         'Dm':Dm.state_dict(),
                         'Dp':Dp.state_dict(),
                         'D_CP':D_CP.state_dict(),
                         'D_CM':D_CM.state_dict(),
                         'coffi':coffi.state_dict()
                        }
                torch.save(state, '/home/tangxu/SDP/0/models/pretrained_model_400.mdl')
                torch.save(W_m, '/home/tangxu/SDP/0/models/pretrained_W_m_400.pkl')
                torch.save(W_m.inverse(), '/home/tangxu/SDP/0/models/pretrained_W_p_400.pkl')


                print('best acc:',val_acc.map)
                """test operation
                """

        if (step % 10 == 0):
            print("loss = %f " % loss)
#                 print("domain_loss = %f " % domain_loss)

            losses.append(loss.cpu().detach().numpy())
                
