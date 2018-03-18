
# coding: utf-8

# In[66]:


import json
import numpy as np
import torch


# In[67]:


from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from read_label_two import label_read


# In[68]:


class ThumosC3dDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, C3d, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(C3d) as fh:
            a = json.load(fh)
        self.videos=[]
        self.videonames=[]
        self.labels=[]
        self.weights = []
        for i in range(len(a)):
            #current videoname
            video = a[i].get("video")
            self.videonames.append(video.split("/")[-1])
            clip=[]
            videoLength = 0
            for j in a[i].get("clips"):
                segment = j["features"]
                videoLength+=1
                self.videos.append(torch.from_numpy(np.asarray(segment)))
            #print(videoLength)
            #print(label_read('annotation/*.txt', video, videoLength).tolist())
            path_name = 'annotation/*.txt'
            #print (video)
            self.labels=self.labels+label_read(path_name, video, videoLength).tolist()
        print(set(self.labels))
        for item in range(21):
            occur = self.labels.count(item)
            occur = 1.0/occur
            self.weights = self.weights + [occur]
        self.weights = torch.FloatTensor(self.weights).cuda()
            
        
    def __len__(self):
        return len(self.videos)
            

    def __getitem__(self, idx):
        
        feature = self.videos[idx]
        label = self.labels[idx]
        sample = {'feature': feature, 'mLabel': label}
        return sample


# In[69]:


c3ddataset = ThumosC3dDataset('training.json')


# In[5]:


dataloader = DataLoader(c3ddataset, batch_size=320)


# In[70]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
def repackage_hidden(h):
    if type(h)==Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
class BiLSTMSentiment_V2(nn.Module):

    def __init__(self,use_gpu, batch_size,sequenceLen,label_dim,embedding_dim=2048,hidden_dim=1024):
        super(BiLSTMSentiment_V2, self).__init__()
        self.sequenceLen = sequenceLen
        self.use_gpu = use_gpu
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.fc1 = nn.Linear(2048,512)
        self.fc2 = nn.Linear(512,label_dim)
        self.hidden=[]
        
        
    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            self.hidden = (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            self.hidden=(Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))
        return self.hidden
    def update_hidden(self,h):
        self.hidden = h
        return self.hidden

    def forward(self, feature):
        self.hidden = repackage_hidden(self.hidden)
        x = feature.view(self.sequenceLen,self.batch_size,2048)
        
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        #print(lstm_out)
        y = lstm_out.transpose(0,1)
        y_out = self.fc2(self.fc1(y))
        y =F.sigmoid(y_out.view(self.batch_size*self.sequenceLen,-1))
        
        return y


# In[71]:


EPOCHS=10000
use_gpu=True
batch_size=10
sequenceLen = 32
label_dim=21

mModal = BiLSTMSentiment_V2(use_gpu,batch_size,sequenceLen,label_dim).cuda()
hidden = mModal.init_hidden()
loss_function = nn.CrossEntropyLoss(weight = c3ddataset.weights)
optimizer = torch.optim.Adam(mModal.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)


# In[72]:


import matplotlib.pyplot as plt
def eval(valid_iter,mModal):
    index=0
    pred = torch.FloatTensor()
    label = torch.ByteTensor()
    correct = 0
    total = 0
    mModal.eval()
    
    for examples in valid_iter:
        x = Variable(examples['feature'].float().cuda())
        y = Variable(examples['mLabel'].float().cuda())
       
        if not y.shape[0] == 320:
            continue
        #hidden = mModal.init_hidden()
        #outputs = mModal(x,hidden)
        outputs = mModal(x)
        
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += y.size(0)
        
        correct += (predicted == y.data.long()).sum()
        
    print('Accuracy of the network on the Validation Set: %d %%' % (100 * correct / total))   
    
    
#     print("===========>ROC_AUC score on valid set")
#     print(roc_auc_score(label, pred))


# In[73]:


all_lossses=[]
total_y = []
total_yhat=[]
index=0
for epoch in range(EPOCHS):
    index=0
    g_loss=0
    mModal.train()
    for example in dataloader:
        optimizer.zero_grad()
        x = Variable(example['feature'].float().cuda())
        y = Variable(example['mLabel'].long().cuda())
        #total_y = total_y+list(y.cpu().detach().numpy())
        if not y.shape[0] == 320:
            continue
#         if index==0:
#             hidden = mModal.init_hidden()
#             index=1
#         else:
#             hidden = mModal.update_hidden()
#         y_hat=mModal(x ,hidden)
        y_hat=mModal(x)
        #total_yhat = total_yhat+list(y_hat.cpu().detach().numpy())
        
        loss = loss_function(y_hat,y)
        
        loss.backward()
        optimizer.step()
        index+=1
        g_loss+=loss.data[0]
        if index%100==0:
            all_lossses.append(loss/100)
            print(g_loss/100)
            g_loss=0
    scheduler.step()
    eval(dataloader,mModal)

