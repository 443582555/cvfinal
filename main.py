
# coding: utf-8

# In[2]:


import json
import numpy as np
import torch

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from read_label import label_read


# In[50]:


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
            self.labels = self.labels+label_read('annotation/*.txt', video, videoLength).tolist()
        print(len(self.labels))
        print(len(self.videos))
        

    def __len__(self):
        return len(self.videos)
            

    def __getitem__(self, idx):
        feature = self.videos[idx]
        label = self.labels[idx]
        sample = {'feature': feature, 'mLabel': label}
        return sample


# In[51]:


c3ddataset = ThumosC3dDataset('training.json')


# In[52]:


dataloader = DataLoader(c3ddataset, batch_size=320)


# In[114]:


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

    def __init__(self,use_gpu, batch_size,sequenceLen,embedding_dim=2048,hidden_dim=200):
        super(BiLSTMSentiment_V2, self).__init__()
        self.sequenceLen = sequenceLen
        self.use_gpu = use_gpu
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.fc = nn.Linear(400,1)
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

    def forward(self, feature,hidden):
        self.update_hidden(hidden)
        x = feature.view(self.sequenceLen,self.batch_size,2048)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = lstm_out
        y_out = self.fc(y)
        y =F.tanh(y_out.view(1,-1).squeeze(0))
        
        return y


# In[115]:


EPOCHS=100
use_gpu=True
batch_size=10
sequenceLen = 32
mModal = BiLSTMSentiment_V2(use_gpu,batch_size,sequenceLen).cuda()
hidden = mModal.init_hidden()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(mModal.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# In[124]:


all_lossses=[]
total_y = []
total_yhat=[]
for epoch in range(EPOCHS):
    index=0
    g_loss=0
    mModal.train()
    for example in dataloader:
        optimizer.zero_grad()
        x = Variable(example['feature'].float().cuda())
        y = Variable(example['mLabel'].float().cuda())
        total_y = total_y+list(y.cpu().detach().numpy())
        if not y.shape[0] == 320:
            continue
        hidden = mModal.init_hidden()
        y_hat=mModal(x ,hidden)
        total_yhat = total_yhat+list(y_hat.cpu().detach().numpy())
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
    #|eval(valid_iter,mModal)


# In[123]:


print(y.shape)
print(y_hat.shape)
import matplotlib.pyplot as plt
plt.plot(y.cpu().numpy(),'r')
plt.plot(y_hat.cpu().detach().numpy(),'y')
plt.show()

