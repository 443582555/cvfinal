# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:40:02 2018

@author: Jiaxiang
"""
import numpy as np
from glob import glob
from scipy import signal

def label_read(path,video_name,length):
    fps = 25
    sigma = 2
    window = signal.gaussian(3*sigma,sigma)
    files = glob(path)
    label_list= np.zeros(length)
    for file in files:
        try:
            with open(file) as f:
                for line in f:
                    #print(line.split()[1])
                    if line.split()[0] == video_name:
                        start = float(line.split()[1])
                        end = float(line.split()[2])
                        start_clip = np.floor(start*25/16).astype(int)
                        end_clip = np.floor(end*25/16).astype(int)
                        # (start)
                        label_list[start_clip] = 1
                        label_list[end_clip] = -1
                        label_list
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
    label_list = np.convolve(label_list,window,'same')
    return(label_list)


#def main:
if __name__=='__main__':
    label_list=label_read(1,2,10000)
    print(np.max(label_list))
