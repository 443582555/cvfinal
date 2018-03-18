# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:40:02 2018

@author: Jiaxiang
"""
import numpy as np
from glob import glob
from scipy import signal
import os

def label_read(path,video,length):
    fps = 24
    #sigma = 1/np.sqrt(2*np.pi)
    #path = 'annotation/*.txt'
    #video = '/media/yifu/NewVolume/validation/video_validation_0000481.mp4'
    video_name = os.path.basename(video)
    video_name = os.path.splitext(video_name)[0]
    #print(video_name)
    files = glob(path)
    #print(len(files))
    label_list= np.zeros(length)
    j = 0
    for file in files:
	#print(file)
        j = j+1
        try:
            with open(file) as f:
                for line in f:
                    #print(line.split()[1])
                    if line.split()[0] == video_name:
                        start = float(line.split()[1])
                        end = float(line.split()[2])
                        start_clip = np.floor(start*24/16).astype(int)
                        end_clip = np.floor(end*24/16).astype(int)
                        # (start)
                        label_list[start_clip:end_clip+1]=j
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
    #label_list = np.convolve(label_list,window,'same')
    #label_list = np.sqrt(2*np.pi)*np.array(sigma)
    #label_list = list(label_list)
    return(label_list)


#def main:
if __name__=='__main__':
    label_list=label_read('annotation/*.txt','/media/yifu/NewVolume/validation/video_validation_0000160.mp4',10000)
    print(np.max(label_list))
