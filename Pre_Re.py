#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:38:08 2017
Precision-Recall曲线

"""
from __future__ import unicode_literals

import sys,os
import numpy as np
import pylab as pl

 

# Use numpy to load the data contained in the file

# ’fakedata.txt’ into a 2-D array called data

def average0(seq, total=0.0): 
  num = 0
  for item in seq: 
    total += item[0] 
    num += 1
  return total / num 

def average1(seq, total=0.0): 
  num = 0
  for item in seq: 
    total += item[1] 
    num += 1
  return total / num 

def F1(seq, recall=0.0, pre=0.0): 
  for item in seq: 
    pre = item[0]
    recall = item[1]
    F1 = 2.0/(1./pre+1./recall)
  return F1 


if __name__ == '__main__':
#    f = open(pathjoin('Pr-out.txt'),'r+')
#    f.read()
    pl.figure(figsize=(5,5))

    
    data = np.loadtxt('Pr-my1out.txt')
    data2 = np.loadtxt('Pr-b4_e50.txt')
    p_data = np.loadtxt('Pr-pout.txt')
    
#    ave_x = average(data)
#    ave_y = average1(data)
    
#    for i in range(1,101,1):
#        data = np.loadtxt('pr/Pr-M'+str(i)+'.txt')
#        ave_x = average(data)
#        ave_y = average1(data)
#        f.write("%.4f"%ave_x)
#        f.write("\t")
#        f.write("%.4f"%ave_y)
#        f.write("\n")
    F1 = 0.0
    F1_x = 0.0
    F1_y = 0.0
    for item in p_data:
        F = 2.0*item[2]*item[1]/(item[2]+item[1])
        if F > F1:
            F1 = F
            F1_x = item[1]
            F1_y = item[2]
    pl.plot(F1_x, F1_y, 'yo')    
    F11 = 0.0
    F11_x = 0.0
    F11_y = 0.0
    for item in data:
        F = 2.0*item[2]*item[1]/(item[2]+item[1])
        if F > F11:
            F11 = F
            F11_x = item[1]
            F11_y = item[2]
    pl.plot(F11_x, F11_y, 'yo')    

    F11 = 0.0
    F11_x = 0.0
    F11_y = 0.0
    for item in data2:
        F = 2.0*item[2]*item[1]/(item[2]+item[1])
        if F > F11:
            F11 = F
            F11_x = item[1]
            F11_y = item[2]
    pl.plot(F11_x, F11_y, 'yo')    
#    x = 1
#    pre = 0.0
#    re = 0.0
    
#    for x in range(1,1,1):
#        data = np.loadtxt('pr/Pr-M'+str(x)+'.txt')
#        print('pr/Pr-M'+str(x)+'.txt')
#        pl.plot(data[:,0], data[:,1], 'ro')
#        for i in range(0,20,1):
#            pre = pre + data[i,0]
#            re = re + data[i,1]
#    #    pre = pre / 20.
#    #    re = re / 20.
#    #    summ= summ + data[:,0]
#        print pre
#        print re
        
     
    
    # plot the first column as x, and second column as y    
    #pl.figure(figsize=(5,5))
    pl.plot(data[:,1], data[:,2], 'r')    
    pl.plot(p_data[:,1], p_data[:,2], 'b')    
    pl.plot(data[:,1], data[:,2], 'g')    
    pl.xlabel('Recall')    
    pl.ylabel('Precision')
    pl.xlim(0.3, 1.0)
    pl.ylim(0.3, 1.0)
    #pl.set_aspect(1)
    pl.savefig('pr-'+ str(F11) +'.png')
    pl.show()
#    f.close
