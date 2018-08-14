#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 20:44:24 2017

@author: Yue
"""

# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sys,os
import numpy as np
import lib
from lib import dicto, glob, getArgvDic, findints,pathjoin
from lib import show, loga, logl, imread, imsave
from lib import Evalu,diceEvalu
from lib import *
from configManager import (getImgGtNames, indexOf, readgt, readimg, 
                           setMod, togt, toimg, makeValEnv, doc)
from train import c, cf, args
setMod('val')

args.out = pathjoin(c.tmpdir,'val/png')
pr_out = pathjoin(c.tmpdir,'pr')
# =============================================================================
# config BEGIN
# =============================================================================
args.update(
        restore=-1,
        step=64,
        
        )
# =============================================================================
# config END
# =============================================================================

if args.restore == -1:
    pas = [p[len(args.prefix):] for p in glob(args.prefix+'*')]
    args.restore = len(pas) and max(map(lambda s:len(findints(s)) and findints(s)[-1],pas))

makeValEnv(args)
#setMod('train')
if __name__ == '__main__':
    import predictInterface 
    c.predictInterface = predictInterface
    predict = predictInterface.predict 
#    c.predict = predict
#    e = Evalu(diceEvalu,
##              evaluName='restore-%s'%restore,
#              valNames=c.names,
##              loadcsv=1,
#              logFormat='dice:{dice:.3f}, loss:{loss:.3f}',
#              sortkey='loss',
##              loged=False,
##              saveResoult=False,
#              )
    c.names.sort(key=lambda x:readgt(x).shape[0])
#    c.names[0] = '01'

#    f = open(pathjoin(args.out,'Pr-all1.txt'),'w')
#    f.read()
#    f.write('Pre\tRec\tAcc\tM\n')
    
    for name in c.names[:]:
        img,gt = readimg(name),readgt(name)>0
        prob = predict(toimg(name))
        re = prob.argmax(2)
#        res= re*1.0
#        e.evalu(re,gt,name)
        show(img,gt,re)
#        imsave(pathjoin(args.out,name+'.png'),uint8(re))

#        for x in range(0,prob.shape[1]):
#            for y in range(0,prob.shape[0]):
#                    res[y][x] = prob[y][x][1]

#        imsave(pathjoin(args.out,name+'res.png'),res)


#        f = open(pathjoin(args.out,'Pr-'+name+'.txt'),'w')
##        f.read()
#        f.write('Pre\tRec\tAcc\tM\n')
#        res1 = res
#        TP,TN,FP,FN,TOL = 0 ;
        for a in range(1,101,1):
            M = 0.01*a
            TP =0
            TN =0
            FN =0
            FP =0
            TOL =0
            f = open(pathjoin(pr_out,'Pr-M'+str(a)+'.txt'),'r+')
            f.read()
#            print(a,TP,TN,FP,FN)
#            print('________')
            for x in range(0,prob.shape[1]):
                for y in range(0,prob.shape[0]):
#                    M = 0.02
                    if (prob[y][x][1] >= M):
                        if gt[y][x][1] == True:
                            TP = TP + 1
                        else:
                            FP = FP + 1
                    else:
                        if gt[y][x][1] == True:
                            FN = FN + 1
                        else:
                            TN = TN + 1
                    TOL = TOL + 1
            print(M,TP*1.0/(TP+FP+1),TP*1.0/(TP+FN+1),TP*2.0/(2*TP+FP+FN))
            f.write("%.4f"%(TP*1.0/(TP+FP+1)))
            f.write("\t")
            f.write("%.4f"%(TP*1.0/(TP+FN+1)))
#            f.write("\t")
#            f.write("%.4f"%(TP*2.0/(2*TP+FP+FN)))
            f.write("\n")
            f.close



#        diff = binaryDiff(re,gt)
#        show(img,diff,re)
#        show(img,diff)
#        show(diff)
#        yellowImg=gt[...,None]*img+(npa-[255,255,0]).astype(np.uint8)*~gt[...,None]
#        show(yellowImg,diff)
    
    print args.restore,e.loss.mean()


#map(lambda n:show(readimg(n),e[n],readgt(n)),e.low(80).index[:])















