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

args.out = pathjoin(c.tmpdir,'val/b4_e50')
prtxt_out = pathjoin(c.tmpdir,'pr/b4_e50')

# =============================================================================
# config BEGIN
# =============================================================================
args.update(
        restore=-1,
        step=None,
        
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
    for name in c.names[:]:
#        img,gt = readimg(name),readgt(name)>0
        img = readimg(name)>0
        prob = predict(toimg(name))
        re = prob.argmax(2)
#        res= re*1.0
#        e.evalu(re,gt,name)
#        show(img,gt,re)
        imsave(pathjoin(u'/home/victoria/0-images/眼底照数据集和标签/new_res',name+'.png'),uint8(re))
#        imsave(pathjoin(args.out,name+'.png'),uint8(re))

#        for x in range(0,prob.shape[1]):
#            for y in range(0,prob.shape[0]):
##                    M = 0.02
#                if (prob[y][x][1] >= prob[y][x][0]):
#                    res[y][x] = prob[y][x][1]
#                else:
#                    res[y][x] = prob[y][x][1]
#
#        imsave(pathjoin(args.out,name+'res.png'),res)

#        for a in range(1,101,1):
#            f = open(prtxt_out+'/Pr-M'+str(a)+'.txt','r+')
#            f.read()
#            M = 0.01*a
#            TP =0
#            TN =0
#            FN =0
#            FP =0
#            TOL =0
#            for x in range(0,prob.shape[1]):
#                for y in range(0,prob.shape[0]):
#                    if (prob[y][x][1] >= M):
#                        if gt[y][x] == False:
#                            FP = FP + 1
#                        else:
#                            TP = TP + 1
#                    else:
#                        if gt[y][x] == False:
#                            TN = TN + 1
#                        else:
#                            FN = FN + 1
#                    TOL = TOL + 1
#            print(TP,FP,TN,FN,TOL)
#            f.write("%.4f"%(TP*1.0/(TP+FP+1)))
#            f.write("\t")
#            f.write("%.4f"%(TP*1.0/(TP+FN+1)))
#            f.write("\t")
#            f.write("%.4f"%M)
#            f.write("\n")
#        f.close
#    print args.restore,e.loss.mean()
#
#        show(res)
#        diff = binaryDiff(re,gt)
#        show(img,diff,re)
#        show(img,diff)
#        show(diff)
#        yellowImg=gt[...,None]*img+(npa-[255,255,0]).astype(np.uint8)*~gt[...,None]
#        show(yellowImg,diff)
    
#    print ww,hh


#map(lambda n:show(readimg(n),e[n],readgt(n)),e.low(80).index[:])















