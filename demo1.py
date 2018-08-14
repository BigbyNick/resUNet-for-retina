# encoding: utf-8
import cv2
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

num = 1
gt_path = '/home/victoria/0-images/眼底照数据集和标签/DRIVE/0'+str(num)+'_manual1.gif'
input_path = '/home/victoria/deep517/projects/retina/res-unet/val/3png/0'+str(num)+'res.png'

# gt = Image.open(gt_path)
im=cv2.imread(input_path,cv2.IMREAD_COLOR)
gt=cv2.imread(gt_path,cv2.IMREAD_COLOR)
img=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(cimg)
# plt.show()

print(im.shape)
print(cimg.shape)
print(cimg)
print(gt)



M=0.0
for a in range(1,101,1):
	# if(name > 1):
	# f = open('pr-'+str(a)+'.txt','r+')
	# f.read()
	# else:
 	# f = open('my/pr-'+str(a)+'.txt','w')
	# f.write("M\tPre\tRecall\n")
	M = a*0.01
	FP = 0	#FP
	TP = 0	#TP
	TN = 0	#TN
	FN = 0	#FN
	for x in range(cimg.shape[0]):
		for y in range(cimg.shape[1]):
			if ( cimg[x][y] >= M*255 ):
				if (gt[x][y] == 255):
					TP = TP+1
				else:
					FP = FP+1
			else:
				if (gt[x][y] == 0):
					TN = TN+1
				else:
					FN = FN+1
	pre = TP*1.0/(TP+FP)
	recall = TP*1.0/(TP+FN)
	# f.write("%.4f\t"%M)
	# f.write("%.4f\t"%pre)
	# f.write("%.4f\n"%recall)
	print(M,pre,recall)
# f.close
