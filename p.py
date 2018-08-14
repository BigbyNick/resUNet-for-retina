from lib import *
from config import *

setMod('train')

tifs = glob(cf.trainGlob)

tif = tifs[0]

img = imread(tif)

gifs = glob(u'/media/victoria/\u5de5\u4f5c/0-images/\u773c\u5e95\u7167\u6570\u636e\u96c6\u548c\u6807\u7b7e/DRIVE/*.gif')

gif = gifs[0]
for gif in gifs:
    gt = imread(gif)
    print gt.ndim

