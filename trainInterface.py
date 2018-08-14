# coding: utf-8
'''
res-unet1-simg
取小图训练 

Parameters
----------
step : int
    填充黑边 将图片shape 调整为step的整数倍
'''
from yllab import *
from lib import *
import logging
logging.basicConfig(level=logging.INFO)
npm = lambda m:m.asnumpy()
npm = FunAddMagicMethod(npm)

import mxnet as mx
import random
from netdef import getNet

class SimpleBatch(object):
    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad

    
labrgb = lambda lab:cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
randint = lambda x:np.random.randint(-x,x)
def imgAug(image,gt,prob=.5):
    if random.random() > prob:
        image = np.fliplr(image)
        gt = np.fliplr(gt)
    if random.random() > prob:
        image = np.flipud(image)
        gt = np.flipud(gt)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.int32)
    # adjust brightness
    hsv[:, :, 2] = hsv[:, :, 2] + random.randint(-15, 15)
    # adjust saturation
    hsv[:, :, 1] = hsv[:, :, 1] + random.randint(-10, 10)
    # adjust hue
    hsv[:, :, 0] = hsv[:, :, 0] + random.randint(-5, 5)
    hsv = np.clip(hsv, 0, 255)
    hsv = hsv.astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image,gt

def handleImgGt(imgs, gts,):
    g.imgs = imgs[0].copy()
    for i in range(len(imgs)):
#        if np.random.randint(2):
#            imgs[i] = np.fliplr(imgs[i])
#            gts[i] = np.fliplr(gts[i])
#        if np.random.randint(2):
#            imgs[i] = np.flipud(imgs[i])
#            gts[i] = np.flipud(gts[i])
        imgs[i],gts[i] = imgAug(imgs[i],gts[i])
    g.im=imgs;g.gt =gts
    gts=gts>.5
    imgs = imgs.transpose(0,3,1,2)/255.
    mximgs = map(mx.nd.array,[imgs])
    mxgtss = map(mx.nd.array,[gts])
    mxdata = SimpleBatch(mximgs,mxgtss)
    return mxdata


class GenSimgInMxnet(GenSimg):
    @property
    def provide_data(self):
        return [('data', (c.batch, 3, c.simgShape[0], c.simgShape[1]))]
    @property
    def provide_label(self):
        return  [('softmax1_label', (c.batch, c.simgShape[0], c.simgShape[1])),]


def saveNow(name = None):
    f=mx.callback.do_checkpoint(name or args.prefix)
    f(-1,mod.symbol,*mod.get_params())
    
c = dicto(
 gpu = 1,
 lr = 0.01,
 epochSize = 10000,
 step=64,
 window=64*2,
 classn=3
 )
c.resize = 1

if __name__ == '__main__':
    from train import args
    
else:
    from configManager import args
c.update(args)
args = c

args.simgShape = args.window
if not isinstance(args.window,(tuple,list,np.ndarray)):
    args.simgShape = (args.window,args.window)

net = getNet(args.classn)

if args.resume:
    print('resume training from epoch {}'.format(args.resume))
    _, arg_params, aux_params = mx.model.load_checkpoint(
        args.prefix, args.resume)
else:
    arg_params = None
    aux_params = None

if 'plot' in args:
    mx.viz.plot_network(net, save_format='pdf', shape={
        'data': (1, 3, 640, 640),
        'softmax1_label': (1, 640, 640), }).render(args.prefix)
    exit(0)
mod = mx.mod.Module(
    symbol=net,
    context=[mx.gpu(k) for k in range(args.gpu)],
    data_names=('data',),
    label_names=('softmax1_label',)
)
c.mod = mod

#if 0:
args.names = args.names[:]
gen = GenSimgInMxnet(args.names, args.simgShape, 
                      handleImgGt=handleImgGt,
                      batch=args.batch,
                      cache=len(args.names),
                      iters=args.epochSize
                      )
#gen = GenSimgInMxnet(args.names,c.batch,handleImgGt=imgGtAdd0Fill(c.step))
g.gen = gen
total_steps = len(c.names) * args.epoch
lr_sch = mx.lr_scheduler.MultiFactorScheduler(
    step=[total_steps // 2, total_steps // 4 * 3], factor=0.1)
def train():
    mod.fit(
        gen,
        begin_epoch=args.resume,
        arg_params=arg_params,
        aux_params=aux_params,
        batch_end_callback=mx.callback.Speedometer(args.batch),
        epoch_end_callback=mx.callback.do_checkpoint(args.prefix),
        optimizer='sgd',
        optimizer_params=(('learning_rate', args.lr), ('momentum', 0.9),
                          ('lr_scheduler', lr_sch), ('wd', 0.0005)),
        num_epoch=args.epoch)
if __name__ == '__main__':
    pass


if 0:
    #%%
    ne = g.gen.next()
#for ne in dd:
    ds,las = ne.data, ne.label
    d,la = npm-ds[0],npm-las[0]
    im = d.transpose(0,2,3,1)
    show(labrgb(uint8(im[0])));show(la)
