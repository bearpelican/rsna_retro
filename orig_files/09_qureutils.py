import sys
sys.path.insert(0, 'gen-efficientnet-pytorch/')

from fastai2.basics           import *
from fastai2.vision.all       import *
from fastai2.medical.imaging  import *
from fastai2.vision.learner import _resnet_split
from fastai2.callback.tracker import *
from fastai2.callback.all import *

from scipy.optimize import minimize_scalar

import pretrainedmodels
def se_resnext50_32x4d(pretrained=True): return pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet')
model_meta[se_resnext50_32x4d] = {'cut':-2, 'split':default_split}

import geffnet
def efficientnet_b0(pretrained=True): return geffnet.efficientnet_b0(as_sequential=True, pretrained=pretrained)
def efficientnet_b1(pretrained=True): return geffnet.efficientnet_b1(as_sequential=True, pretrained=pretrained)
def efficientnet_b2(pretrained=True): return geffnet.efficientnet_b2(as_sequential=True, pretrained=pretrained)
def efficientnet_b3(pretrained=True): return geffnet.efficientnet_b3(as_sequential=True, pretrained=pretrained)
def mixnet_s (pretrained=True): return geffnet.mixnet_s(as_sequential=True, pretrained=pretrained)
def mixnet_m (pretrained=True): return geffnet.mixnet_m(as_sequential=True, pretrained=pretrained)
def mixnet_l (pretrained=True): return geffnet.mixnet_l(as_sequential=True, pretrained=pretrained)
def mixnet_xl(pretrained=True): return geffnet.mixnet_l(as_sequential=True, pretrained=pretrained)
for o in (efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3,mixnet_s,mixnet_m,mixnet_l,mixnet_xl):
    model_meta[o] = {'cut':-4, 'split':default_split}

np.set_printoptions(linewidth=120)
matplotlib.rcParams['image.cmap'] = 'bone'
set_num_threads(1)

brain_wins = [dicom_windows.brain,dicom_windows.subdural]
htypes = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']

def filename(o): return os.path.splitext(os.path.basename(o))[0]

set_seed(42)

def get_pil_fn(p):
    def _f(fn): return PILCTScan.create((p/fn).with_suffix('.jpg'))
    return _f

@Transform
def remove_hist(x:TensorImage): return x[:,[0,1,2]]
remove_hist.order=1

def get_wgts(df, splits):
    wgts = df['any'][splits[0]].values
    return wgts * (1/0.14 - 2) + 1

def accuracy_any(inp, targ, thresh=0.5, sigmoid=True):
    inp,targ = flatten_check(inp[:,0],targ[:,0])
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()

loss_weights = to_device(tensor(2.0, 1, 1, 1, 1, 1))
loss_weights = loss_weights/loss_weights.sum()*6

def get_loss(scale=1.0):
    pw = to_device(tensor([scale]*6))
    return BaseLoss(nn.BCEWithLogitsLoss, weight=loss_weights, pos_weight=pw,
                    floatify=True, flatten=False, is_2d=False, activation=torch.sigmoid)

lf2 = get_loss()

def opt_val_loss(preds, targs, full=False):
    preds,targs = to_device((preds,targs))
    def f(x): return lf2(preds*x,targs).cpu()
    res = minimize_scalar(f, bounds=(0.2,2), method='Bounded', options={'xatol':0.001})
    return res if full else res.fun

opt_func = partial(Adam, wd=0.0, eps=1e-8, sqr_mom=0.999)
opt_val_met = AccumMetric(opt_val_loss,flatten=False)
metrics=[accuracy_multi,accuracy_any,lf2,opt_val_met]

def get_learner(dbch, arch, lf, pretrained=True, opt_func=None, metrics=None, fp16=True):
    if metrics is None: metrics=[accuracy_multi,accuracy_any,lf2,opt_val_met]
    if opt_func is None: opt_func = partial(Adam, wd=0.0, eps=1e-8, sqr_mom=0.999)
    config=dict(ps=0., lin_ftrs=[], concat_pool=False)
    learn = cnn_learner(dbch, arch, pretrained=pretrained, loss_func=lf, lr=3e-3,
                        opt_func=opt_func, metrics=metrics, config=config)
    return learn.to_fp16() if fp16 else learn

def do_fit(learn, epochs, lr, freeze=True, pct=None, do_slice=False, **kwargs):
    if do_slice: lr = slice(lr*5,lr)
    if freeze:
        learn.freeze()
        cb = ShortEpochCallback(pct=pct, short_valid=False) if pct else None
        learn.fit_one_cycle(1, lr, cbs=cb, div=2, div_final=1, pct_start=0.3)
    learn.unfreeze()
    learn.fit_one_cycle(epochs, lr, **kwargs)

def fix_pxrepr(dcm):
    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100: return dcm
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000
    return dcm

@Transform
def view_5c(x:TensorImage):
    bs,_,_,w = x.shape
    return x.view(bs,5,w,w)

moms=(0.9,0.9,0.9)
no_1cycle = dict(div=1, div_final=1, pct_start=0, moms=moms)
