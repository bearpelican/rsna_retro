import sys; sys.path.insert(0, 'gen-efficientnet-pytorch/')

import torch
torch.backends.cudnn.benchmark = True
# import res2fg
from fastai2.basics           import *
from fastai2.vision.all       import *
from fastai2.medical.imaging  import *
from fastai2.callback.tracker import *
from fastai2.callback.all     import *
from fastai2.vision.learner   import _resnet_split
# from fastai2.vision.models    import xsenet

from scipy.optimize import minimize_scalar

set_num_threads(1)
pd.options.display.max_rows = 999

def xse_model(nm, mod, last_l=10, **kwargs):
    t = torch.load(f'xrn_stem.pth', map_location='cpu')
    #t = torch.load(f'pretrained/80_{nm}.pth', map_location='cpu')
    #del(t[f'{last_l}.weight'], t[f'{last_l}.bias'])
    m = getattr(mod,nm)(**kwargs)
    m.load_state_dict(t, strict=False)
    return m

# def res2net18(**kwargs): return xse_model('res2net18', res2fg, 'fc', c_out=6)
def xresnet18(**kwargs): return xse_model('xresnet18', xresnet)
def xresnet34(**kwargs): return xse_model('xresnet34', xresnet)
def xresnet18_deep      (**kwargs): return xse_model('xresnet18_deep'     , xresnet, 12)
def xresnet34_deep      (**kwargs): return xse_model('xresnet34_deep'     , xresnet, 12)
def xresnet50_deep      (**kwargs): return xse_model('xresnet50_deep'     , xresnet, 12)
def xresnet18_deeper    (**kwargs): return xse_model('xresnet18_deeper'   , xresnet, 14)
def xresnet34_deeper    (**kwargs): return xse_model('xresnet34_deeper'   , xresnet, 14)
def xresnet50_deeper    (**kwargs): return xse_model('xresnet50_deeper'   , xresnet, 14)
# def xse_resnext18_32x4d (**kwargs): return xse_model('xse_resnext18_32x4d', xsenet)
# def xse_resnext34_32x4d (**kwargs): return xse_model('xse_resnext34_32x4d', xsenet)
# def xse_resnext50_32x4d (**kwargs): return xse_model('xse_resnext50_32x4d', xsenet)
# def xse_resnext18_deep  (**kwargs): return xse_model('xse_resnext18_deep' , xsenet, 12)
# def xse_resnext34_deep  (**kwargs): return xse_model('xse_resnext34_deep' , xsenet, 12)
# def xse_resnext50_deep  (**kwargs): return xse_model('xse_resnext50_deep' , xsenet, 12)
# def xse_resnext18_deeper(**kwargs): return xse_model('xse_resnext18_deeper',xsenet, 14)
# def xse_resnext34_deeper(**kwargs): return xse_model('xse_resnext34_deeper',xsenet, 14)
# def xse_resnext50_deeper(**kwargs): return xse_model('xse_resnext50_deeper',xsenet, 14)

# model_meta[res2net18] = model_meta[resnet152]
for o in (
    xresnet18, xresnet18_deep, xresnet18_deeper, #xse_resnext18_32x4d, xse_resnext18_deep, xse_resnext18_deeper,
    xresnet34, xresnet34_deep, xresnet34_deeper, #xse_resnext34_32x4d, xse_resnext34_deep, xse_resnext34_deeper,
    xresnet50_deep, xresnet50_deeper, #xse_resnext50_32x4d, xse_resnext50_deep, xse_resnext50_deeper
): model_meta[o] = model_meta[xresnet152]

def se_resnext50_32x4d(pretrained=True): return pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet')
model_meta[se_resnext50_32x4d] = {'cut':-2, 'split':default_split}

# import geffnet
# def efficientnet_b0(pretrained=True): return geffnet.efficientnet_b0(as_sequential=True, pretrained=pretrained)
# def efficientnet_b1(pretrained=True): return geffnet.efficientnet_b1(as_sequential=True, pretrained=pretrained)
# def efficientnet_b2(pretrained=True): return geffnet.efficientnet_b2(as_sequential=True, pretrained=pretrained)
# def efficientnet_b3(pretrained=True): return geffnet.efficientnet_b3(as_sequential=True, pretrained=pretrained)
# def mixnet_s (pretrained=True): return geffnet.mixnet_s(as_sequential=True, pretrained=pretrained)
# def mixnet_m (pretrained=True): return geffnet.mixnet_m(as_sequential=True, pretrained=pretrained)
# def mixnet_l (pretrained=True): return geffnet.mixnet_l(as_sequential=True, pretrained=pretrained)
# def mixnet_xl(pretrained=True): return geffnet.mixnet_l(as_sequential=True, pretrained=pretrained)
# for o in (efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3,mixnet_s,mixnet_m,mixnet_l,mixnet_xl):
#     model_meta[o] = {'cut':-4, 'split':default_split}

np.set_printoptions(linewidth=120)
matplotlib.rcParams['image.cmap'] = 'bone'
set_num_threads(1)

path = Path('~/data/rsna').expanduser()
path_trn = path/'stage_1_train_images'
path_tst = path/'tst_jpg'

path_meta = path/'meta'
path_jpg512 = path/'nocrop_jpg'
path_jpg256 = path/'nocrop_jpg256'

path_pred = path/'preds'
path_pred_test = path/'preds_test'
path_slice = path/'slices'
path_slbls = path/'slice_lbls'
path_cts = path/'cts'
path_ct_lbls = path/'ct_lbls'

df_comb = pd.read_feather(path_meta/'both_df_comb.fth').set_index('SOPInstanceUID')
df_tst  = pd.read_feather(path_meta/'df_tst.fth').set_index('SOPInstanceUID')
# bins = (path_meta/'bins.pkl').load()
# brain_wins = [dicom_windows.brain,dicom_windows.subdural]
# soft_wins = brain_wins + [dicom_windows.abdomen_soft]
# brain_args = dict(wins=brain_wins, bins=bins)
# soft_args = dict(wins=soft_wins, bins=0)
htypes = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']
avg_lbls = tensor(df_comb[htypes].mean())
final_bias = to_device(logit(avg_lbls))

fns = L(list(df_comb.index))
set_seed(42)
patients = df_comb.PatientID.unique()
np.random.shuffle(patients)
patient_grps = Path('patient_grps.pkl').load()
val_sops = Path('val_sops.pkl').load()
idx = L.range(df_comb)
mask = df_comb.index.isin(set(val_sops))
splits = idx[~mask],idx[mask]

def patient_cv(idx): return np.concatenate([patient_grps[o] for o in range_of(patient_grps) if o!=idx])
def fn2label(fn): return df_comb.loc[fn][htypes].values.astype(np.float32)

def get_pil_fn(p):
    def _f(fn): return PILCTScan.create(p/f'{fn}.jpg')
    return _f

@Transform
def remove_hist(x:TensorImage): return x[:,[0,1,2]]
remove_hist.order=1

@Transform
def remove_soft(x:TensorImage): return x[:,[0,1,3]]
remove_soft.order=1

def get_wgts(df, splits):
    wgts = df['any'][splits[0]].values
    return wgts * (1/0.14 - 2) + 1

def get_data_gen(fns, bs, img_tfm, mean, std, splits, sz=None, nw=8,
        wgts=None, batch_xtra=None, after_item=None, with_aug=True, **kwargs):
    tfms = [[img_tfm], [fn2label,EncodedMultiCategorize(htypes)]]
    dsrc = DataSource(fns, tfms, splits=splits)
    nrm = Normalize(tensor(mean),tensor(std))
    batch_tfms = L(nrm, Cuda()) + L(batch_xtra)
    if with_aug: batch_tfms += aug_transforms(**kwargs)
    if sz is not None:
        batch_tfms = batch_tfms+[RandomResizedCropGPU(sz, min_scale=0.7, ratio=(1.,1.), valid_scale=0.9)]
    if wgts is None:
        return dsrc.databunch(bs=bs, num_workers=nw, after_item=after_item, after_batch=batch_tfms)
    else:
        return dsrc.weighted_databunch(wgts, bs=bs, num_workers=nw, after_item=after_item, after_batch=batch_tfms)

def get_data_pil(fns, bs, splits, sz=None, use_hist=True, nw=8, path=None,
                 wgts=None, mean=None, std=None, with_aug=True, batch_xtra=None):
    remover = [] if path is not None else [[remove_hist,remove_soft][use_hist]]
    if path is None: path = [path_jpg256,path_jpg512][sz is None or sz>256]
    if mean is None: mean = [0.2,0.2,0.4] if use_hist else [0.2]
    if std  is None: std  = [0.3]
    batch_xtra = L(batch_xtra) + [IntToFloatTensor()] + remover
    return get_data_gen(fns, bs, get_pil_fn(path), mean, std, splits=splits, sz=sz, wgts=wgts, with_aug=with_aug,
                        batch_xtra=batch_xtra, after_item=[ToTensor], nw=nw, max_rotate=30.)

def accuracy_any(inp, targ, thresh=0.5, sigmoid=True):
    inp,targ = flatten_check(inp[:,0],targ[:,0])
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()

loss_weights = to_device(tensor(2.0, 1, 1, 1, 1, 1))
loss_weights = loss_weights/loss_weights.sum()*6

def get_loss(scale=None):
    if scale is not None: scale = to_device(tensor([scale]*6))
    return BaseLoss(nn.BCEWithLogitsLoss, weight=loss_weights, #pos_weight=scale,
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

def get_learner(dbch, arch, lf, pretrained=True, opt_func=None, metrics=None, fp16=True, config=None):
    if metrics is None: metrics=[accuracy_multi,accuracy_any,lf2,opt_val_met]
    if opt_func is None: opt_func = partial(Adam, wd=0.0, eps=1e-8, sqr_mom=0.999)
    if config is None: config=dict(ps=0., lin_ftrs=[], concat_pool=False)
    learn = cnn_learner(dbch, arch, pretrained=pretrained, loss_func=lf, lr=3e-3,
                        opt_func=opt_func, metrics=metrics, config=config)
    return learn.to_fp16() if fp16 else learn

def do_fit(learn, epochs, lr, freeze=True, do_slice=False, **kwargs):
    if do_slice: lr = slice(lr*3,lr)
    if freeze:
        learn.freeze()
        learn.fit_one_cycle(1, lr, div=2, div_final=1, pct_start=0.1)
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

def dcm_tfm(fn):
    fn = (path_trn/fn).with_suffix('.dcm')
    try: x = fix_pxrepr(fn.dcmread())
    except Exception as e: print(fn,e); raise SkipItemException
    if x.Rows != 512 or x.Columns != 512: x.zoom_to((512,512))
    return TensorCTScan(x.scaled_px)

# def get_data_dcm(fns, bs, splits, sz=None, win_args=brain_args, nw=8):
def get_data_dcm(fns, bs, splits, sz=None, nw=8):
    return get_data_gen(fns, bs, dcm_tfm, [0.2], [0.3], splits=splits, sz=sz, nw=nw, max_rotate=30.)

def submission(df_tst, preds, fn='submission'):
    ids,labels = [],[]
    for idx,pred in zip(df_tst.index, preds):
        for i,label in enumerate(htypes):
            ids.append(f"{idx}_{label}")
            labels.append('{0:1.10f}'.format(pred[i].item()))
    df_csv = pd.DataFrame({'ID': ids, 'Label': labels})
    df_csv.to_csv(f'{fn}.csv', index=False)
    return df_csv

def pre_stem():
    m = nn.Sequential(
            ConvLayer(3, 16, 1, act_cls=nn.ReLU6, bn_1st=False),
            ConvLayer(16, 5, 1, act_cls=nn.Tanh,  bn_1st=False),
    )
    load_model('pre_stem.pth', m, None)
    return m.cuda()

def add_pos(df):
    df = df.sort_values(['SeriesInstanceUID','ImagePositionPatient2'])
    sopid = pd.Series(df.index, index=df.index)
    series_id = df.SeriesInstanceUID

    next_series = series_id.shift(-1)
    right = sopid.shift(-1).fillna('')
    right_dist = df.ImagePositionPatient2.diff().shift(-1)
    right.loc[series_id != next_series] = ''
    right_dist.loc[series_id != next_series] = 0.0

    prev_series = series_id.shift(1)
    left = sopid.shift(1).fillna('')
    left_dist = right_dist.shift(1).fillna(0.0)
    left.loc[series_id != prev_series] = ''
    left_dist.loc[series_id != prev_series] = 0.0

    return df.assign(Right=right,Left=left,LeftDist=left_dist,RightDist=right_dist).reset_index()

@Transform
def view_5c(x:TensorImage):
    bs,_,_,w = x.shape
    return x.view(bs,5,w,w)

def convmodel_5c(m):
    l0 = m[0][0]
    wf = l0.weight
    sb,sc,sh,sw = wf.shape
    wt = torch.zeros(sb,5,sh,sw, dtype=torch.float)
    wt[:,:3] = wf
    wt[:,1] /= 3
    wt[:,3] = wt[:,1]
    wt[:,4] = wt[:,1]
    wf.data = wt
    l0.in_channels = 5

moms=(0.9,0.9,0.9)
no_1cycle = dict(div=1, div_final=1, pct_start=0.1, moms=moms)

def get_rsna_data_func(n_gpus, use_hist=False, nw=8, batch_xtra=None):
    wgts = get_wgts(df_comb, splits)
    def get_data(bs, sz, use_wgt=True, batch_xtra=None):
        return get_data_pil(fns, bs, splits=splits, sz=sz, use_hist=use_hist,
                nw=nw, batch_xtra=batch_xtra, wgts = wgts if use_wgt else None)
    return get_data

def get_data_5c(fns, bs, splits, sz=None, use_hist=True, nw=8, path=None,
                 wgts=None, mean=None, std=None, batch_xtra=None):
    if path is None: path = path_5c
    if mean is None: mean = [0.1690, 0.1317, 0.4073, 0.4129, 0.4140]
    if std  is None: std  = [0.3116, 0.2718, 0.2941, 0.2942, 0.2936]
    batch_xtra = L(batch_xtra) + [IntToFloatTensor(), view_5c]
    return get_data_gen(fns, bs, get_pil_fn(path), mean, std, splits=splits, sz=sz, wgts=wgts,
                        batch_xtra=batch_xtra,
                        after_item=[ToTensor], nw=nw, max_rotate=30.)

def get_default_learner(gpu, n_gpus, get_data, arch, set_avg_lbls=False, **kwargs):
    learn = get_learner(get_data(128,128), arch, get_loss(), **kwargs)
    if set_avg_lbls: learn.model[1][-1].bias.data = to_device(logit(avg_lbls))
    if gpu is None: learn.to_parallel()
    if n_gpus>1: learn = learn.to_distributed(gpu)
    return learn

def appianish_cycle_train(learn, get_data, no_pretrain=False, schedule=None, lr=3e-4):
    do_slice=freeze= not no_pretrain
    if not schedule: schedule = [(64,192), (64,352), (64,512), (64,None)]

    """
    learn.loss_func = get_loss()
    bs,sz = schedule[3]
    learn.dbunch = get_data(bs, None, use_wgt=False)
    do_fit(learn, 1, lr*2/3*2/3, freeze=False, **no_1cycle)
    """

    if no_pretrain:
        learn.loss_func = get_loss()
        learn.dbunch = get_data(64, 160, use_wgt=False)
        do_fit(learn, 4, lr, freeze=False, **no_1cycle)

    #if not schedule: schedule = [(64,192), (64,352), (64,512), (64,None)]
    #if not schedule: schedule = [(16,192),(16,352),(16,512),(16,None)]
    learn.loss_func = get_loss()
    bs,sz = schedule[0]
    learn.dbunch = get_data(bs, 192, use_wgt=True)
    do_fit(learn, 2, lr, do_slice=do_slice, freeze=freeze , **no_1cycle)
    bs,sz = schedule[1]
    learn.dbunch = get_data(bs, 352, use_wgt=True)
    do_fit(learn, 2, lr*2/3, do_slice=do_slice, freeze=freeze, **no_1cycle)
    bs,sz = schedule[2]
    learn.dbunch = get_data(bs, 512, use_wgt=True)
    do_fit(learn, 1, lr*2/3*2/3, do_slice=False, freeze=False, **no_1cycle)

    learn.loss_func = get_loss()
    bs,sz = schedule[3]
    learn.dbunch = get_data(bs, None, use_wgt=False)
    do_fit(learn, 1, lr*2/3*2/3, freeze=False, **no_1cycle)
    do_fit(learn, 1, lr*2/3*2/3, freeze=False, div=1, pct_start=0.01)

def train_save( pre, gpu, n_gpus, nw, magic_seed, get_data_func, get_learner_func, train_func,
        arch=None, train=False, pred=False, is_master=False, **kwargs):
    set_seed(magic_seed)
    get_data = get_data_func(n_gpus, nw=nw)
    learn = get_learner_func(gpu, n_gpus, get_data, arch=arch)
    train_func(learn, get_data, **kwargs)
    if is_master: learn.save(pre)

def predict( learn, pre, get_data, is_master=False):
    if is_master: print(pre)
    learn.dbunch = get_data(128, None, use_wgt=None)
    pred_penul(learn, pre, get_data, is_master)

def save_preds(learn, ds_idx, pre):
    dss = ('trn','val')[ds_idx]
    print(f'{pre}-{dss}')
    save_preds = path/'penul'/f'{pre}-{dss}'
    save_targs = path/'penul'/f'{pre}-{dss}-targs'
    save_preds.mkdir(exist_ok=True)
    save_targs.mkdir(exist_ok=True)
    learn.get_preds(ds_idx=ds_idx, save_preds=save_preds, save_targs=save_targs, act=noop)

def dummy_loss(x,*args,**kwargs): return x.sum()

def pred_penul(learn, pre, get_data, is_master):
    learn.load(pre)
    learn.model = learn.model[0]
    learn.metrics = []
    learn.loss_func = dummy_loss
    save_preds(learn, 0, pre)
    save_preds(learn, 1, pre)

def old_predict(pre, gpu, n_gpus, nw, magic_seed, get_data_func, get_learner_func, train_func,
        arch=None, train=False, pred=False, is_master=False, **kwargs):
    get_data = get_data_func(n_gpus, nw=nw)
    learn = get_learner_func(gpu, n_gpus, get_data, arch=arch)
    learn.dbunch = get_data(128, None, use_wgt=None)
    pred_penul(learn, pre, get_data, is_master)

