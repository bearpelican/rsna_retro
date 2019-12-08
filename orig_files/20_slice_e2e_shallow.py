from rsnautils import *
from fastai2.callback.data import *
from fastai2.patch_tables import patch_tables
from fastai2.test import *
patch_tables()

val_sid = df_comb.loc[val_sops].SeriesInstanceUID.unique()
sids = df_comb[df_comb.is_qure==False].SeriesInstanceUID.unique()
idx = L.range(sids)
val_sid = set(list(val_sid))
mask = L(o in val_sid for o in sids)
s_splits = L(idx[~mask],idx[mask])
#s_splits = L(s_splits[0][:10], s_splits[1][:10])

df = df_comb.reset_index().set_index('SeriesInstanceUID').sort_values("ImagePositionPatient2")

class ReadCT:
    def __init__(self,sz):
        self.path = [path_jpg256,path_jpg512][sz is None or sz>256]
        self.rfn = get_pil_fn(self.path)
        self.tt = ToTensor()

    def one(self, sop): return self.tt(self.rfn(sop))[:3]

    def x(self, sid):
        sids = df.SOPInstanceUID[sid].values
        xs = [self.one(sop) for sop in sids]
        return TensorCTScan(torch.stack(xs))

    def y(self, sid): return TensorMultiCategory(df.loc[sid,htypes].values).float()

rct = ReadCT(512)
dsrc = DataSource(sids, [[rct.x],[rct.y]], splits=s_splits)
#dsrc = DataSource(sids, [[rct.x],[rct.y]], splits=[s_splits[0][:1],s_splits[1]])
mean,std = 0.2,0.3
nrm = Normalize(tensor(mean),tensor(std))
tfm = Flip(p=1)
batch_tfms = [nrm, Cuda(), IntToFloatTensor(), tfm]
#batch_tfms += aug_transforms(max_rotate=30., batch=True)

# if sz is not None:
#     batch_tfms = batch_tfms+[RandomResizedCropGPU(sz, min_scale=0.7, ratio=(1.,1.), valid_scale=0.9)]

dbunch = DataBunch(
    TfmdDL(dsrc.train, bs=None, after_batch=batch_tfms, num_workers=8, shuffle=True),
    TfmdDL(dsrc.valid, bs=None, after_batch=batch_tfms, num_workers=8)
)
dbunch.device = default_device()
dbunch.c = 6

loss_func = get_loss()

class Batchify(Module):
    def forward(self, x): return x[None].transpose(1,2)

class DeBatchify(Module):
    def forward(self, x): return x[0].transpose(0,1)

class Batchify2(Module):
    def forward(self, x): return x[...,0,0][None].transpose(1,2)

class DeBatchify2(Module):
    def forward(self, x): return x[0].transpose(0,1)

def conv3(ni,nf,stride=1, norm_type=NormType.Batch):
    return ConvLayer(ni, nf, (5,3,3), stride=(1,stride,stride), ndim=3, padding=(2,1,1), norm_type=norm_type)

def conv1(ni,nf,stride=1, norm_type=NormType.Batch):
    return ConvLayer(ni, nf, 5, ndim=1, padding=2, norm_type=norm_type)

metrics=[accuracy_multi,accuracy_any,lf2,opt_val_met]
opt_func = partial(Adam, wd=0.0, eps=1e-8, sqr_mom=0.999)

def xresnet34_deep2(pretrained): return XResNet(1, [3,4,6,3,1,1,1,1])
model_meta[xresnet34_deep2] = model_meta[xresnet34]
def xresnet34_deepish2(pretrained): return XResNet(1, [3,4,6,3,1,1])
model_meta[xresnet34_deepish2] = model_meta[xresnet34]

def get_m(depth, arch, init_stem, pre):
    if depth==0:
        m = nn.Sequential(Batchify(),
            conv3(512,256,2), # 8
            conv3(256,128,2), # 4
            conv3(128, 64,2), # 2
            DeBatchify(), nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(64,6))
    elif depth==1:
        m = nn.Sequential(Batchify(),
            conv3(256,256,1, norm_type=None),
            conv3(256,128,1, norm_type=None),
            conv3(128, 64,2),
            DeBatchify(), nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(64,6))
    else:
        m = nn.Sequential(Batchify2(),
            conv1(256,256,1, norm_type=None), # 8
            conv1(256,128,1, norm_type=None), # 4
            conv1(128, 64,1, norm_type=None), # 2
            DeBatchify2(), Flatten(), nn.Linear(64,6))

    if init_stem: init_cnn(m)
    else:
        sd = torch.load(f'models/{pre}-3d.pth', map_location='cpu')['model']
        m.load_state_dict(sd)

    config=dict(custom_head=m, init=None)
    learn = cnn_learner(dbunch, globals()[arch], pretrained=False, loss_func=loss_func, lr=3e-3,
                        metrics=metrics, config=config).to_fp16()
    return learn

def do(gpu,pre,depth,arch,lr,freeze, tta):
    learn = get_m(depth, arch, freeze, pre)
    sd = torch.load(f'models/{pre}.pth', map_location='cpu')['model']
    learn.model.load_state_dict(sd, strict=False)
    if gpu is not None: learn.to_distributed(gpu)

    if True:
        learn.model[-1][-1].bias.data = to_device(logit(avg_lbls))
        learn.create_opt()
        print('freeze')
        learn.freeze()
        learn.fit_one_cycle(2, lr*10)
        learn.unfreeze()
        if not rank_distrib(): learn.save(f'{pre}-e2e-fr')

    learn.create_opt()
    learn.fit_one_cycle(2, slice(lr))
    if not rank_distrib(): learn.save(f'{pre}-e2e')

