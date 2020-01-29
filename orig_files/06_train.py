import sys,importlib
if len(sys.argv)>3: lr,epochs = sys.argv[3:]
else: lr,epochs = 0,0

module = importlib.import_module(sys.argv[1])
gpu = int(sys.argv[2])
import torch
torch.cuda.set_device(gpu)

data = module.get_data()
learn = module.get_learner(data)
res = module.train(learn, float(lr), int(epochs))

from rsnautils import *
from fastai2.callback.data import *

fns = L(list(df_comb.fname)).map(filename)
splits = split_data(df_comb,0)
wgts = get_wgts(df_comb, splits)

def get_data(bs, sz, use_wgt=True):
    return get_data_5c(fns, bs, splits=splits, sz=sz, use_hist=False, wgts = wgts if use_wgt else None)

dbch = get_data(32,128)
loss_func = get_loss()

learn = get_learner(dbch, resnet18, loss_func, pretrained=True)
learn.model[1][-1].bias.data = to_device(logit(avg_lbls))
convmodel_5c(learn.model)

learn.loss_func = get_loss(0.14*2)
learn.dls = get_data(128, 192, use_wgt=True)
do_fit(learn, 2, lr     * 2, freeze=False, **no_1cycle)
learn.dls = get_data(128, 352, use_wgt=True)
do_fit(learn, 2, lr*2/3 * 2, freeze=False, **no_1cycle)
learn.dls = get_data(64,  512, use_wgt=True)
do_fit(learn, 1, lr*2/3*2/3, freeze=False, **no_1cycle)

learn.loss_func = get_loss()
learn.dls = get_data(64, None, use_wgt=False)
do_fit(learn, 1, lr*2/3*2/3, freeze=False, **no_1cycle)
do_fit(learn, 1, lr*2/3*2/3, freeze=False, div=1, pct_start=0)

learn.save(pre)

