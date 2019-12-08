from slice_e2e_shallow import *
from fastai2.test import *

pre,arch,depth,tta,fr = sys.argv[1:]
depth,tta,fr = int(depth),int(tta),int(fr)
print(pre)

learn = get_m(depth, arch, True, pre)
fn = f'{pre}-e2e'
if fr: fn += '-fr'
o=2.0
(path_pred/f'{fn}-opt.pkl').save(o)
learn.load(fn)
preds,targs = learn.get_preds(act=noop)
optv = opt_val_loss(preds, targs, full=True)
(path_pred/f'{fn}-opt.pkl').save(optv.x)
"""
preds,targs = to_device((preds,targs))
sc_preds = optv.x*preds
dest_fn = (path_pred/f'{fn}-flip.pkl')
dest_fn.save(sc_preds.cpu())
"""

