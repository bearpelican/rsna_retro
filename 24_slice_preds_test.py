from slice_e2e_test import *
from fastai2.test import *

pre,arch,depth,tta,fr = sys.argv[1:]
depth,tta,fr = int(depth),int(tta),int(fr)
print(pre)

learn = get_m(depth, arch, True, pre)
fn = f'{pre}-e2e'
if fr: fn += '-fr'
assert path_pred_test.exists()
optx = (path_pred/f'{fn}-opt.pkl').load()
print(optx)
learn.load(fn)
preds,targs = learn.get_preds(act=noop)
sc_preds = optx*preds
dest_fn = (path_pred_test/f'{fn}.pkl')
dest_fn.save(sc_preds.cpu())

dl = learn.dbunch.valid_dl
with dl.set_split_idx(0): preds,targs = learn.get_preds(act=noop, dl=dl)
sc_preds = optx*preds
dest_fn = (path_pred_test/f'{fn}-flip.pkl')
dest_fn.save(sc_preds.cpu())

