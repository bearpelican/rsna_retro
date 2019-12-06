import torch
torch.cuda.set_device(2)

from rsnautils import *
from fastai2.callback.data import *
from fastai2.patch_tables import patch_tables
patch_tables()

nw = 8
path_penul = path/'pp'
path_slice = path/'slices'
path_slice.mkdir(exist_ok=True)
path_slbls = path/'slice_lbls'
path_slbls.mkdir(exist_ok=True)

trn_sops = fns[splits[0]]
trn_sid = df_comb.loc[trn_sops].SeriesInstanceUID.unique()
mask = df_comb.SeriesInstanceUID.isin(trn_sid)
df_val = df_comb[~mask]
df_trn = df_comb[mask]

def sort_name(fn): return int(fn.name)
def get_sz(fn): return fn.load_array().shape

def save_ct_slices(o, dest):
    fn,si,chunk = o
    arr = fn.load_array()
    for i,(it,sn) in enumerate(zip(arr,chunk)): (dest/sn).save_array(it)

def get_sizes(batches):
    shapes = parallel(get_sz, batches, n_workers=16)
    bss = shapes.itemgot(0)
    return [0] + L(list(np.cumsum(bss)))

def save_all_slices(ds_idx, istarg):
    dfidx = (df_trn,df_val)[ds_idx].reset_index()
    dss = ('trn','val')[ds_idx]
    s = f'{pre}-{dss}'
    if istarg: s += '-targs'
    src = path_penul/s
    batches = src.ls().sorted(key=sort_name)
    cumsz = get_sizes(batches)
    chunks = (dfidx[cumsz[o]:cumsz[o+1]].SOPInstanceUID for o in range(len(dfidx)-1))
    p = (path_slice,path_slbls)[istarg]
    dest = p/pre
    dest.mkdir(exist_ok=True)
    parallel(save_ct_slices, zip(batches,cumsz,chunks), n_workers=12, total=len(batches), dest=dest)

pre = sys.argv[1]
print(pre)
print('01')
save_all_slices(0, 1)
print('00')
save_all_slices(0, 0)
print('11')
save_all_slices(1, 1)
print('10')
save_all_slices(1, 0)

path_cts = path/'cts'
path_cts.mkdir(exist_ok=True)
path_ct_lbls = path/'ct_lbls'
path_ct_lbls.mkdir(exist_ok=True)

def save_cts(si, src, dest):
    sops = df_comb[df_comb.SeriesInstanceUID==si].sort_values('ImagePositionPatient2').index
    arr = np.stack([(src/sop).load_array() for sop in sops])
    (dest/si).save_array(arr)

print('a')
src = path_slbls/pre
dest = path_ct_lbls/pre
dest.mkdir(exist_ok=True)

parallel(save_cts, df_comb.SeriesInstanceUID.unique(), n_workers=16, src=src, dest=dest);

print('b')
src = path_slice/pre
dest = path_cts/pre
dest.mkdir(exist_ok=True)

parallel(save_cts, df_comb.SeriesInstanceUID.unique(), n_workers=16, src=src, dest=dest);

