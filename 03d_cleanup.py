from rsnautils import *

def dcm_tfm(fn):
    try:
        x = fn.dcmread()
        fix_pxrepr(x)
    except Exception as e:
        print(fn,e)
        raise SkipItemException
    if x.Rows != 512 or x.Columns != 512: x.zoom_to((512,512))
    return x.scaled_px

def save_file(o, dest, ext, tiff):
    fname,px = o
    fn = dest/Path(fname).with_suffix(f'{ext}')
    wins = (dicom_windows.brain, dicom_windows.subdural, dicom_windows.abdomen_soft)
    if tiff: px.save_tif16(fn, bins=bins, compress=False)
    else:    px.save_jpg(fn, wins, bins=bins)

def process_batch(pxs, fnames, dest, tiff, crop, resize, n_workers=4):
    ext = '.tif' if tiff else '.jpg'
    if crop or resize: pxs = TensorImage(pxs.cuda())
    if resize:
        tfm = AffineCoordTfm(size=256)
        pxs = tfm(pxs.unsqueeze(1)).squeeze()
    if crop:
        masks = pxs.mask_from_blur(dicom_windows.brain)
        bbs = mask2bbox(masks)
        pxs = crop_resize(pxs, bbs, 256)
    if crop or resize: pxs = pxs.cpu().squeeze()
    parallel(save_file, zip(fnames, pxs), n_workers=n_workers, progress=False, dest=dest, ext=ext, tiff=tiff)

@call_parse
def main(
    resize:Param("Resize to 256px"   , int)=0,
    test  :Param("Process test set"  , int)=0,
    tiff  :Param("Save TIFF format"  , int)=0,
    crop  :Param("Crop to brain area", int)=0,
    trial :Param("Just do 2 batches" , int)=0,
    bs    :Param("Batch size"        , int)=256,
    n_workers:Param("Number of workers", int)=8,
):
    print('resize,test,tiff,crop,trial,bs,n_workers')
    print(resize,test,tiff,crop,trial,bs,n_workers)
    df = df_tst if test else df_comb
    path_dcm = path_tst if test else path_trn
    fns = [path_dcm/f'{filename(o)}.dcm' for o in df.fname.values]
    dest_fn = ('tst_'  if test else
               'crop_' if crop else
               'nocrop_')
    dest_fn += 'tif' if tiff else 'jpg'
    if resize: dest_fn += '256'
    dest = path/dest_fn
    dest.mkdir(exist_ok=True)

    #fns2 = set([o.with_suffix('').name for o in dest.ls()])
    #fns3 = set([Path(o).with_suffix('').name for o in fns])
    #fns4 = [o for o in fns3 if o not in fns2]
    #fns5 = [path_trn/Path(o).with_suffix('.dcm') for o in fns4]

    dsrc = DataSource(fns, [[dcm_tfm],[os.path.basename]])
    dl = TfmdDL(dsrc, bs=bs, num_workers=1)

    for i,b in enumerate(progress_bar(dl)):
        process_batch(*b, dest=dest, tiff=tiff, crop=crop, resize=resize, n_workers=n_workers)
        if trial and i==2: return

