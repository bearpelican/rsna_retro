import fire,sys,os,torch
SCRIPT_NAME = os.path.basename(sys.argv[0]).replace('.py', '')
torch.backends.cudnn.benchmark = True

def main(pre, depth, arch, lr=1e-4, freeze=0, gpu=None, tta=0):
    from fastai2.distributed import setup_distrib, num_distrib, rank_distrib
    gpu,n_gpus,is_master = setup_distrib(gpu), num_distrib() or 1, rank_distrib() == 0
    from slice_e2e_shallow import do
    do(gpu, pre, depth, arch, lr=lr, freeze=freeze, tta=tta)

if __name__ == '__main__': fire.Fire(main)


