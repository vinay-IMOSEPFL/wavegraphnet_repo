import argparse, sys, random, pickle
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, scipy.fft
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.splits import get_train_val_test_ids
from utils.data_loader import parse_damage_label, DAMAGE_LABELS
from utils.precompute import LOOKBACK_POINTS, N_ATTENTION_FREQS, FIXED_FFT_BIN_INDICES, compute_amp_stats
from utils.checkpointer import save_checkpoint, checkpoint_path
from models.cnn1d import PaperCnnBaseline
from utils.logger import log_result, EpochLogger, evaluate_val, evaluate_test

_TQDM_DISABLE = not sys.stdout.isatty()

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class Cnn1DDataset(Dataset):
    def __init__(self, data_map, sample_id_list, bins, amp_means, amp_stds, lookback):
        self.data_map=data_map; self.sample_ids=sample_id_list
        self.lookback=lookback; self.bins=bins
        self.amp_means=torch.tensor(amp_means,dtype=torch.float32)
        self.amp_stds =torch.tensor(amp_stds, dtype=torch.float32)
        self.nf=len(bins)
    def __len__(self): return len(self.sample_ids)
    def __getitem__(self,idx):
        sid=self.sample_ids[idx]; sig=self.data_map[sid]; np_=sig.shape[1]
        ff=scipy.fft.rfft(sig[:self.lookback,:],n=self.lookback,axis=0)
        a=torch.from_numpy(np.abs(ff[self.bins,:]).astype(np.float32)).T
        ph=torch.from_numpy(np.angle(ff[self.bins,:]).astype(np.float32)).T
        na=(a-self.amp_means)/self.amp_stds
        x=torch.zeros((np_*2,self.nf),dtype=torch.float32)
        x[0::2]=na; x[1::2]=ph
        dmg=parse_damage_label(sid); xd,yd=-0.001,-0.001
        if dmg!="undamaged" and dmg in DAMAGE_LABELS: xd,yd=DAMAGE_LABELS[dmg]
        return x,torch.tensor([xd,yd],dtype=torch.float)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--split",      default="A",choices=["A","B"])
    p.add_argument("--epochs",     type=int,  default=150)
    p.add_argument("--batch_size", type=int,  default=32)
    p.add_argument("--lr",         type=float,default=1e-3)
    p.add_argument("--seed",       type=int,  default=42)
    p.add_argument("--val_every",  type=int,  default=10)
    args=p.parse_args()
    set_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("data/processed/ogw_data.pkl","rb") as f: data_map=pickle.load(f)
    train_ids,val_ids,test_ids=get_train_val_test_ids(args.split,list(data_map.keys()),seed=args.seed)
    print(f"\n{'='*65}\n  1D CNN | split={args.split} seed={args.seed}\n{'='*65}",flush=True)
    print("─ Computing amplitude statistics …")
    amp_means,amp_stds=compute_amp_stats(data_map,train_ids,FIXED_FFT_BIN_INDICES)
    np_=list(data_map.values())[0].shape[1]
    ds_kw=dict(data_map=data_map,bins=FIXED_FFT_BIN_INDICES,
               amp_means=amp_means,amp_stds=amp_stds,lookback=LOOKBACK_POINTS)
    tr =DataLoader(Cnn1DDataset(sample_id_list=train_ids,**ds_kw),batch_size=args.batch_size,shuffle=True, num_workers=2)
    vl =DataLoader(Cnn1DDataset(sample_id_list=val_ids,  **ds_kw),batch_size=args.batch_size,shuffle=False,num_workers=2)
    te =DataLoader(Cnn1DDataset(sample_id_list=test_ids, **ds_kw),batch_size=args.batch_size,shuffle=False,num_workers=2)
    model=PaperCnnBaseline(in_channels=np_*2,num_classes=2).to(device)
    opt=optim.Adam(model.parameters(),lr=args.lr)
    sch=optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.8,patience=20)
    crit=nn.MSELoss()
    ckpt=checkpoint_path(args.split,"1D CNN",args.seed)
    logger=EpochLogger(args.split,"1D CNN",args.seed)
    best_vm=float("inf"); best_tm=float("inf"); best_tmm=float("nan")

    for epoch in range(1,args.epochs+1):
        model.train(); tl=0.0
        for x,y in tqdm(tr,leave=False,disable=_TQDM_DISABLE):
            x,y=x.to(device),y.to(device); opt.zero_grad()
            loss=crit(model(x),y); loss.backward(); opt.step()
            tl+=loss.item()*x.size(0)
        tl/=len(tr.dataset); sch.step(tl)
        if epoch%args.val_every==0 or epoch in(1,args.epochs):
            vm,vmm=evaluate_val(model,vl,device); tm,tmm=evaluate_test(model,te,device)
            logger.log(epoch,"train",tl,vm,vmm,tm,tmm,lr=opt.param_groups[0]["lr"])
            if vm<best_vm:
                best_vm=vm; best_tm=tm; best_tmm=tmm
                save_checkpoint(ckpt,config=vars(args),test_loss=tm,val_loss=vm,model=model.state_dict())
                print(f"  ★ New best val MSE={vm:.5f} → test MSE={tm:.5f} ({tmm:.1f}mm) [saved]",flush=True)

    logger.close()
    log_result(args.split,f"1D CNN (seed={args.seed})",best_tm)
    print(f"\n[DONE] 1D CNN | best_val_mse={best_vm:.5f} | test_mse={best_tm:.5f} | test_mae={best_tmm:.1f}mm",flush=True)

if __name__=="__main__": main()