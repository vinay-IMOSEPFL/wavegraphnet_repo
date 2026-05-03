import argparse, sys, random, pickle
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as PyGData, Batch
from tqdm import tqdm

from utils.splits import get_train_val_test_ids
from utils.data_loader import CoupledModelDataset
from utils.precompute import build_all_stats, N_ATTENTION_FREQS
from utils.checkpointer import save_checkpoint, checkpoint_path
from models.wavegraphnet_new import InverseGNN, ForwardPhysicsGNN, DynamicWeightedLoss
from utils.logger import log_result, EpochLogger, evaluate_val, evaluate_test

_TQDM_DISABLE = not sys.stdout.isatty()

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def make_fwd_batch(data_inv, prop_ei, device):
    prop_ei = prop_ei.to(device)
    return Batch.from_data_list([
        PyGData(x=data_inv.x[data_inv.batch==i], edge_index=prop_ei)
        for i in range(data_inv.num_graphs)
    ]).to(device)

def train_fwd_only(fwd, loader, opt, prop_ei, device):
    fwd.train(); total = 0.0
    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        opt.zero_grad()
        di = batch["data_inv"].to(device); yt = batch["y_true"].to(device).squeeze(1)
        de = batch["delta_e_true"].to(device)
        w  = de + 0.01
        loss = (w*(fwd(make_fwd_batch(di, prop_ei, device), yt) - de)**2).mean()*100.0
        loss.backward(); opt.step(); total += loss.item()*di.num_graphs
    return total/len(loader.dataset)

def train_coupled(inv, fwd, dyn, loader, opt, prop_ei, device):
    inv.train(); fwd.train(); crit = nn.MSELoss()
    tl=tf=0.0
    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        opt.zero_grad()
        di=batch["data_inv"].to(device); yt=batch["y_true"].to(device).squeeze(1)
        de=batch["delta_e_true"].to(device)
        pc=inv(di); ll=crit(pc,yt)
        lf=crit(fwd(make_fwd_batch(di,prop_ei,device),pc),de)
        loss=dyn([ll,lf]); loss.backward()
        torch.nn.utils.clip_grad_norm_(inv.parameters(),1.0)
        torch.nn.utils.clip_grad_norm_(fwd.parameters(),1.0)
        opt.step(); n=di.num_graphs; tl+=ll.item()*n; tf+=lf.item()*n
    n=len(loader.dataset); return tl/n, tf/n

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--split",      default="B", choices=["A","B"])
    p.add_argument("--mode",       default="coupled", choices=["coupled","inverse_only"])
    p.add_argument("--fwd_epochs", type=int,   default=100)
    p.add_argument("--epochs",     type=int,   default=150)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--val_every",  type=int,   default=10)
    args=p.parse_args()
    set_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/processed/ogw_data.pkl","rb") as f: data_map=pickle.load(f)
    train_ids,val_ids,test_ids=get_train_val_test_ids(args.split,list(data_map.keys()),seed=args.seed)
    full_label="WaveGraphNet (Improved)"
    print(f"\n{'='*65}\n  {full_label} | split={args.split} seed={args.seed}\n{'='*65}",flush=True)

    stats=build_all_stats(data_map,train_ids)
    ef=3+N_ATTENTION_FREQS*2
    ds_kw=dict(data_map=data_map,
               inv_static_edge_index=stats["k12_edge_index"],
               inv_edge_feature_col_idxs=stats["inv_edge_feature_col_idxs"],
               fwd_propagation_col_idxs=stats["propagation_col_idxs"],
               fixed_fft_bin_indices=stats["fixed_fft_bin_indices"],
               amp_means=stats["amp_means"],amp_stds=stats["amp_stds"],
               lookback_fft=stats["lookback_fft"],
               average_baseline_energy_profile=stats["average_baseline_energy_profile"],
               global_max_delta_e=stats["global_max_delta_e"])
    tr=DataLoader(CoupledModelDataset(sample_id_list=train_ids,**ds_kw),batch_size=args.batch_size,shuffle=True)
    vl=DataLoader(CoupledModelDataset(sample_id_list=val_ids,  **ds_kw),batch_size=args.batch_size,shuffle=False)
    te=DataLoader(CoupledModelDataset(sample_id_list=test_ids, **ds_kw),batch_size=args.batch_size,shuffle=False)

    n_prop=len(stats["propagation_pair_indices"])
    inv=InverseGNN(node_in=2,edge_in=ef,hidden=128).to(device)
    fwd=ForwardPhysicsGNN(edge_in=6,hidden=128,num_propagation_pairs=n_prop).to(device)
    dyn=DynamicWeightedLoss(num_losses=2).to(device)
    prop_ei=stats["propagation_edge_index"]
    ckpt=checkpoint_path(args.split,full_label,args.seed)
    logger=EpochLogger(args.split,full_label,args.seed)

    if args.mode=="coupled" and args.fwd_epochs>0:
        print(f"\n--- Stage 1: Forward pre-training ({args.fwd_epochs} epochs) ---",flush=True)
        of=optim.Adam(fwd.parameters(),lr=args.lr)
        sf=optim.lr_scheduler.ReduceLROnPlateau(of,factor=0.8,patience=20)
        for ep in range(1,args.fwd_epochs+1):
            fl=train_fwd_only(fwd,tr,of,prop_ei,device); sf.step(fl)
            if ep%args.val_every==0 or ep in(1,args.fwd_epochs):
                vm,vmm=evaluate_val(inv,vl,device); tm,tmm=evaluate_test(inv,te,device)
                logger.log(ep,"fwd_pt",fl,vm,vmm,tm,tmm,lr=of.param_groups[0]["lr"])

    print(f"\n--- Stage 2: Coupled ({args.epochs} epochs) ---",flush=True)
    opt=optim.Adam(list(inv.parameters())+list(fwd.parameters())+list(dyn.parameters()),lr=args.lr)
    sch=optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.8,patience=20)
    best_vm=float("inf"); best_tm=float("inf"); best_tmm=float("nan")

    for epoch in range(1,args.epochs+1):
        ll,lf=train_coupled(inv,fwd,dyn,tr,opt,prop_ei,device); sch.step(ll)
        if epoch%args.val_every==0 or epoch in(1,args.epochs):
            vm,vmm=evaluate_val(inv,vl,device); tm,tmm=evaluate_test(inv,te,device)
            logger.log(epoch,"train",ll,vm,vmm,tm,tmm,lr=opt.param_groups[0]["lr"])
            if vm<best_vm:
                best_vm=vm; best_tm=tm; best_tmm=tmm
                save_checkpoint(ckpt,config=vars(args),test_loss=tm,val_loss=vm,
                                inv_model=inv.state_dict(),fwd_model=fwd.state_dict(),
                                dynamic_loss=dyn.state_dict())
                print(f"  ★ New best val MSE={vm:.5f} → test MSE={tm:.5f} ({tmm:.1f}mm) [saved]",flush=True)

    logger.close()
    log_result(args.split,f"{full_label} (seed={args.seed})",best_tm)
    print(f"\n[DONE] {full_label} | best_val_mse={best_vm:.5f} | test_mse={best_tm:.5f} | test_mae={best_tmm:.1f}mm",flush=True)

if __name__=="__main__": main()