# -*- coding: utf-8 -*-
"""D85：在 frozen D82 上只訓練 RIDE logits 修正。"""
import argparse, csv, json, os
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_d77_fused_lora import load_frozen_lora_model, fuse_tom_logits, evaluate_fixed_fusion
from run_six_class_validation import LABEL_INDEX
from train_six_class_candidate import batch_from_schedule, gaussian_smooth_targets

RIDE = LABEL_INDEX['RIDE']
QUOTAS = {'d36_whack_real': 300, 'd36_archive_synthetic': 100}

class RideOnlyHead(nn.Module):
    """中文註解：重用 frozen D82 head，僅在 RIDE 通道加入一個 rank-4 修正。"""
    def __init__(self, base):
        super().__init__()
        self.base = base
        for parameter in base.parameters(): parameter.requires_grad = False
        self.coefficients = nn.Parameter(torch.zeros(base.rank))
    def forward(self, x):
        """中文註解：除 RIDE 外逐值保留 D82 logits。"""
        logits = self.base(x)
        delta = self.base.scale * (self.base.down(x) * self.coefficients.view(1, -1, 1)).sum(dim=1)
        output = logits.clone()
        output[:, RIDE, :] = output[:, RIDE, :] + delta
        return output

def load_d82(model, payload, key, device):
    """中文註解：載入既有 D82 adapter，供 D85 凍結為起點。"""
    state = payload[key]
    with torch.no_grad():
        model.onset_head.down.weight.copy_(state['down.weight'].to(device))
        model.onset_head.up.weight.copy_(state['up.weight'].to(device))

def schedule(path):
    """中文註解：從 D84 唯讀候選固定取 Whack 300 與 Archive 100。"""
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8')))
    out = []
    for source, quota in QUOTAS.items():
        candidates = [row for row in rows if row['source'] == source]
        if len(candidates) < quota: raise ValueError(f'{source} has {len(candidates)}, need {quota}')
        out += [candidates[index * len(candidates) // quota] for index in range(quota)]
    return [{'label':'RIDE','key':r['key'],'anchor':float(r['anchor'])} for r in out]

def main():
    """中文註解：建立 D85 candidate，僅儲存 RIDE 修正與固定驗收報告。"""
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',default='validation_runs/d85_ride_only_candidate'); p.add_argument('--epochs',type=int,default=5); p.add_argument('--lr',type=float,default=.001); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--self-check',action='store_true'); a=p.parse_args()
    if a.self_check:
        h=RideOnlyHead(type('B',(nn.Module,),{'rank':4,'scale':2.0,'down':nn.Conv1d(3,4,1),'forward':lambda s,x:torch.zeros(x.shape[0],6,x.shape[2])})()); assert h(torch.randn(1,3,2)).shape==(1,6,2); print('D85 self-check passed.'); return
    if os.path.exists(a.output_dir): raise FileExistsError(a.output_dir)
    meta=json.load(open('mixed_d54_stem/metadata_d54.json',encoding='utf-8')); rows=schedule('validation_runs/d84_ride_sd_competitor_feasibility/ride_sd_candidates.csv'); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    payload=torch.load('validation_runs/d82_d77_fused_lora_candidate/d82_d77_fused_lora_adapter.pth',map_location='cpu',weights_only=False)
    d76=load_frozen_lora_model(payload['base_d76_checkpoint'],device,payload['rank'],payload['alpha']); d64=load_frozen_lora_model(payload['base_d64_checkpoint'],device,payload['rank'],payload['alpha']); load_d82(d76,payload,'d76_onset_lora',device); load_d82(d64,payload,'d64_onset_lora',device); d76.onset_head=RideOnlyHead(d76.onset_head).to(device); d76.eval(); d64.eval(); opt=torch.optim.Adam([d76.onset_head.coefficients],lr=a.lr); os.makedirs(a.output_dir); best=-1
    for epoch in range(1,a.epochs+1):
        losses=[]
        for start in range(0,len(rows),a.batch_size):
            x,y,_=batch_from_schedule(rows,meta,start,a.batch_size,use_true_superflux=True,input_mode='drumsep-mix'); x=torch.from_numpy(x).float().to(device); y=gaussian_smooth_targets(torch.from_numpy(y).float().to(device)); opt.zero_grad(); z76,_=d76(x); z64,_=d64(x); loss=F.binary_cross_entropy_with_logits(fuse_tom_logits(z76,z64)[...,RIDE],y[...,RIDE]); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        macro,per,gate=evaluate_fixed_fusion(d76,d64,meta,'validation_runs/d61_kd_negative_candidate/independent_validation/selected_windows.json',os.path.join(a.output_dir,f'epoch_{epoch:02d}_fixed_validation'))
        if macro>best: best=macro; torch.save({'phase':'D85','d82_adapter':'validation_runs/d82_d77_fused_lora_candidate/d82_d77_fused_lora_adapter.pth','coefficients':d76.onset_head.coefficients.detach().cpu(),'epoch':epoch,'macro_f1':macro,'per_class':per},os.path.join(a.output_dir,'d85_ride_only_adapter.pth'))
        print(json.dumps({'epoch':epoch,'loss':sum(losses)/len(losses),'macro_f1':macro,'per_class':per,'gate':gate},ensure_ascii=False))
if __name__=='__main__': main()
