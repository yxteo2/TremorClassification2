"""Single-action source-pretrained encoder pilot; no REST or PADS.

NewData OUT is source-only. 2015 OUT is target, five grouped outer folds.
Fixed seed 0, no architecture/hyperparameter search. Not a TS2Vec reproduction.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.signal import butter, sosfiltfilt, resample_poly
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from experiments.task_power_benchmark import features

ARMS=('spectral','scratch','pretrained_frozen','pretrained_last_block')


def windows(x):
    if x.shape[0]!=9 or x.shape[1]<400 or not np.isfinite(x).all():
        raise ValueError('Expected finite nine-channel OUT recording >=4 seconds')
    z=resample_poly(sosfiltfilt(butter(4,[3,15],fs=100,btype='bandpass',output='sos'),x,axis=-1),2,5,axis=-1)
    starts=np.arange(0,z.shape[1]-159,160)
    return np.stack([z[:,s:s+160] for s in starts]).astype('float32')


def load_bags(records,prefix):
    bags={};desc={};labels={}
    for r in records:
        if r.condition!='OUT':raise ValueError('Only OUT is permitted')
        p=prefix+':'+r.subject
        if p in labels and labels[p]!=r.y:raise ValueError('Conflicting labels')
        labels[p]=r.y;bags.setdefault(p,[]).extend(windows(r.x))
        desc.setdefault(p,[]).append(np.concatenate([features(r.x[s:s+3]) for s in (0,3,6)]))
    ps=sorted(bags)
    # Deterministically cap compute, without selecting by labels or signal quality.
    b=[np.stack(bags[p])[np.linspace(0,len(bags[p])-1,min(8,len(bags[p])),dtype=int)] for p in ps]
    return b,np.array([labels[p] for p in ps]),np.array(ps),np.array([np.mean(desc[p],axis=0) for p in ps])


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=nn.Sequential(nn.Conv1d(9,16,9,stride=2,padding=4),nn.GELU())
        self.last=nn.Sequential(nn.Conv1d(16,16,9,stride=2,padding=4),nn.GELU())
    def forward(self,x):return self.last(self.first(x))


def pack(bags,scale):
    x=torch.tensor(np.concatenate(bags)/scale,dtype=torch.float32)
    owner=torch.tensor(np.repeat(np.arange(len(bags)),[len(b) for b in bags]))
    return x,owner


def pooled(encoder,x,owner,n):
    h=encoder(x).mean(-1)
    out=h.new_zeros((n,h.shape[1])).index_add(0,owner,h)
    return out/torch.bincount(owner,minlength=n).to(h.dtype)[:,None]


def pretrain(bags):
    torch.manual_seed(0);rng=np.random.default_rng(0)
    # Source-only amplitude scale; no target test or validation data used.
    scale=np.sqrt(np.mean([np.mean(b*b,axis=(0,2)) for b in bags],axis=0))[None,:,None]+1e-6
    enc=Encoder();decoder=nn.Sequential(nn.ConvTranspose1d(16,16,8,2,3),nn.GELU(),nn.ConvTranspose1d(16,9,8,2,3))
    opt=torch.optim.Adam(list(enc.parameters())+list(decoder.parameters()),lr=.001)
    for epoch in range(30):
        for _ in range(5):
            # One window per source patient prevents long records dominating SSL.
            x=torch.tensor(np.stack([b[rng.integers(len(b))] for b in bags])/scale,dtype=torch.float32)
            mask=(torch.rand(len(x),1,x.shape[-1])<.2).expand_as(x)
            reconstructed=decoder(enc(x.masked_fill(mask,0)))
            loss=((reconstructed-x)**2)[mask].mean()
            opt.zero_grad();loss.backward();opt.step()
    return enc,scale


def supervised(bags,y,val_bags,val_y,test_bags,scale,initial=None):
    torch.manual_seed(0)
    enc=Encoder() if initial is None else copy.deepcopy(initial)
    if initial is not None:
        for p in enc.first.parameters():p.requires_grad=False
    head=nn.Linear(16,3)
    opt=torch.optim.Adam([p for p in list(enc.parameters())+list(head.parameters()) if p.requires_grad],lr=.001 if initial is None else .0003,weight_decay=.001)
    xx,oo=pack(bags,scale);vx,vo=pack(val_bags,scale);tx,to=pack(test_bags,scale)
    weight=torch.tensor(len(y)/(3*np.bincount(y,minlength=3)),dtype=torch.float32)
    lossfn=nn.CrossEntropyLoss(weight=weight);yy=torch.tensor(y);vy=torch.tensor(val_y)
    best=float('inf');state=None;bestepoch=0
    for epoch in range(100):
        enc.train();head.train();opt.zero_grad()
        loss=lossfn(head(pooled(enc,xx,oo,len(y))),yy);loss.backward();opt.step()
        enc.eval();head.eval()
        with torch.no_grad():value=lossfn(head(pooled(enc,vx,vo,len(val_y))),vy).item()
        if value<best:
            best=value;state=(copy.deepcopy(enc.state_dict()),copy.deepcopy(head.state_dict()));bestepoch=epoch+1
    enc.load_state_dict(state[0]);head.load_state_dict(state[1]);enc.eval();head.eval()
    with torch.no_grad():p=head(pooled(enc,tx,to,len(test_bags))).softmax(-1).numpy()
    return p,bestepoch


def metrics(y,p):
    pr,re,f,s=precision_recall_fscore_support(y,p,labels=[0,1,2],zero_division=0)
    return dict(accuracy=accuracy_score(y,p),macro_f1=float(f.mean()),precision=pr.tolist(),recall=re.tolist(),support=s.tolist(),confusion=confusion_matrix(y,p,labels=[0,1,2]).tolist())


def main():
    from common.quaternion_data import load_quaternion_recordings_multi
    from common.load_2025 import load_2025_all
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',required=True);args=a.parse_args();out=Path(args.output)
    if out.exists():a.error('Choose a new output directory')
    torch.set_num_threads(1)
    print('Loading OUT only',flush=True)
    b,y,p,d=load_bags(load_quaternion_recordings_multi('Data',['OUT']),'2015')
    nb,ny,npats,nd=load_bags(load_2025_all(conditions=('OUT',)),'NewData')
    if np.bincount(y).tolist()!=[61,75,15] or np.bincount(ny).tolist()!=[27,23,6]:raise ValueError('Unexpected patient counts')
    out.mkdir(parents=True)
    protocol=dict(action='OUT',target_counts=np.bincount(y).tolist(),source_counts=np.bincount(ny).tolist(),seed=0,
        source_pretraining='NewData only, 30 epochs x 5 steps; 20% time masks, reconstruction MSE; no source labels',
        window='3-15Hz,40Hz,4s nonoverlap,max8/patient; partial tails discarded, no padding',
        supervised='same NewData plus target outer train in every arm; inner 20% target validation; 100 epochs, minimum validation weighted CE',
        primary='pretrained_frozen minus scratch macro-F1; last-block adaptation secondary',
        notes='Exploratory masked autoencoder pilot, not TS2Vec; source subjects assumed distinct from target; no patient exclusions by model errors')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('Pretraining source encoder',flush=True);enc,scale=pretrain(nb)
    torch.save(dict(encoder=enc.state_dict(),scale=scale),out/'source_encoder.pt')
    enc.eval()
    def embed(bags):
        xx,oo=pack(bags,scale)
        with torch.no_grad():return pooled(enc,xx,oo,len(bags)).numpy()
    # Independent per-patient transform; no fitting to target features.
    emb=embed(b);nemb=embed(nb)
    probs={arm:np.zeros((len(y),3)) for arm in ARMS};splits=[];epochs=[]
    for fold,(tv,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=0).split(d,y)):
        tr0,va0=next(StratifiedShuffleSplit(1,test_size=.2,random_state=fold).split(d[tv],y[tv]));tr,va=tv[tr0],tv[va0]
        splits.append(dict(train=p[tr].tolist(),validation=p[va].tolist(),test=p[te].tolist(),source_train=npats.tolist()))
        yy=np.r_[y[tr],ny];training=[b[i] for i in tr]+nb;validation=[b[i] for i in va];testing=[b[i] for i in te]
        for arm,x,xn in [('spectral',d,nd),('pretrained_frozen',emb,nemb)]:
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=3000))
            clf.fit(np.vstack([x[tr],xn]),yy);probs[arm][te]=clf.predict_proba(x[te])
        ep={}
        for arm,init in [('scratch',None),('pretrained_last_block',enc)]:
            probs[arm][te],ep[arm]=supervised(training,yy,validation,y[va],testing,scale,init)
        epochs.append(ep)
        np.savez_compressed(out/f'fold_{fold}.npz',patients=p[te],y=y[te],**{arm:probs[arm][te] for arm in ARMS})
        print(f'Fold {fold+1}/5 complete',flush=True)
    pred={a:v.argmax(1) for a,v in probs.items()};result={a:metrics(y,v) for a,v in pred.items()}
    rng=np.random.default_rng(0);delta={a:[] for a in ['pretrained_frozen','pretrained_last_block']}
    for _ in range(2000):
        ix=rng.integers(len(y),size=len(y));base=metrics(y[ix],pred['scratch'][ix])['macro_f1']
        for arm in delta:delta[arm].append(metrics(y[ix],pred[arm][ix])['macro_f1']-base)
    result.update(protocol=protocol,epochs=epochs,paired_macro_f1_95_vs_scratch={a:np.percentile(v,[2.5,97.5]).tolist() for a,v in delta.items()})
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(out/'splits.json').write_text(json.dumps(splits,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
