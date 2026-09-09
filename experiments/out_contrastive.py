"""OUT-only contrastive transfer with a matched frozen-random control.

Inspired by hierarchical contrastive learning, not an exact TS2Vec reproduction.
Three fixed seeds, source-only pretraining; mean/log-variance pooled features.
"""
import argparse
import copy
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from experiments.out_transfer import Encoder, load_bags, pack, metrics

SEEDS=(0,1,2)
ARMS=('spectral','random_frozen','contrastive_frozen','fusion_half')


def paired_loss(a,b):
    """Last axis is embedding, second-last is instances or time positions."""
    n=a.shape[-2]
    if n==1:return (a.sum()+b.sum())*0
    z=F.normalize(torch.cat([a,b],dim=-2),dim=-1)
    similarity=(z@z.transpose(-1,-2))/.2
    similarity=similarity.masked_fill(torch.eye(2*n,device=z.device,dtype=torch.bool),float('-inf'))
    labels=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return F.cross_entropy(similarity.reshape(-1,2*n),labels.repeat(z.shape[0]))


def hierarchy(a,b):
    """Equal instance/time losses averaged across max-pooling resolutions."""
    losses=[]
    while True:
        aa=a.transpose(1,2);bb=b.transpose(1,2)  # B,T,C
        ins=paired_loss(aa.transpose(0,1),bb.transpose(0,1))
        if a.shape[-1]>1:losses.append((ins+paired_loss(aa,bb))/2)
        else:losses.append(ins);break
        a=F.max_pool1d(a,2);b=F.max_pool1d(b,2)
    return torch.stack(losses).mean()


def pretrain(bags,seed,scale):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    enc=Encoder();random=copy.deepcopy(enc)
    opt=torch.optim.Adam(enc.parameters(),lr=.001,weight_decay=.0001)
    losses=[]
    for epoch in range(40):
        values=[]
        for _ in range(5):
            # Distinct source patients; no same-patient negative pairs in a batch.
            ix=rng.choice(len(bags),min(32,len(bags)),replace=False)
            x=torch.tensor(np.stack([bags[i][rng.integers(len(bags[i]))] for i in ix])/scale,dtype=torch.float32)
            m1=torch.rand(len(x),1,x.shape[-1])<.2;m2=torch.rand(len(x),1,x.shape[-1])<.2
            loss=hierarchy(enc(x.masked_fill(m1,0)),enc(x.masked_fill(m2,0)))
            if not torch.isfinite(loss):raise ValueError('Nonfinite pretraining loss')
            opt.zero_grad();loss.backward();opt.step();values.append(loss.item())
        losses.append(float(np.mean(values)))
    enc.eval();random.eval()
    return random,enc,losses


def embed(enc,bags,scale):
    x,owners=pack(bags,scale)
    with torch.no_grad():
        h=enc(x)
        v=torch.cat([h.mean(-1),torch.log(h.var(-1,unbiased=False)+1e-6)],dim=1)
        pooled=v.new_zeros((len(bags),v.shape[1])).index_add(0,owners,v)
        return (pooled/torch.bincount(owners)[:,None]).numpy()


def head(x,y,test):
    m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=3000))
    m.fit(x,y)
    return m.predict_proba(test)


def main():
    from common.quaternion_data import load_quaternion_recordings_multi
    from common.load_2025 import load_2025_all
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',required=True);args=a.parse_args();out=Path(args.output)
    if out.exists():a.error('Choose a new output directory')
    torch.set_num_threads(1)
    print('Loading OUT only',flush=True)
    b,y,p,d=load_bags(load_quaternion_recordings_multi('Data',['OUT']),'2015')
    nb,ny,pn,nd=load_bags(load_2025_all(conditions=('OUT',)),'NewData')
    if np.bincount(y).tolist()!=[61,75,15] or np.bincount(ny).tolist()!=[27,23,6]:raise ValueError('Unexpected cohort counts')
    scale=np.sqrt(np.mean([np.mean(v*v,axis=(0,2)) for v in nb],axis=0))[None,:,None]+1e-6
    out.mkdir(parents=True)
    protocol=dict(action='OUT',target_counts=np.bincount(y).tolist(),source_counts=np.bincount(ny).tolist(),seeds=SEEDS,
        pretraining='source NewData only; 40 epochs x 5 batches; 32 distinct patients; 20% independent time masks; normalized contrastive temperature .2',
        primary='contrastive_frozen minus matched random_frozen macro-F1',
        secondary='contrastive and fixed half fusion versus spectral',
        head='fixed balanced logistic C=.1; fold-local StandardScaler; 32 mean/log-variance features for both encoders',
        comparison='same encoder initialization, pooling, source scale, labelled training patients and classifier for random/pretrained',
        target_evaluation='same 5 seed-0 outer folds and reserved 20% inner validation as PR #7; reserved validation is unused in this fixed-head comparison',
        hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('target_spectral',d),('source_spectral',nd)]},
        note='Small TS2Vec-inspired adaptation, not original implementation or large pretrained checkpoint. No exclusions or seed selection.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    all_emb={};losses={}
    for seed in SEEDS:
        print('Pretraining seed',seed,flush=True)
        random,trained,loss=pretrain(nb,seed,scale);losses[str(seed)]=loss
        all_emb[seed]={name:(embed(enc,b,scale),embed(enc,nb,scale)) for name,enc in [('random_frozen',random),('contrastive_frozen',trained)]}
        torch.save(trained.state_dict(),out/f'encoder_seed{seed}.pt')
    probs={arm:np.zeros((len(y),3)) for arm in ARMS};seed_probs={seed:{arm:np.zeros((len(y),3)) for arm in ARMS[1:3]} for seed in SEEDS};splits=[]
    for fold,(tv,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=0).split(d,y)):
        tr0,va0=next(StratifiedShuffleSplit(1,test_size=.2,random_state=fold).split(d[tv],y[tv]));tr,va=tv[tr0],tv[va0]
        splits.append(dict(train=p[tr].tolist(),validation=p[va].tolist(),test=p[te].tolist(),source_train=pn.tolist()))
        yy=np.r_[y[tr],ny]
        probs['spectral'][te]=head(np.vstack([d[tr],nd]),yy,d[te])
        for seed in SEEDS:
            for arm in ARMS[1:3]:
                x,xn=all_emb[seed][arm];seed_probs[seed][arm][te]=head(np.vstack([x[tr],xn]),yy,x[te])
        for arm in ARMS[1:3]:probs[arm][te]=np.mean([seed_probs[s][arm][te] for s in SEEDS],axis=0)
        probs['fusion_half'][te]=(probs['spectral'][te]+probs['contrastive_frozen'][te])/2
        print('Fold',fold+1,'complete',flush=True)
    pred={a:v.argmax(1) for a,v in probs.items()};result={a:metrics(y,v) for a,v in pred.items()}
    comparisons=[('contrastive_frozen','random_frozen'),('contrastive_frozen','spectral'),('fusion_half','spectral')]
    delta={f'{a}-minus-{b}':[] for a,b in comparisons};rng=np.random.default_rng(0)
    for _ in range(2000):
        ix=rng.integers(len(y),size=len(y));score={a:metrics(y[ix],v[ix])['macro_f1'] for a,v in pred.items()}
        for a,b in comparisons:delta[f'{a}-minus-{b}'].append(score[a]-score[b])
    result.update(protocol=protocol,source_epoch_losses=losses,
        per_seed={str(s):{a:metrics(y,v.argmax(1)) for a,v in seed_probs[s].items()} for s in SEEDS},
        paired_macro_f1_95={k:np.percentile(v,[2.5,97.5]).tolist() for k,v in delta.items()})
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(out/'splits.json').write_text(json.dumps(splits,indent=2))
    np.savez_compressed(out/'predictions.npz',patients=p,y=y,**probs)
    for a in ARMS:print(a,result[a]['macro_f1'],result[a]['precision'][2],result[a]['recall'][2])
    print(result['paired_macro_f1_95'])

if __name__=='__main__':main()
