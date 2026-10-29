"""Evaluation adapter for published learned models. No new training/architecture."""
from pathlib import Path
import hashlib, json, os, subprocess, sys, time

JEPA_COMMIT = '13cf1d9c7e476f53c17714d2e0f1dc239a883ce0'
DINO_COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
DINO_WEIGHT_SHA = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def git_head(path):
    return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()

def load_model(repo,dinorepo,checkpoints,kind='jepa',device='cpu'):
    import torch, yaml
    repo,dinorepo,checkpoints=map(lambda x:Path(x).resolve(),(repo,dinorepo,checkpoints))
    if git_head(repo)!=JEPA_COMMIT or git_head(dinorepo)!=DINO_COMMIT:
        raise ValueError('Source revision mismatch')
    sys.path.insert(0,str(repo))
    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_transforms
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module
    from app.vjepa_wm.utils import clean_state_dict
    directory='jepa-wm' if kind=='jepa' else 'dino-wm'
    cfgpath=repo/f'configs/evals/simu_env_planning/pt/{directory}/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml'
    cfg=yaml.safe_load(cfgpath.read_text())['model_kwargs']
    cfg['pretrain_kwargs']['heads_cfg']={}
    stats=get_data_stats('pusht')
    pre=Preprocessor(**{k:torch.tensor(stats[k]) for k in
        ('action_mean','action_std','state_mean','state_std','proprio_mean','proprio_std')},
        transform=make_transforms(img_size=224,normalize=cfg['data_aug']['normalize'],
        random_horizontal_flip=False,random_resize_aspect_ratio=(1.,1.),random_resize_scale=(1.,1.)))
    checkpoint=checkpoints/f'{kind}_wm_pusht.pth.tar'
    expected=json.loads((Path(__file__).parent/'checkpoints.json').read_text())[f'{kind}_wm_pusht']['sha256']
    if sha256(checkpoint)!=expected: raise ValueError('Checkpoint hash mismatch')
    original_load=torch.hub.load
    def pinned_load(repository,name,*args,**kwargs):
        if repository=='facebookresearch/dinov2':
            return original_load(str(dinorepo),name,*args,source='local',**kwargs)
        return original_load(repository,name,*args,**kwargs)
    torch.hub.load=pinned_load
    started=time.perf_counter()
    try:
        model=init_module(folder=str(checkpoints),checkpoint=checkpoint.name,
            model_kwargs=cfg['pretrain_kwargs'],device=torch.device(device),
            action_dim=2,proprio_dim=4,preprocessor=pre,cfgs_data=cfg['data'],wrapper_kwargs=cfg['wrapper_kwargs'])
    finally: torch.hub.load=original_load
    # Upstream strict=False is independently checked against every named parameter.
    data=torch.load(checkpoint,map_location='cpu',weights_only=True); checks={}
    for name in ('predictor','action_encoder','proprio_encoder'):
        module=getattr(model.model,name,None)
        if module is None: continue
        params=dict(module.named_parameters())
        if not params: continue
        if data.get(name) is None: raise ValueError('Missing pretrained component '+name)
        weights=clean_state_dict(data[name])
        weights={k.replace('state_encoder.','proprio_encoder.'):v for k,v in weights.items()}
        for k,p in params.items():
            if k not in weights or not torch.equal(p.detach().cpu(),weights[k]):
                raise ValueError('Unverified pretrained parameter '+name+'.'+k)
        checks[name]={'parameter_tensors':len(params),'exact_match':True}
    encpath=Path(torch.hub.get_dir())/'checkpoints/dinov2_vits14_pretrain.pth'
    if sha256(encpath)!=DINO_WEIGHT_SHA: raise ValueError('DINO encoder weight hash mismatch')
    model.eval(); model.requires_grad_(False)
    prov={'kind':kind,'jepa_commit':JEPA_COMMIT,'dinov2_commit':DINO_COMMIT,
        'checkpoint_sha256':expected,'dinov2_weight_sha256':DINO_WEIGHT_SHA,
        'config_sha256':sha256(cfgpath),'unused_visualization_heads_disabled':True,
        'parameter_checks':checks,'load_seconds':time.perf_counter()-started}
    return model,pre,prov

def make_env(repo):
    os.environ.setdefault('SDL_VIDEODRIVER','dummy'); os.environ.setdefault('SDL_AUDIODRIVER','dummy')
    sys.path.insert(0,str(Path(repo).resolve()))
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv
    return PushTEnv(with_velocity=True,with_target=True,render_size=224)

def encode_obs(model,obs):
    import numpy as np, torch
    return model.encode({'visual':torch.from_numpy(np.asarray(obs['visual']).copy()).permute(2,0,1)[None,None],
        'proprio':torch.from_numpy(np.asarray(obs['proprio'],dtype=np.float32).copy())[None,None]})

def predict_costs(model,pre,obs,goal,actions):
    """Terminal official L2+0.1 proprio objective; one reset frame; 5 raw steps/macro."""
    import numpy as np, torch
    actions=np.asarray(actions,dtype=np.float32); b,t,a=actions.shape
    if a!=2 or t%5: raise ValueError('Expected [B,5*H,2] raw relative controls')
    started=time.perf_counter()
    with torch.inference_mode():
        z=encode_obs(model,obs); zg=encode_obs(model,goal)
        normalized=pre.normalize_actions(torch.from_numpy(actions))
        packed=normalized.reshape(b,t//5,10).transpose(0,1).to(model.device)
        rollout=model.unroll(z,packed)
        cost=((rollout['visual'][-1]-zg['visual'][0,0])**2).flatten(1).mean(1)
        cost+=.1*((rollout['proprio'][-1]-zg['proprio'][0,0])**2).flatten(1).mean(1)
        cost=cost.detach().cpu().numpy()
    if not np.isfinite(cost).all(): raise FloatingPointError('Nonfinite forecast')
    return cost,time.perf_counter()-started
