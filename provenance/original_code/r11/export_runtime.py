"""Export the runtime normalizer, encoder matrices and provenance used by Claude's adapter (2026-10-28; requested by
ChatGPT's R8-R10 review). Reads the pinned upstream repository's own statistics function; loads no world model.
Also reports the physical-unit sensitivity W diag(1/std) of each proprio encoder.
Usage: python3 -I export_runtime.py out.json
"""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch

WM = Path('/home/claude/wm'); REPO = WM / 'runtime/jepa-wms'; DINO = WM / 'runtime/dinov2'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


sys.path.insert(0, str(REPO))
from app.plan_common.datasets import get_data_stats
st = {k: np.asarray(v, float).tolist() for k, v in get_data_stats('pusht').items() if k in ('action_mean', 'action_std', 'state_mean', 'state_std', 'proprio_mean', 'proprio_std')}
out = {'pusht_stats_from_upstream_get_data_stats': st,
       'jepa_wms_commit': subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
       'dinov2_commit': subprocess.run(['git', '-C', str(DINO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
       'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
       'code_sha256': {f: sha(WM / p) for f, p in [('jepa_adapter.py', 'v06_reconstructed/jepa_adapter.py'), ('official_tasks.py', 'd3/official_tasks.py'),
                                                    ('wm_costs.py', 'd3/wm_costs.py'), ('checkpoints.json', 'v06_reconstructed/checkpoints.json')]},
       'eval_config_sha256': {k: sha(REPO / f'configs/evals/simu_env_planning/pt/{d}/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml')
                              for k, d in [('jepa', 'jepa-wm'), ('dino', 'dino-wm')]},
       'models': {}}
pstd = np.asarray(st['proprio_std'], float)
for kind in ['jepa', 'dino']:
    ck = WM / f'runtime/checkpoints/{kind}_wm_pusht.pth.tar'
    pe = torch.load(ck, map_location='cpu', weights_only=False)['proprio_encoder']
    W = pe['module.patch_embed.weight'][:, :, 0].double().numpy(); b = pe['module.patch_embed.bias'].double().numpy()
    J = W / pstd[None]   # embedding change per physical unit of (x, y, vx, vy)
    out['models'][kind] = {'checkpoint_sha256': sha(ck), 'W': W.tolist(), 'b': b.tolist(), 'column_norms_normalized_units': np.linalg.norm(W, axis=0).tolist(),
                           'column_norms_physical_units': np.linalg.norm(J, axis=0).tolist(),
                           'velocity_to_position_ratio_physical': float(np.hypot(*np.linalg.norm(J, axis=0)[2:]) / np.hypot(*np.linalg.norm(J, axis=0)[:2])),
                           'singular_values': np.linalg.svd(W, compute_uv=False).tolist()}
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps({'stats': st, 'commits': [out['jepa_wms_commit'], out['dinov2_commit']],
                  'phys_ratio': {k: v['velocity_to_position_ratio_physical'] for k, v in out['models'].items()},
                  'sv': {k: [round(x, 4) for x in v['singular_values']] for k, v in out['models'].items()}}, indent=1))
