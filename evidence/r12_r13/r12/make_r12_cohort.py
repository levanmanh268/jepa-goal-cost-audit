"""R12 fresh cohort (2026-10-28). Draws 32 PushT TRAIN-split trajectories not used by any earlier cohort, then derives
offset, env seed and CEM seed exactly as r8_collect.py does (keyed by default_rng([SEED, trajectory])).
No planning, no simulator outcome: identities only.
Usage: python3 -I make_r12_cohort.py <out_json>
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

WM = Path('/home/claude/wm'); D3 = WM / 'd3'; TRAIN = D3 / 'data_official/pusht_noise/train'
SEED = 42000000; N = 32
R8_EPS = [3635, 2466, 3108, 16642, 3665, 18150, 15360, 12440, 1941, 10673, 12946, 4828, 6331, 10136, 13906, 10710]
R10_EPS = [15720, 1021, 16600, 12404, 15364, 17531, 10308, 16094, 7913, 15537, 10157, 4859, 5725, 14537, 2288, 5586]


def main():
    sys.path.insert(0, str(D3)); import official_tasks as ot
    states, acts, lens = ot.load_val(str(TRAIN)); T = ot.FRAMESKIP * ot.GOAL_H + 1; n_ep = len(lens)
    # cross-check the earlier cohorts' own locked draws
    assert [int(x) for x in np.random.default_rng(38000000).choice(n_ep, 16, replace=False)] == R8_EPS
    used = set(R8_EPS) | set(R10_EPS)
    for p in [WM / 'r8/cohort/r6_index.json', WM / 'r10/cohort/r6_index.json']:
        used |= {int(x) for x in json.loads(p.read_text())['config']['episodes'].split(',')}
    cand = [int(x) for x in np.random.default_rng(SEED).choice(n_ep, 4 * N, replace=False)]
    eps = [i for i in cand if i not in used and lens[i] >= T][:N]
    assert len(eps) == N and len(set(eps)) == N and not (set(eps) & used)
    starts = []
    for k, i in enumerate(eps):
        rng = np.random.default_rng([SEED, i]); off = int(rng.integers(0, lens[i] - T + 1))
        es = int(rng.integers(0, 2 ** 31 - 1)); cs = int(rng.integers(0, 2 ** 31 - 1))
        starts.append({'k': k, 'val_episode': i, 'offset': off, 'env_seed': es, 'cem_seed': cs})
    data_sha = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(TRAIN.iterdir())}
    out = {'seed': SEED, 'n': N, 'split': 'train', 'excluded_episodes': sorted(used), 'episodes': eps, 'starts': starts,
           'train_sha256': data_sha, 'T': T}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1)); print('episodes', eps)


if __name__ == '__main__':
    main()
