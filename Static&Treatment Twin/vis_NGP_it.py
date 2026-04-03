import os
import subprocess
import argparse

exp_list = [
    '01_1', '01_2', '04_1', '04_2', '05_1',
    '05_2', '10_1', '10_2', '12_1',
    '12_2','15_1', '15_2','17_1', '17_2','18_1', '18_2'
]

# default dataset root where per-experiment folders live (adjust if needed)
BASE_DATA_ROOT = '/data/shenchengkang/sck/thyroid/data_wuhan'

# ckpt parent for heart models
CKPT_PARENT = '/data/shenchengkang/sck/thyroid/ckpts/heart'

gpu_list = ['0', '1', '2', '3']


def spawn_workers(exp_names, base_dir=BASE_DATA_ROOT):
    processes = []
    for i, exp_name in enumerate(exp_names):
        gpu_id = gpu_list[i % len(gpu_list)]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_id

        cmd = [
            'python', os.path.abspath(__file__),
            '--worker', exp_name,
            '--data_root', base_dir
        ]

        print(f"\n=== Launching visualizer for {exp_name} on GPU {gpu_id} ===")
        print(' '.join(cmd))
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    for p in processes:
        p.wait()


def worker(exp_name, data_root):
    # perform the same visualization steps as vis_NGP.py but use explicit ckpt path
    import torch
    import numpy as np
    import os
    import nrrd
    from einops import rearrange

    #from models.networks_siren import Siren
    from models.networks_siren import Finer
    from utils import load_ckpt

    # build full dataset root and ckpt path
    dataset_root = os.path.join(data_root, exp_name)
    ckpt_path = os.path.join(CKPT_PARENT, exp_name, 'epoch=19_slim.ckpt')

    print(f"Worker for {exp_name}: dataset_root={dataset_root}, ckpt={ckpt_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Finer().to(device)
    # load checkpoint if exists
    if os.path.exists(ckpt_path):
        load_ckpt(model, ckpt_path)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"Warning: checkpoint not found: {ckpt_path}")

    @torch.no_grad()
    def nerf_func(points):
        result = model(points)
        return result

    # produce a 160x160x160 grid as in vis_NGP.py
    z,y,x = torch.meshgrid(torch.arange(160, dtype=torch.float32),
                        torch.arange(160, dtype=torch.float32),
                        torch.arange(160, dtype=torch.float32),indexing=None)
    dirs_x = x
    dirs_y = y
    dirs_z = z
    rays_dir = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1).to(device)
    rays_dir = rays_dir/160 - 0.5
    rays_dir = rays_dir.reshape(-1, 3)

    res = nerf_func(rays_dir)
    res = rearrange(res.cpu().numpy(), '(h w d) c -> h w d c', h=160,w=160)
    res = res.squeeze(3)
    res = (res * 255).astype(np.uint8)

    out_dir = os.path.join(os.path.dirname(__file__), 'out_3d')
    os.makedirs(out_dir, exist_ok=True)
    filename = exp_name
    np.save(os.path.join(out_dir, filename + '_3d.npy'), res)

    header = {
        'type':'float',
        'dimension':3,
        'spacing':'left-posterior-superior',
        'size':res.shape,
        'encoding':'gzip'
    }
    nrrd.write(os.path.join(out_dir, filename + '_3d.nrrd'), res, header)
    print(f"Saved outputs to {out_dir} for {exp_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=str, default=None,
                        help='Run single worker for given exp_name')
    parser.add_argument('--data_root', type=str, default=BASE_DATA_ROOT,
                        help='Base data root containing experiment folders')
    args = parser.parse_args()

    if args.worker:
        worker(args.worker, args.data_root)
    else:
        # default: spawn workers for exp_list
        spawn_workers(exp_list, args.data_root)


if __name__ == '__main__':
    main()
