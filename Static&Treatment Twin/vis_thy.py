import os
import argparse
import torch
import numpy as np
import nrrd
from einops import rearrange

from models.networks_siren import Finer
from utils import load_ckpt


def visualize(ckpt_path, output_dir='./out_3d'):
    """
    Visualize a single checkpoint.

    Args:
        ckpt_path: Path to the checkpoint file (.ckpt)
        output_dir: Directory to save outputs
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Finer().to(device)
    load_ckpt(model, ckpt_path)
    print(f"Loaded checkpoint: {ckpt_path}")

    @torch.no_grad()
    def nerf_func(points):
        return model(points)

    # Produce a 160x160x160 grid
    z, y, x = torch.meshgrid(
        torch.arange(160, dtype=torch.float32),
        torch.arange(160, dtype=torch.float32),
        torch.arange(160, dtype=torch.float32),
        indexing=None
    )
    rays_dir = torch.stack([x, y, z], dim=-1).to(device)
    rays_dir = rays_dir / 160 - 0.5
    rays_dir = rays_dir.reshape(-1, 3)

    res = nerf_func(rays_dir)
    res = rearrange(res.cpu().numpy(), '(h w d) c -> h w d c', h=160, w=160)
    res = res.squeeze(3)
    res = (res * 255).astype(np.uint8)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    # Use checkpoint name as filename
    filename = os.path.splitext(os.path.basename(ckpt_path))[0]

    np.save(os.path.join(output_dir, filename + '_3d.npy'), res)

    header = {
        'type': 'float',
        'dimension': 3,
        'spacing': 'left-posterior-superior',
        'size': res.shape,
        'encoding': 'gzip'
    }
    nrrd.write(os.path.join(output_dir, filename + '_3d.nrrd'), res, header)

    print(f"Saved outputs to {output_dir}/{filename}_3d.*")


def main():
    parser = argparse.ArgumentParser(description='Visualize a single NeRF checkpoint')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint file (.ckpt)')
    parser.add_argument('--out_dir', type=str, default='./out_3d',
                        help='Output directory (default: ./out_3d)')
    args = parser.parse_args()

    visualize(args.ckpt, args.out_dir)


if __name__ == '__main__':
    main()
