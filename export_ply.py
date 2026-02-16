"""Export PLY from a checkpoint file."""

import argparse
import torch
from gsplat import export_splats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint file")
    parser.add_argument("--output", required=True, help="Output PLY file path")
    args = parser.parse_args()

    data = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    splats = data["splats"]

    export_splats(
        means=splats["means"],
        scales=splats["scales"],
        quats=splats["quats"],
        opacities=splats["opacities"],
        sh0=splats["sh0"],
        shN=splats["shN"],
        format="ply",
        save_to=args.output,
    )
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
