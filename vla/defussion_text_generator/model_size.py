"""Print the model size without training anything.

    python vla/defussion_text_generator/model_size.py

Safe to run in a second terminal while run_cv.py is going — it touches
nothing on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config          # noqa: E402
from model import VLAModel, parameter_report   # noqa: E402

IMG_DIM = 576      # MobileNetV3-Small pooled features
TEL_DIM = 16       # see Config.telemetry_keys


def main():
    cfg = Config()
    print(f"config: hidden={cfg.hidden_dim} blocks={cfg.n_blocks} "
          f"img_proj={cfg.img_proj_dim} tel_proj={cfg.tel_proj_dim} "
          f"pca_dim<={cfg.pca_dim}\n")

    for latent_dim in sorted({min(cfg.pca_dim, 7), cfg.pca_dim}):
        rep = parameter_report(VLAModel(IMG_DIM, TEL_DIM, latent_dim, cfg))
        print(f"latent_dim = {latent_dim}")
        print(f"  total          : {rep['total_params']:,} parameters")
        print(f"  size           : {rep['size_fp32_mb']} MB fp32 / "
              f"{rep['size_fp16_mb']} MB fp16")
        for name, n in rep["by_module"].items():
            print(f"    {name:<10s} {n:>10,}")
        print(f"  hoisted once per frame : {rep['flow_hoisted_params']:>10,} params")
        print(f"  re-run per Euler step  : {rep['flow_per_step_params']:>10,} params"
              f"  (~{2 * rep['flow_per_step_params'] / 1e6:.2f} MFLOP)")
        print()

    print("Frozen MobileNetV3-Small (~2.5 M params) is NOT counted: it never")
    print("trains, and its features are extracted once and cached.")


if __name__ == "__main__":
    main()
