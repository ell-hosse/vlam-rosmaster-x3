"""Settings for the velocity-head VLA.

Same skeleton as ../defussion_text_generator/config.py. The differences that
matter:

  * text_field  = "action_text"      (nested under `action` in the annotations)
  * telemetry   = a window of past COMMANDS, not the 16-d sensor vector
  * the head predicts a COMMAND CHUNK, not waypoints; waypoints come from the
    frozen stage-2 calculator in ../trajectory_predictor.
"""

from dataclasses import dataclass
from pathlib import Path


# repo_root/vla/velocity_head/config.py  ->  repo_root
REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SIBLING_CACHE = HERE.parent / "defussion_text_generator" / "cache"


STAGE2_ROOT = REPO_ROOT / "vla" / "trajectory_predictor"


def find_calculator_dir(root: Path = STAGE2_ROOT) -> Path:
    """Locate the exported stage-2 fold models.

    `--export <name>` lets you call the output folder anything, so rather than
    hard-coding one name this looks for the directory that actually holds
    fold_*.json. Falls back to <root>/models so the error message downstream
    still names a sensible path when nothing has been exported yet.
    """
    if not root.is_dir():
        return root / "models"
    candidates = [root] + sorted(p for p in root.iterdir() if p.is_dir())
    for path in candidates:
        if any(path.glob("fold_*.json")):
            return path
    return root / "models"


@dataclass
class Config:
    # ---------------- paths ----------------
    dataset_dir: Path = REPO_ROOT / "dataset"
    out_dir: Path = HERE / "runs"
    # Reuse the main model's frozen-CNN cache when it exists: the features are
    # keyed by (scenario, frame) and are byte-identical, so all three variants
    # see the same images and none of them re-runs MobileNet.
    cache_dir: Path = SIBLING_CACHE if SIBLING_CACHE.exists() else HERE / "cache"
    # Where the exported stage-2 fold models live. Left as None it is
    # auto-discovered: any directory under vla/trajectory_predictor/ that
    # contains fold_*.json, whatever it happens to be called.
    calculator_dir: Path = None

    # ---------------- data ----------------
    # 'action_text' has 8 unique strings in this dataset (vs 9 for
    # trajectory_text, 240 for caption_detailed). Anything nested is reached
    # with a dot: "action.action_text" and "action_text" both work.
    text_field: str = "action_text"

    # Input window: the H commands BEFORE the current frame, u[t-H .. t-1].
    hist: int = 6                 # 6 frames @ 2 Hz = 3.0 s of history

    # Output chunk: the F commands STARTING AT the current frame, u[t .. t+F-1].
    # This alignment is not a choice -- it is what generates the recorded
    # waypoints. Fitting the plant with the chunk shifted one step later
    # doubles its residual (0.037 m -> 0.072 m). See README.
    horizon: int = 6              # 6 commands = the 3.0 s / 6-waypoint horizon

    # ---------------- frozen encoders ----------------
    cnn_name: str = "mobilenet_v3_small"
    image_size: int = 224
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---------------- text latent space ----------------
    # PCA refit inside every fold on TRAIN texts only; capped at
    # (n_unique_train_texts - 1), so with action_text it lands at ~6-7.
    pca_dim: int = 8

    # ---------------- model ----------------
    img_proj_dim: int = 128
    tel_proj_dim: int = 32
    hidden_dim: int = 256
    n_blocks: int = 3

    # ---------------- flow matching ----------------
    flow_steps: int = 4
    step_sweep: tuple = (1, 2, 4, 8)
    n_samples_eval: int = 5       # for min-ADE / min-FDE over K samples

    # ---------------- training ----------------
    epochs: int = 300
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    w_flow: float = 1.0
    w_vel: float = 1.0            # loss on the 6 linear-velocity outputs
    w_ang: float = 1.0            # loss on the 6 angular-velocity outputs
    w_action: float = 0.5         # auxiliary: the action class of the current step
    p_sampled_z: float = 0.5

    seed: int = 0
    device: str = "cpu"

    action_classes: tuple = (
        "FORWARD", "RIGHT_TURN", "LEFT_TURN", "STOP", "REVERSE_LEFT",
    )

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.cache_dir = Path(self.cache_dir)
        self.calculator_dir = (
            Path(self.calculator_dir) if self.calculator_dir is not None
            else find_calculator_dir()
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def n_actions(self) -> int:
        return len(self.action_classes)

    @property
    def tel_dim(self) -> int:
        """Flattened input window: H frames x (v, w)."""
        return self.hist * 2

    @property
    def vel_dim(self) -> int:
        """Flattened output chunk: F steps x (v, w)."""
        return self.horizon * 2
