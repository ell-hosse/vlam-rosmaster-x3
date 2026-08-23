"""All tunable settings in one place."""

from dataclasses import dataclass, field
from pathlib import Path


# repo_root/vla/defussion_text_generator/config.py  ->  repo_root
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # ---------------- paths ----------------
    dataset_dir: Path = REPO_ROOT / "dataset"
    out_dir: Path = Path(__file__).resolve().parent / "runs"
    cache_dir: Path = Path(__file__).resolve().parent / "cache"

    # ---------------- data ----------------
    # Which annotation field to embed. 'trajectory_text' has 9 unique strings in
    # this dataset; 'caption_detailed' has ~240 and is a much richer target.
    text_field: str = "trajectory_text"
    n_waypoints: int = 4          # first N of the 6 stored waypoints (2 s @ 2 Hz)
    drop_first_waypoint: bool = True   # waypoint[0] is ~the current pose: trivial

    telemetry_keys: tuple = (
        "odom_linear_velocity",           # 3
        "odom_angular_velocity",          # 3  (steering rate)
        "imu_linear_acceleration",        # 3
        "imu_attitude",                   # 3  (roll, pitch, yaw)
        "wheel_speed_encoder_per_sec",    # 4
    )                                     # -> 16 dims

    # ---------------- frozen encoders ----------------
    cnn_name: str = "mobilenet_v3_small"        # torchvision, ImageNet weights
    image_size: int = 224
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---------------- text latent space ----------------
    # PCA is refit inside every fold on TRAIN TEXTS ONLY (no leakage).
    # Capped at (n_unique_train_texts - 1) automatically.
    pca_dim: int = 8

    # ---------------- model ----------------
    img_proj_dim: int = 128
    tel_proj_dim: int = 32
    hidden_dim: int = 256
    n_blocks: int = 3

    # ---------------- flow matching ----------------
    # Rectified flow: straight path from noise to data, so few Euler steps suffice.
    flow_steps: int = 4
    step_sweep: tuple = (1, 2, 4, 8)   # reported at eval time
    n_samples_eval: int = 5            # for min-ADE / min-FDE over K samples

    # ---------------- training ----------------
    epochs: int = 300
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    w_flow: float = 1.0
    w_traj: float = 1.0
    w_action: float = 1.0
    # Fraction of training steps where the trajectory/action head is fed a
    # *sampled* z instead of the true one, so it is not brittle at inference.
    p_sampled_z: float = 0.5

    seed: int = 0
    device: str = "cpu"

    action_classes: tuple = (
        "FORWARD", "RIGHT_TURN", "LEFT_TURN", "STOP", "REVERSE_LEFT",
    )

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.cache_dir = Path(self.cache_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def n_actions(self) -> int:
        return len(self.action_classes)

    @property
    def traj_dim(self) -> int:
        return self.n_waypoints * 2
