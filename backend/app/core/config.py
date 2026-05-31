from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    STORAGE_DIR: Path = Path(__file__).resolve().parent.parent.parent / "storage"
    MODELS_DIR: Path = Path(__file__).resolve().parent.parent.parent / "storage" / "models"
    DEMO_DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent / "demo_data"

    # ── Training quality knobs (Phase 3) ──
    # 2026-04-28: bumped epoch / patience budgets to give the smaller lr range
    # (1e-5 .. 1e-4) enough iterations to converge — the previous 50/10 budget
    # plateaued early and produced bouncing val curves on demo-scale data.
    MAX_EPOCHS: int = 200
    MAX_HPO_EPOCHS: int = 30
    PATIENCE: int = 30
    HPO_PATIENCE: int = 6
    LR_SCHED_PATIENCE: int = 6
    OPTUNA_TRIALS: int = 150
    GRADIENT_CLIP: float = 1.0
    # PyTorch Lightning precision:
    #   "16-mixed"  → fp16 autocast (CUDA, fastest on recent GPUs)
    #   "bf16-mixed"→ bfloat16 autocast (H100/A100)
    #   "32-true"   → fp32 (portable)
    PRECISION: str = "16-mixed"
    DETERMINISTIC_SEED: int = 42


settings = Settings()
