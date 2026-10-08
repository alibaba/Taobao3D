"""Minimal Python API example for one image."""

from pathlib import Path

from taoflowforge import InferenceConfig, Stage0Config, Stage1Config, Stage2Config
from taoflowforge.pipeline import TaoFlowForgePipeline

WEIGHTS = Path("weights")
config = InferenceConfig(
    stage0=Stage0Config(
        checkpoint=WEIGHTS / "stage0.pt",
        compile_model=False,
    ),
    stage1=Stage1Config(checkpoint=WEIGHTS / "stage1.pt"),
    stage2=Stage2Config(checkpoint=WEIGHTS / "stage2.pt"),
    # Optional: provide a HuggingFace DINOv3 directory when stage checkpoints
    # do not embed DINO weights.  Omit if checkpoints already include them.
    # dino_checkpoint=Path("path/to/dinov3-large"),
    seed=42,
    output_dir=Path("output/example"),
)

with TaoFlowForgePipeline.from_config(config) as pipeline:
    result = pipeline.run("examples/bookshelf_desk.png")

print(result.glb_path)
