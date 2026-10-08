"""Command-line interface for full and stage-wise TaoFlowForge inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import InferenceConfig, Stage0Config, Stage1Config, Stage2Config
from .pipeline import TaoFlowForgePipeline
from .postprocess import export_obj


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", type=Path, required=True, help="Input RGB/RGBA image")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument(
        "--dino-checkpoint", type=Path, default=None,
        help="HuggingFace DINOv3 pretrained directory (required when stage checkpoints lack DINO weights)",
    )
    parser.add_argument("--no-progress", action="store_true")


def _add_stage0_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage0-checkpoint", type=Path, required=True)
    parser.add_argument("--stage0-num-steps", type=int, default=50)
    parser.add_argument("--stage0-cfg-scale", type=float, default=7.5)
    parser.add_argument("--stage0-t-shift", type=float, default=2.718)
    parser.add_argument("--stage0-threshold", type=float, default=0.5)
    parser.add_argument("--stage0-max-cells", type=int, default=15000)
    parser.add_argument("--compile-stage0", action="store_true")


def _add_stage1_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-num-steps", type=int, default=30)
    parser.add_argument("--stage1-cfg-scale", type=float, default=7.0)
    parser.add_argument("--stage1-t-shift", type=float, default=1.0)
    parser.add_argument("--stage1-threshold", type=float, default=0.5)
    parser.add_argument("--stage1-max-vertices", type=int, default=20000)


def _add_stage2_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-num-steps", type=int, default=50)
    parser.add_argument("--stage2-cfg-scale", type=float, default=3.0)
    parser.add_argument("--stage2-t-shift", type=float, default=1.0)
    parser.add_argument("--edge-threshold", type=float, default=0.5)


def _stage0_config(args: argparse.Namespace) -> Stage0Config:
    return Stage0Config(
        checkpoint=args.stage0_checkpoint,
        num_steps=args.stage0_num_steps,
        cfg_scale=args.stage0_cfg_scale,
        t_shift=args.stage0_t_shift,
        occupancy_threshold=args.stage0_threshold,
        max_cells=args.stage0_max_cells,
        compile_model=args.compile_stage0,
    )


def _stage1_config(args: argparse.Namespace) -> Stage1Config:
    return Stage1Config(
        checkpoint=args.stage1_checkpoint,
        num_steps=args.stage1_num_steps,
        cfg_scale=args.stage1_cfg_scale,
        t_shift=args.stage1_t_shift,
        occupancy_threshold=args.stage1_threshold,
        max_vertices=args.stage1_max_vertices,
    )


def _stage2_config(args: argparse.Namespace) -> Stage2Config:
    return Stage2Config(
        checkpoint=args.stage2_checkpoint,
        num_steps=args.stage2_num_steps,
        cfg_scale=args.stage2_cfg_scale,
        t_shift=args.stage2_t_shift,
        edge_threshold=args.edge_threshold,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TaoFlowForge image-to-mesh inference")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    full = commands.add_parser("run", help="Run Stage 0 -> Stage 1 -> Stage 2")
    _add_runtime_arguments(full)
    _add_stage0_arguments(full)
    _add_stage1_arguments(full)
    _add_stage2_arguments(full)
    full.add_argument("--output-dir", type=Path, required=True)
    full.add_argument("--seed", type=int, default=42)
    full.add_argument("--offload", choices=("auto", "on", "off"), default="auto")
    full.add_argument("--resume-from", type=Path)
    full.add_argument("--no-fill-holes", action="store_true")

    stage0 = commands.add_parser("stage0", help="Generate a Stage 0 artifact")
    _add_runtime_arguments(stage0)
    _add_stage0_arguments(stage0)
    stage0.add_argument("--output-artifact", type=Path, required=True)
    stage0.add_argument("--seed", type=int, default=42)

    stage1 = commands.add_parser("stage1", help="Continue a Stage 0 artifact")
    _add_runtime_arguments(stage1)
    _add_stage1_arguments(stage1)
    stage1.add_argument("--input-artifact", type=Path, required=True)
    stage1.add_argument("--output-artifact", type=Path, required=True)

    stage2 = commands.add_parser("stage2", help="Continue a Stage 1 artifact")
    _add_runtime_arguments(stage2)
    _add_stage2_arguments(stage2)
    stage2.add_argument("--input-artifact", type=Path, required=True)
    stage2.add_argument("--output-artifact", type=Path, required=True)
    stage2.add_argument("--output-mesh", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    show_progress = not args.no_progress

    if args.command == "run":
        config = InferenceConfig(
            stage0=_stage0_config(args),
            stage1=_stage1_config(args),
            stage2=_stage2_config(args),
            dino_checkpoint=args.dino_checkpoint,
            seed=args.seed,
            offload=args.offload,
            fill_holes=not args.no_fill_holes,
            image_size=args.image_size,
            output_dir=args.output_dir,
        )
        with TaoFlowForgePipeline.from_config(config, device=args.device) as pipeline:
            for update in pipeline.iter_run(
                args.image,
                resume_from=args.resume_from,
                show_progress=show_progress,
            ):
                print(update.message, flush=True)
                if update.result is not None:
                    print(update.result.glb_path or update.result.final_mesh_path)
        return 0

    if args.command == "stage0":
        with TaoFlowForgePipeline(
            stage0=_stage0_config(args),
            dino_checkpoint=args.dino_checkpoint,
            seed=args.seed,
            image_size=args.image_size,
            device=args.device,
        ) as pipeline:
            coordinates = pipeline.run_stage0(
                args.image,
                args.output_artifact,
                show_progress=show_progress,
            )
        print(f"Saved {len(coordinates)} coordinates to {args.output_artifact}")
        return 0

    if args.command == "stage1":
        with TaoFlowForgePipeline(
            stage1=_stage1_config(args),
            dino_checkpoint=args.dino_checkpoint,
            image_size=args.image_size,
            device=args.device,
        ) as pipeline:
            vertices = pipeline.run_stage1(
                args.image,
                args.input_artifact,
                args.output_artifact,
                show_progress=show_progress,
            )
        print(f"Saved {len(vertices)} vertices to {args.output_artifact}")
        return 0

    with TaoFlowForgePipeline(
        stage2=_stage2_config(args),
        dino_checkpoint=args.dino_checkpoint,
        image_size=args.image_size,
        device=args.device,
    ) as pipeline:
        mesh = pipeline.run_stage2(
            args.image,
            args.input_artifact,
            args.output_artifact,
            show_progress=show_progress,
        )
        export_obj(mesh, args.output_mesh)
    print(f"Saved {len(mesh.faces)} faces to {args.output_mesh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
