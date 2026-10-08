"""Interactive Gradio interface for the TaoFlowForge three-stage pipeline."""

from __future__ import annotations

import argparse
import html
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import trimesh

from taoflowforge import InferenceConfig, Stage0Config, Stage1Config, Stage2Config
from taoflowforge.artifacts import load_stage_artifact
from taoflowforge.pipeline import TaoFlowForgePipeline

_EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
_EXAMPLE_IMAGES = [
    str(_EXAMPLES_DIR / "bookshelf_desk.png"),
    str(_EXAMPLES_DIR / "easel.png"),
    str(_EXAMPLES_DIR / "mannequin.png"),
    str(_EXAMPLES_DIR / "person01.png"),
    str(_EXAMPLES_DIR / "person02.png"),
    str(_EXAMPLES_DIR / "person03.png"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the TaoFlowForge Gradio demo")
    parser.add_argument("--stage0-checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dino-checkpoint", type=Path, default=None,
        help="HuggingFace DINOv3 pretrained directory (required when stage checkpoints lack DINO weights)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--offload", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser


def build_pipeline(args: argparse.Namespace) -> TaoFlowForgePipeline:
    config = InferenceConfig(
        stage0=Stage0Config(
            checkpoint=args.stage0_checkpoint,
            compile_model=False,
        ),
        stage1=Stage1Config(checkpoint=args.stage1_checkpoint),
        stage2=Stage2Config(checkpoint=args.stage2_checkpoint),
        dino_checkpoint=args.dino_checkpoint,
        offload=args.offload,
        fill_holes=True,
    )
    pipeline = TaoFlowForgePipeline.from_config(config, device=args.device)
    pipeline.load_models()
    return pipeline


_MAX_WIREFRAME_EDGES = 80_000

_APP_CSS = """
#taoflowforge-hero {
    background: linear-gradient(125deg, #fb923c 0%, #fdba74 52%, #fff7ed 100%);
    border-radius: 22px;
    box-shadow: 0 18px 48px rgba(15, 23, 42, 0.12);
    color: #7c2d12;
    margin-bottom: 18px;
    overflow: hidden;
    padding: 28px 34px;
}
#taoflowforge-hero h1 {
    color: #7c2d12;
    font-size: clamp(2rem, 4vw, 3.35rem);
    letter-spacing: -0.045em;
    line-height: 1;
    margin: 8px 0 12px;
}
#taoflowforge-hero p {
    color: #9a3412;
    font-size: 1.02rem;
    margin: 0;
}
.hero-badge {
    background: rgba(255, 255, 255, 0.62);
    border: 1px solid rgba(124, 45, 18, 0.2);
    color: #7c2d12;
    border-radius: 999px;
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    padding: 6px 11px;
    text-transform: uppercase;
}
.app-panel {
    background: var(--block-background-fill);
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
    padding: 8px;
}
#generate-button {
    min-height: 48px;
    font-size: 1rem;
    font-weight: 700;
}
#status-panel {
    border-radius: 14px;
    margin-bottom: 18px;
    overflow: hidden;
    position: sticky;
    top: 8px;
    z-index: 20;
}
.status-card {
    align-items: center;
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 6px solid #64748b;
    border-radius: 12px;
    display: flex;
    font-size: 1rem;
    font-weight: 600;
    gap: 12px;
    min-height: 54px;
    padding: 12px 16px;
}
.status-card.running {
    background: linear-gradient(90deg, #fff7ed 0%, #ffffff 100%);
    border-color: #fdba74;
    border-left-color: #f97316;
    box-shadow: 0 8px 24px rgba(249, 115, 22, 0.18);
    color: #9a3412;
}
.status-card.success { border-left-color: #16a34a; }
.status-card.error { border-left-color: #dc2626; }
.status-dot {
    background: #64748b;
    border-radius: 50%;
    flex: 0 0 auto;
    height: 11px;
    width: 11px;
}
.status-card.running .status-dot {
    animation: status-pulse 1.4s ease-out infinite;
    background: #f97316;
}
.status-card.success .status-dot { background: #16a34a; }
.status-card.error .status-dot { background: #dc2626; }
.status-label {
    border: 1px solid currentColor;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    padding: 3px 8px;
}
@keyframes status-pulse {
    0% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0.45); }
    70% { box-shadow: 0 0 0 9px rgba(249, 115, 22, 0); }
    100% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0); }
}
.stage-summary {
    color: var(--body-text-color-subdued);
    font-size: 0.88rem;
    line-height: 1.55;
}
"""


def _rotate_vertices(vertices: np.ndarray) -> np.ndarray:
    """Match the camera-facing orientation used by the original demo."""
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices with shape (N,3), got {vertices.shape}")
    return np.column_stack((-vertices[:, 0], vertices[:, 2], vertices[:, 1]))


def _scene_layout() -> dict:
    axis = {
        "showbackground": True,
        "backgroundcolor": "#f8fafc",
        "gridcolor": "#e2e8f0",
        "zerolinecolor": "#94a3b8",
        "showspikes": False,
    }
    return {
        "xaxis": {**axis, "title": "X"},
        "yaxis": {**axis, "title": "Y"},
        "zaxis": {**axis, "title": "Z"},
        "aspectmode": "data",
        "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.05}},
    }


def _empty_figure(title: str, message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": "#64748b", "size": 16},
    )
    figure.update_layout(
        title={"text": title, "x": 0.02},
        height=560,
        margin={"l": 0, "r": 0, "t": 54, "b": 0},
        template="plotly_white",
    )
    return figure


def create_pointcloud_plot(
    vertices: np.ndarray,
    *,
    title: str,
    point_label: str,
    colorscale: str = "Viridis",
) -> go.Figure:
    """Create the Stage 0/1 interactive point-cloud visualization."""
    vertices = _rotate_vertices(vertices)
    if len(vertices) == 0:
        return _empty_figure(title, f"No {point_label.lower()} were generated")
    marker_size = 3.2 if len(vertices) <= 5_000 else 2.0
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode="markers",
                marker={
                    "size": marker_size,
                    "color": vertices[:, 2],
                    "colorscale": colorscale,
                    "opacity": 0.88,
                    "showscale": False,
                },
                name=point_label,
                hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title={"text": f"{title} · {len(vertices):,} points", "x": 0.02},
        scene=_scene_layout(),
        height=560,
        margin={"l": 0, "r": 0, "t": 54, "b": 0},
        template="plotly_white",
        showlegend=False,
    )
    return figure


def create_mesh_wireframe_plot(
    mesh: trimesh.Trimesh,
    *,
    title: str = "Stage 2 · Triangle Mesh",
) -> go.Figure:
    """Create an interactive surface plot with a topology wireframe overlay."""
    source_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(source_vertices) == 0:
        return _empty_figure(title, "No mesh vertices were generated")
    if len(faces) == 0:
        return create_pointcloud_plot(
            source_vertices,
            title=f"{title} (no faces)",
            point_label="Vertices",
            colorscale="Oranges",
        )

    vertices = _rotate_vertices(source_vertices)
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
        axis=0,
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    visible_edges = edges
    if len(edges) > _MAX_WIREFRAME_EDGES:
        indices = np.linspace(0, len(edges) - 1, _MAX_WIREFRAME_EDGES, dtype=np.int64)
        visible_edges = edges[indices]

    edge_points = np.full((len(visible_edges) * 3, 3), np.nan, dtype=np.float32)
    edge_points[0::3] = vertices[visible_edges[:, 0]]
    edge_points[1::3] = vertices[visible_edges[:, 1]]
    surface = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=vertices[:, 2],
        colorscale="Oranges",
        flatshading=True,
        opacity=0.82,
        showscale=False,
        lighting={"ambient": 0.55, "diffuse": 0.75, "specular": 0.25},
        name="Surface",
        hoverinfo="skip",
    )
    wireframe = go.Scatter3d(
        x=edge_points[:, 0],
        y=edge_points[:, 1],
        z=edge_points[:, 2],
        mode="lines",
        line={"color": "rgba(15, 23, 42, 0.72)", "width": 1},
        name="Topology",
        hoverinfo="skip",
    )
    figure = go.Figure(data=[surface, wireframe])
    figure.update_layout(
        title={
            "text": (
                f"{title} · {len(vertices):,} vertices · "
                f"{len(faces):,} faces"
            ),
            "x": 0.02,
        },
        scene=_scene_layout(),
        height=560,
        margin={"l": 0, "r": 0, "t": 54, "b": 0},
        template="plotly_white",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
    )
    return figure


def _load_stage_outputs(output_dir: Path, stage: int) -> dict[str, np.ndarray]:
    saved_stage, _, outputs, _ = load_stage_artifact(
        output_dir / f"stage{stage}.npz"
    )
    if saved_stage != stage:
        raise ValueError(f"Expected Stage {stage} artifact, got Stage {saved_stage}")
    return outputs


def _status_card(message: str, state: str = "running") -> str:
    safe_message = html.escape(message)
    label = {
        "idle": "READY",
        "running": "RUNNING",
        "success": "COMPLETE",
        "error": "ERROR",
    }.get(state, state.upper())
    return (
        f'<div class="status-card {state}" role="status" aria-live="polite">'
        f'<span class="status-dot"></span>'
        f'<span class="status-label">{label}</span>'
        f'<span>{safe_message}</span></div>'
    )


def build_app(pipeline: TaoFlowForgePipeline) -> gr.Blocks:
    def generate(image, seed):
        s0_figure = None
        s1_figure = None
        s2_figure = None
        mesh_download = None
        if image is None:
            yield (
                _status_card("Upload an image first.", "error"),
                gr.update(selected="stage0"),
                s0_figure,
                s1_figure,
                s2_figure,
                mesh_download,
            )
            return

        output_dir = Path(tempfile.mkdtemp(prefix="taoflowforge-"))
        yield (
            _status_card("Preparing input and running Stage 0..."),
            gr.update(selected="stage0"),
            s0_figure,
            s1_figure,
            s2_figure,
            mesh_download,
        )
        try:
            for update in pipeline.iter_run(
                image,
                output_dir=output_dir,
                seed=int(seed),
                show_progress=False,
            ):
                selected_tab = "stage2"
                status_state = "running"
                status_message = update.message
                if update.stage == 0:
                    outputs = _load_stage_outputs(output_dir, 0)
                    coordinates = np.asarray(outputs["coordinates"], dtype=np.float32)
                    centers = (coordinates + 0.5) / 64.0 * 2.0 - 1.0
                    s0_figure = create_pointcloud_plot(
                        centers,
                        title="Stage 0 · Occupied Cell Centers",
                        point_label="Occupied cells",
                        colorscale="Plasma",
                    )
                    selected_tab = "stage0"
                    status_message += " · Stage 1 is running..."
                elif update.stage == 1:
                    outputs = _load_stage_outputs(output_dir, 1)
                    s1_figure = create_pointcloud_plot(
                        outputs["vertices"],
                        title="Stage 1 · Refined Vertices",
                        point_label="Predicted vertices",
                    )
                    selected_tab = "stage1"
                    status_message += " · Stage 2 is running..."
                elif update.stage == 2:
                    outputs = _load_stage_outputs(output_dir, 2)
                    raw_mesh = trimesh.Trimesh(
                        vertices=outputs["vertices"],
                        faces=outputs["faces"],
                        process=False,
                    )
                    s2_figure = create_mesh_wireframe_plot(
                        raw_mesh,
                        title="Stage 2 · Raw Triangle Mesh",
                    )
                    status_message += " · Finalizing mesh..."
                elif update.result is not None:
                    s2_figure = create_mesh_wireframe_plot(
                        update.result.mesh,
                        title="Final · Repaired Triangle Mesh",
                    )
                    mesh_download = str(update.result.final_mesh_path)
                    status_message = (
                        f"Pipeline complete · Results saved in {update.result.output_dir}"
                    )
                    status_state = "success"

                yield (
                    _status_card(status_message, status_state),
                    gr.update(selected=selected_tab),
                    s0_figure,
                    s1_figure,
                    s2_figure,
                    mesh_download,
                )
        except Exception as error:
            yield (
                _status_card(f"{type(error).__name__}: {error}", "error"),
                gr.update(),
                s0_figure,
                s1_figure,
                s2_figure,
                mesh_download,
            )

    orange = gr.themes.Color(
        c50="#fff7ed",
        c100="#ffedd5",
        c200="#fed7aa",
        c300="#fdba74",
        c400="#fb923c",
        c500="#f97316",
        c600="#ea580c",
        c700="#c2410c",
        c800="#9a3412",
        c900="#7c2d12",
        c950="#431407",
        name="taoflowforge-orange",
    )
    theme = gr.themes.Soft(
        primary_hue=orange,
        secondary_hue=orange,
        neutral_hue="slate",
    )
    with gr.Blocks(
        title="TaoFlowForge · Three-Stage 3D Generation",
        theme=theme,
        css=_APP_CSS,
        fill_width=True,
    ) as demo:
        gr.HTML(
            '<section id="taoflowforge-hero">'
            '<span class="hero-badge">Image to Triangle Mesh</span>'
            '<h1>TaoFlowForge</h1>'
            '<p>Progressive 3D generation through occupancy, vertices, and topology.</p>'
            "</section>"
        )
        status = gr.HTML(
            _status_card("Ready. Upload an image to begin.", "idle"),
            elem_id="status-panel",
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes="app-panel"):
                image = gr.Image(
                    type="pil",
                    label="Input image",
                    height=340,
                )
                seed = gr.Number(
                    value=42,
                    precision=0,
                    label="Random seed",
                    info="Use the same seed to reproduce a result.",
                )
                generate_button = gr.Button(
                    "Generate 3D Mesh",
                    variant="primary",
                    size="lg",
                    elem_id="generate-button",
                )
                gr.Examples(
                    examples=_EXAMPLE_IMAGES,
                    inputs=image,
                    label="Example inputs",
                    examples_per_page=7,
                )
                gr.Markdown(
                    "**Three-stage workflow**  \n"
                    "`S0` 64³ occupancy → `S1` 512³ vertices → `S2` triangle topology",
                    elem_classes="stage-summary",
                )

            with gr.Column(scale=2, elem_classes="app-panel"):
                with gr.Tabs(selected="stage0") as result_tabs:
                    with gr.Tab("S0 · Occupancy", id="stage0"):
                        s0_plot = gr.Plot(
                            label="Stage 0 occupied cell centers",
                            show_label=False,
                        )
                    with gr.Tab("S1 · Vertices", id="stage1"):
                        s1_plot = gr.Plot(
                            label="Stage 1 predicted vertices",
                            show_label=False,
                        )
                    with gr.Tab("S2 · Topology", id="stage2"):
                        s2_plot = gr.Plot(
                            label="Stage 2 triangle mesh",
                            show_label=False,
                        )
                    with gr.Tab("Download", id="download"):
                        gr.Markdown(
                            "### Final mesh\n"
                            "The OBJ retains all generated vertices and includes the "
                            "post-processed topology and normals."
                        )
                        download = gr.File(label="Download OBJ", interactive=False)

        generate_button.click(
            generate,
            inputs=[image, seed],
            outputs=[status, result_tabs, s0_plot, s1_plot, s2_plot, download],
        )
    return demo


def main() -> None:
    args = build_parser().parse_args()
    pipeline = build_pipeline(args)
    app = build_app(pipeline)
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
