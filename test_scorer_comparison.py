"""Compare VLM scorers on real dataset images.

Loads canvas images from the training dataset, extracts workspace views,
scores them with each requested scorer, and outputs a comparison report.
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from control.canvas_utils import ACTION_COLORS, SEPARATOR_WIDTH, FRAME_SIZE
from run_control import get_scorer

# Reverse color lookup: RGB tuple -> action int
COLOR_TO_ACTION = {v: k for k, v in ACTION_COLORS.items()}


def decode_action(canvas: np.ndarray) -> int:
    """Decode action from canvas separator color."""
    pixel = tuple(canvas[100, FRAME_SIZE[1] + SEPARATOR_WIDTH // 2])
    return COLOR_TO_ACTION.get(pixel, -1)


def load_canvas_sample(
    dataset_dir: str, n_samples: int, seed: int
) -> list[dict]:
    """Load a random sample of canvas images and extract workspace views."""
    dataset_path = Path(dataset_dir)
    canvas_files = sorted(dataset_path.glob("canvas_*.png"))
    if not canvas_files:
        print(f"Error: No canvas_*.png files found in {dataset_dir}")
        sys.exit(1)

    n_samples = min(n_samples, len(canvas_files))
    sampled = random.Random(seed).sample(canvas_files, n_samples)
    sampled.sort()

    samples = []
    for f in sampled:
        canvas = np.array(Image.open(f).convert("RGB"))
        pred_start = FRAME_SIZE[1] + SEPARATOR_WIDTH  # 224 + 32 = 256
        pred_end = pred_start + FRAME_SIZE[1]  # 256 + 224 = 480

        # Extract workspace views (top 224 rows = base camera)
        prediction_view = canvas[:FRAME_SIZE[0], pred_start:pred_end]
        context_view = canvas[:FRAME_SIZE[0], :FRAME_SIZE[1]]
        action = decode_action(canvas)

        samples.append({
            "filename": f.name,
            "prediction_view": prediction_view,
            "context_view": context_view,
            "action": action,
        })

    return samples


def load_curated_sample(
    dataset_dir: str, n_per_action: int = 5, seed: int = 42
) -> list[dict]:
    """Load a balanced sample with n_per_action from each action type.

    Samples evenly spaced across each action group to capture different
    arm positions. Reading only the separator pixel for action classification.
    """
    dataset_path = Path(dataset_dir)
    canvas_files = sorted(dataset_path.glob("canvas_*.png"))
    if not canvas_files:
        print(f"Error: No canvas_*.png files found in {dataset_dir}")
        sys.exit(1)

    # Classify by action (read only separator pixel for speed)
    by_action = {1: [], 2: [], 3: []}
    sep_col = FRAME_SIZE[1] + SEPARATOR_WIDTH // 2  # 240
    for f in canvas_files:
        img = Image.open(f)
        pixel = img.getpixel((sep_col, 100))[:3]
        action = COLOR_TO_ACTION.get(pixel, -1)
        if action in by_action:
            by_action[action].append(f)

    action_names = {1: "move+", 2: "move-", 3: "hold"}
    print(f"  Dataset actions: " + ", ".join(
        f"{action_names[a]}={len(files)}" for a, files in by_action.items()
    ))

    # Sample evenly spaced from each action group
    sampled_files = []
    for action_id in [3, 1, 2]:  # hold first, then moves
        pool = by_action[action_id]
        n = min(n_per_action, len(pool))
        indices = [int(i * len(pool) / n) for i in range(n)]
        sampled_files.extend(pool[i] for i in indices)

    # Load full images for sampled files
    samples = []
    pred_start = FRAME_SIZE[1] + SEPARATOR_WIDTH
    pred_end = pred_start + FRAME_SIZE[1]
    for f in sampled_files:
        canvas = np.array(Image.open(f).convert("RGB"))
        prediction_view = canvas[:FRAME_SIZE[0], pred_start:pred_end]
        context_view = canvas[:FRAME_SIZE[0], :FRAME_SIZE[1]]
        action = decode_action(canvas)
        samples.append({
            "filename": f.name,
            "prediction_view": prediction_view,
            "context_view": context_view,
            "action": action,
        })

    return samples


def format_prompt(task: str) -> str:
    """Wrap a task description into the standard VQA scoring prompt."""
    return (
        f"This image shows a robot's view. The task is: {task}. "
        "On a scale of 1 to 100, how well does this image show the task being achieved? "
        "Reply with only a number."
    )


def run_scorer_on_frames(
    scorer, frames: list[np.ndarray], task_prompt: str, verbose: bool = False
) -> dict:
    """Score all frames with a scorer, returning scores and timing."""
    # Warm up (first inference is slower due to CUDA kernel compilation)
    warmup_start = time.time()
    scorer.score_frames([frames[0]], task_prompt)
    warmup_s = time.time() - warmup_start

    # Score all frames
    scores = []
    total_start = time.time()
    for i, frame in enumerate(frames):
        t0 = time.time()
        s = scorer.score_frames([frame], task_prompt)[0]
        elapsed = time.time() - t0
        scores.append(s)
        if verbose:
            print(f"    Frame {i+1}/{len(frames)}: {s:.1f} ({elapsed:.2f}s)")
    total_s = time.time() - total_start

    parse_failures = sum(1 for s in scores if s == 0.0)

    return {
        "scores": scores,
        "warmup_s": warmup_s,
        "total_s": total_s,
        "per_frame_s": total_s / len(frames),
        "parse_failures": parse_failures,
    }


def compute_summary(all_results: dict[str, dict]) -> dict:
    """Compute per-scorer stats and cross-scorer agreement."""
    summary = {"per_scorer": {}, "agreement": {}}

    for name, result in all_results.items():
        scores = np.array(result["scores"])
        summary["per_scorer"][name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
            "total_s": result["total_s"],
            "per_frame_s": result["per_frame_s"],
            "warmup_s": result["warmup_s"],
            "parse_failures": result["parse_failures"],
            "n_frames": len(result["scores"]),
        }

    # Cross-scorer agreement (if 2+ scorers)
    scorer_names = list(all_results.keys())
    if len(scorer_names) >= 2:
        a = np.array(all_results[scorer_names[0]]["scores"])
        b = np.array(all_results[scorer_names[1]]["scores"])

        # Pearson correlation
        if np.std(a) > 0 and np.std(b) > 0:
            pearson = float(np.corrcoef(a, b)[0, 1])
        else:
            pearson = None

        # Spearman (rank correlation via numpy)
        if np.std(a) > 0 and np.std(b) > 0:
            rank_a = np.argsort(np.argsort(a)).astype(float)
            rank_b = np.argsort(np.argsort(b)).astype(float)
            spearman = float(np.corrcoef(rank_a, rank_b)[0, 1])
        else:
            spearman = None

        summary["agreement"] = {
            "scorers": scorer_names[:2],
            "pearson": pearson,
            "spearman": spearman,
            "mean_abs_diff": float(np.mean(np.abs(a - b))),
        }

    return summary


def print_report(summary: dict, task_prompt: str) -> None:
    """Print formatted comparison report."""
    print("\n=== Results ===\n")
    header = f"{'Scorer':<14} | {'Mean':>5} | {'Std':>5} | {'Min':>4} | {'Max':>5} | {'Med':>5} | {'Time/frame':>10} | {'Failures':>8}"
    print(header)
    print("-" * len(header))

    for name, stats in summary["per_scorer"].items():
        print(
            f"{name:<14} | {stats['mean']:5.1f} | {stats['std']:5.2f} | {stats['min']:4.1f} | "
            f"{stats['max']:5.1f} | {stats['median']:5.1f} | {stats['per_frame_s']:8.2f}s | "
            f"{stats['parse_failures']:>3}/{stats['n_frames']}"
        )

    if summary.get("agreement"):
        ag = summary["agreement"]
        print(f"\n=== Cross-Scorer Agreement ({ag['scorers'][0]} vs {ag['scorers'][1]}) ===\n")
        if ag["pearson"] is not None:
            print(f"  Pearson correlation:  {ag['pearson']:.3f}")
        if ag["spearman"] is not None:
            print(f"  Spearman correlation: {ag['spearman']:.3f}")
        print(f"  Mean absolute diff:  {ag['mean_abs_diff']:.2f}")


def img_to_base64(img: np.ndarray) -> str:
    """Encode an RGB numpy image as a base64 PNG data URI."""
    import base64
    import cv2
    _, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/png;base64,{b64}"


def score_color(score: float) -> str:
    """Return a CSS color for a score value (1-100 scale)."""
    if score <= 20:
        return "#e74c3c"
    elif score <= 50:
        return "#f39c12"
    elif score <= 70:
        return "#3498db"
    else:
        return "#27ae60"


ACTION_NAMES = {0: "buffer", 1: "move+", 2: "move-", 3: "hold", -1: "unknown"}


def generate_html_report(
    samples: list[dict],
    all_results: dict[str, dict],
    summary: dict,
    config: dict,
    output_path: Path,
) -> None:
    """Generate an HTML report with embedded images and scores."""
    scorer_names = list(all_results.keys())

    # Build scorer summary rows
    summary_rows = ""
    for name in scorer_names:
        s = summary["per_scorer"][name]
        summary_rows += f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td>{s['mean']:.1f}</td>
            <td>{s['std']:.2f}</td>
            <td>{s['min']:.1f}</td>
            <td>{s['max']:.1f}</td>
            <td>{s['median']:.1f}</td>
            <td>{s['per_frame_s']:.2f}s</td>
            <td>{s['parse_failures']}/{s['n_frames']}</td>
        </tr>"""

    # Build agreement section
    agreement_html = ""
    if summary.get("agreement") and summary["agreement"]:
        ag = summary["agreement"]
        agreement_html = f"""
    <h2>Cross-Scorer Agreement</h2>
    <p><strong>{ag['scorers'][0]}</strong> vs <strong>{ag['scorers'][1]}</strong></p>
    <table>
        <tr><td>Pearson correlation</td><td><strong>{ag['pearson']:.3f}</strong></td></tr>
        <tr><td>Spearman correlation</td><td><strong>{f"{ag['spearman']:.3f}" if ag['spearman'] is not None else 'N/A'}</strong></td></tr>
        <tr><td>Mean absolute difference</td><td><strong>{ag['mean_abs_diff']:.2f}</strong></td></tr>
    </table>""" if ag.get("pearson") is not None else ""

    # Build per-image cards
    cards_html = ""
    for i, sample in enumerate(samples):
        pred_b64 = img_to_base64(sample["prediction_view"])
        ctx_b64 = img_to_base64(sample["context_view"])
        action = ACTION_NAMES.get(sample["action"], "?")

        score_cells = ""
        for name in scorer_names:
            score = all_results[name]["scores"][i]
            color = score_color(score)
            score_cells += f'<div class="score" style="border-left: 4px solid {color}"><span class="scorer-name">{name}</span><span class="score-val">{score:.1f}</span></div>'

        cards_html += f"""
    <div class="card">
        <div class="card-header">
            <span class="filename">{sample['filename']}</span>
            <span class="action action-{action.replace('+','plus').replace('-','minus')}">{action}</span>
        </div>
        <div class="card-body">
            <div class="images">
                <div class="img-container">
                    <img src="{ctx_b64}" alt="context">
                    <div class="img-label">Context</div>
                </div>
                <div class="img-container">
                    <img src="{pred_b64}" alt="prediction">
                    <div class="img-label">Prediction (scored)</div>
                </div>
            </div>
            <div class="scores">{score_cells}</div>
        </div>
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VLM Scorer Comparison Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
    h1 {{ margin-bottom: 5px; }}
    .meta {{ color: #666; font-size: 14px; margin-bottom: 20px; }}
    .meta code {{ background: #e8e8e8; padding: 2px 6px; border-radius: 3px; font-size: 13px; }}
    table {{ border-collapse: collapse; margin: 10px 0 20px; }}
    th, td {{ padding: 8px 14px; border: 1px solid #ddd; text-align: left; }}
    th {{ background: #f0f0f0; }}
    .card {{ background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); margin-bottom: 16px; overflow: hidden; }}
    .card-header {{ padding: 10px 16px; background: #fafafa; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }}
    .filename {{ font-weight: 600; font-size: 14px; }}
    .action {{ font-size: 12px; padding: 2px 8px; border-radius: 12px; font-weight: 600; }}
    .action-moveplus {{ background: #d4edda; color: #155724; }}
    .action-moveminus {{ background: #cce5ff; color: #004085; }}
    .action-hold {{ background: #f8d7da; color: #721c24; }}
    .card-body {{ padding: 16px; display: flex; gap: 20px; align-items: flex-start; }}
    .images {{ display: flex; gap: 12px; }}
    .img-container {{ text-align: center; }}
    .img-container img {{ width: 224px; height: 224px; image-rendering: pixelated; border: 1px solid #ddd; border-radius: 4px; }}
    .img-label {{ font-size: 11px; color: #888; margin-top: 4px; }}
    .scores {{ display: flex; flex-direction: column; gap: 8px; min-width: 160px; }}
    .score {{ display: flex; justify-content: space-between; align-items: center; padding: 6px 12px; background: #fafafa; border-radius: 4px; }}
    .scorer-name {{ font-size: 13px; color: #666; }}
    .score-val {{ font-size: 20px; font-weight: 700; }}
    h2 {{ margin-top: 30px; }}
</style>
</head>
<body>
    <h1>VLM Scorer Comparison</h1>
    <div class="meta">
        <p>Dataset: <code>{config['dataset_dir']}</code></p>
        <p>Samples: {config['n_samples']} (seed={config['seed']})</p>
        <p>Prompt: <code>{config['task_prompt']}</code></p>
        <p>Scorers: {', '.join(scorer_names)}</p>
    </div>

    <h2>Summary</h2>
    <table>
        <tr><th>Scorer</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Median</th><th>Time/frame</th><th>Failures</th></tr>
        {summary_rows}
    </table>

    {agreement_html}

    <h2>Per-Image Results</h2>
    {cards_html}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def save_results(
    samples: list[dict],
    all_results: dict[str, dict],
    summary: dict,
    config: dict,
    output_dir: Path,
    save_images: bool,
) -> None:
    """Save results to JSON and optionally save frame images."""
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-image results
    per_image = []
    for i, sample in enumerate(samples):
        entry = {
            "filename": sample["filename"],
            "action": sample["action"],
            "scores": {},
        }
        for scorer_name, result in all_results.items():
            entry["scores"][scorer_name] = result["scores"][i]
        per_image.append(entry)

    output = {
        "config": config,
        "per_image": per_image,
        "timing": {
            name: {
                "warmup_s": r["warmup_s"],
                "total_s": r["total_s"],
                "per_frame_s": r["per_frame_s"],
            }
            for name, r in all_results.items()
        },
        "summary": summary,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    if save_images:
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        for sample in samples:
            stem = Path(sample["filename"]).stem
            cv2.imwrite(
                str(frames_dir / f"{stem}_prediction.png"),
                cv2.cvtColor(sample["prediction_view"], cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(frames_dir / f"{stem}_context.png"),
                cv2.cvtColor(sample["context_view"], cv2.COLOR_RGB2BGR),
            )
        print(f"Frame images saved to: {frames_dir}")

    # Always generate HTML report
    html_path = output_dir / "report.html"
    generate_html_report(samples, all_results, summary, config, html_path)
    print(f"HTML report: {html_path}")


def main():
    p = argparse.ArgumentParser(description="Compare VLM scorers on dataset images")
    p.add_argument("--dataset-dir", type=str,
                   default="../canvas-world-model/local/datasets/hold-1500-combined")
    p.add_argument("--scorers", type=str, default="qwen,moondream",
                   help="Comma-separated scorer names")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", type=str, default="center the keyboard")
    p.add_argument("--task-prompt", type=str, default="",
                   help="Full VLM prompt (overrides --task)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="local/scorer_comparison")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--curated", action="store_true",
                   help="Use curated balanced sample (hold vs move)")
    p.add_argument("--n-per-action", type=int, default=5,
                   help="Samples per action type when using --curated")
    args = p.parse_args()

    task_prompt = args.task_prompt or format_prompt(args.task)
    scorer_names = [s.strip() for s in args.scorers.split(",")]

    print(f"=== VLM Scorer Comparison ===")
    print(f"Dataset: {args.dataset_dir}")
    if args.curated:
        print(f"Curated: {args.n_per_action} per action (seed={args.seed})")
    else:
        print(f"Samples: {args.n_samples} (seed={args.seed})")
    print(f"Prompt: {task_prompt}")
    print(f"Scorers: {', '.join(scorer_names)}")
    print(f"Device: {args.device}")

    # Load dataset
    if args.curated:
        print(f"\nLoading curated sample ({args.n_per_action} per action)...")
    else:
        print(f"\nLoading {args.n_samples} canvas images...")
    if args.curated:
        samples = load_curated_sample(args.dataset_dir, args.n_per_action, args.seed)
    else:
        samples = load_canvas_sample(args.dataset_dir, args.n_samples, args.seed)
    frames = [s["prediction_view"] for s in samples]
    print(f"  Loaded {len(samples)} images, frame shape: {frames[0].shape}")

    # Action distribution
    action_names = {0: "buffer", 1: "move+", 2: "move-", 3: "hold", -1: "unknown"}
    action_counts = {}
    for s in samples:
        name = action_names.get(s["action"], "?")
        action_counts[name] = action_counts.get(name, 0) + 1
    print(f"  Actions: {action_counts}")

    # Run each scorer
    all_results = {}
    for scorer_name in scorer_names:
        print(f"\n--- Loading {scorer_name} ---")
        try:
            scorer = get_scorer(scorer_name)
        except ValueError as e:
            print(f"  Skipping: {e}")
            continue

        try:
            t0 = time.time()
            scorer.load(args.device)
            load_s = time.time() - t0
            print(f"  Loaded in {load_s:.1f}s")
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        print(f"\n--- Scoring with {scorer.name()} ({len(frames)} frames) ---")
        try:
            result = run_scorer_on_frames(scorer, frames, task_prompt, verbose=args.verbose)
            all_results[scorer.name()] = result
            print(f"  Warmup: {result['warmup_s']:.2f}s")
            print(f"  Scoring: {len(frames)} frames in {result['total_s']:.1f}s ({result['per_frame_s']:.2f}s/frame)")
        except Exception as e:
            print(f"  Scoring failed: {e}")

        # Free VRAM
        import gc
        import torch
        del scorer
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass  # CUDA context may be corrupted if scorer crashed

    if not all_results:
        print("\nNo scorers ran successfully.")
        sys.exit(1)

    # Summarize and report
    summary = compute_summary(all_results)
    print_report(summary, task_prompt)

    # Save
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        "dataset_dir": args.dataset_dir,
        "n_samples": len(samples),
        "seed": args.seed,
        "task_prompt": task_prompt,
        "scorers": scorer_names,
        "device": args.device,
    }
    save_results(samples, all_results, summary, config, run_dir, args.save_images)


if __name__ == "__main__":
    main()
