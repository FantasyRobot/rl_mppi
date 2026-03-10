import argparse
import json
import os

import matplotlib.pyplot as plt


def moving_average(values, window):
    if window <= 1 or len(values) < 2:
        return list(values)
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        segment = values[start : index + 1]
        smoothed.append(sum(segment) / len(segment))
    return smoothed


def load_metrics(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No metrics found in {path}")
    return rows


def extract_series(rows):
    epochs = [row["epoch"] for row in rows]
    train_loss = [row.get("train_loss") for row in rows]
    val_loss = [row.get("val_loss") for row in rows]
    eval_epochs = [row["epoch"] for row in rows if "eval" in row]
    eval_success = [row["eval"].get("success_rate") for row in rows if "eval" in row]
    eval_deadline = [row["eval"].get("avg_deadline_satisfaction") for row in rows if "eval" in row]
    eval_collisions = [row["eval"].get("avg_collisions") for row in rows if "eval" in row]
    eval_collision_stop = [row["eval"].get("collision_stop_rate") for row in rows if "eval" in row]
    best_epoch = None
    best_score = None
    for row in rows:
        if row.get("best_updated"):
            best_epoch = row["epoch"]
            best_score = row.get("selection_score")
    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "eval_epochs": eval_epochs,
        "eval_success": eval_success,
        "eval_deadline": eval_deadline,
        "eval_collisions": eval_collisions,
        "eval_collision_stop": eval_collision_stop,
        "best_epoch": best_epoch,
        "best_score": best_score,
    }


def _plot_series(axis, x_values, y_values, *, color, label, smooth_window, show_raw_points, point_label=None):
    if not x_values or not y_values:
        return
    if show_raw_points:
        axis.plot(x_values, y_values, color=color, alpha=0.25, linewidth=1.0)
        axis.scatter(x_values, y_values, color=color, alpha=0.75, s=18, label=point_label or f"{label} raw")
    smooth_values = moving_average(y_values, smooth_window)
    axis.plot(x_values, smooth_values, color=color, linewidth=2.0, label=label)


def _mark_best_epoch(axis, best_epoch):
    if best_epoch is None:
        return
    axis.axvline(best_epoch, color="0.25", linestyle="--", linewidth=1.0, alpha=0.7)


def plot_metrics(series, title, output_path, smooth_window=3, show_raw_points=True):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    best_epoch = series.get("best_epoch")

    axes[0, 0].plot(series["epochs"], series["train_loss"], label="train loss", color="tab:blue")
    axes[0, 0].plot(series["epochs"], series["val_loss"], label="val loss", color="tab:orange")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].grid(True, alpha=0.25)
    _mark_best_epoch(axes[0, 0], best_epoch)
    axes[0, 0].legend()

    _plot_series(
        axes[0, 1],
        series["eval_epochs"],
        series["eval_success"],
        color="tab:green",
        label="success trend",
        smooth_window=smooth_window,
        show_raw_points=show_raw_points,
        point_label="success raw",
    )
    axes[0, 1].set_title("Eval Success Rate")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].grid(True, alpha=0.25)
    _mark_best_epoch(axes[0, 1], best_epoch)
    axes[0, 1].legend()

    _plot_series(
        axes[1, 0],
        series["eval_epochs"],
        series["eval_deadline"],
        color="tab:red",
        label="deadline trend",
        smooth_window=smooth_window,
        show_raw_points=show_raw_points,
        point_label="deadline raw",
    )
    axes[1, 0].set_title("Eval Deadline Satisfaction")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(True, alpha=0.25)
    _mark_best_epoch(axes[1, 0], best_epoch)
    axes[1, 0].legend()

    _plot_series(
        axes[1, 1],
        series["eval_epochs"],
        series["eval_collisions"],
        color="tab:purple",
        label="collision trend",
        smooth_window=smooth_window,
        show_raw_points=show_raw_points,
        point_label="collision raw",
    )
    axes[1, 1].set_title("Eval Collisions")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].grid(True, alpha=0.25)
    _mark_best_epoch(axes[1, 1], best_epoch)

    collision_stop_axis = axes[1, 1].twinx()
    _plot_series(
        collision_stop_axis,
        series["eval_epochs"],
        series["eval_collision_stop"],
        color="tab:brown",
        label="collision-stop trend",
        smooth_window=smooth_window,
        show_raw_points=False,
    )
    collision_stop_axis.set_ylabel("collision-stop rate")
    collision_stop_axis.set_ylim(0.0, 1.05)
    _mark_best_epoch(collision_stop_axis, best_epoch)

    handles, labels = axes[1, 1].get_legend_handles_labels()
    stop_handles, stop_labels = collision_stop_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles + stop_handles, labels + stop_labels, loc="upper left")

    if best_epoch is not None:
        suffix = f" | best epoch={best_epoch}"
        if series.get("best_score") is not None:
            suffix += f" score={series['best_score']:.3f}"
        title = f"{title}{suffix}"

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from metrics.jsonl")
    parser.add_argument("--metrics_path", type=str, required=True)
    parser.add_argument("--plot_path", type=str, required=True)
    parser.add_argument("--title", type=str, default="Training Metrics")
    parser.add_argument("--smooth_window", type=int, default=3)
    parser.add_argument("--hide_raw_points", action="store_true")
    args = parser.parse_args()

    rows = load_metrics(args.metrics_path)
    series = extract_series(rows)
    plot_metrics(
        series,
        args.title,
        args.plot_path,
        smooth_window=max(1, int(args.smooth_window)),
        show_raw_points=not args.hide_raw_points,
    )
    print(f"[SAVE] plot={args.plot_path}")


if __name__ == "__main__":
    main()