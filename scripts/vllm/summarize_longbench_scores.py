import argparse
import json
from pathlib import Path
from collections import defaultdict


def _average_metrics(metrics):
    if not metrics:
        return None
    first_value = next(iter(metrics.values()))
    if isinstance(first_value, dict):
        avg = defaultdict(list)
        for _, value in metrics.items():
            for key, item in value.items():
                avg[key].append(item)
        return {k: round(sum(v) / len(v), 2) for k, v in avg.items()}
    return round(sum(metrics.values()) / len(metrics), 2)


def main():
    parser = argparse.ArgumentParser("Summarize LongBench scores from task json files.")
    parser.add_argument("--result_dir", required=True, help="Directory that stores LongBench task json files.")
    parser.add_argument(
        "--save_path",
        default=None,
        help="Output summary path. Default: <result_dir>/score_summary.json",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Result dir not found: {result_dir}")

    ordered_tasks = []
    config_path = result_dir / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config_obj = json.load(f)
        if isinstance(config_obj.get("tasks"), list):
            ordered_tasks = config_obj["tasks"]

    task_scores = {}
    existing_task_files = {
        p.stem: p
        for p in result_dir.glob("*.json")
        if p.name not in {"config.json", "score_summary.json"}
    }

    # Prefer configured task order from config.json.
    for task in ordered_tasks:
        task_file = existing_task_files.pop(task, None)
        if task_file is None:
            continue
        with task_file.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                continue
            task_scores[task] = json.loads(first_line)

    # Append any leftover files in lexical order.
    for task_name in sorted(existing_task_files.keys()):
        task_file = existing_task_files[task_name]
        with task_file.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                continue
            task_scores[task_name] = json.loads(first_line)

    summary = {
        "result_dir": str(result_dir),
        "num_tasks": len(task_scores),
        "task_scores": task_scores,
        "avg": _average_metrics(task_scores),
    }

    save_path = Path(args.save_path) if args.save_path else result_dir / "score_summary.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
