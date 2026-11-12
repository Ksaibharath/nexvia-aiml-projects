"""Cross-platform pipeline & server launcher (works on Windows PowerShell/CMD, macOS, and Linux)."""
import argparse
import subprocess
import sys


STEPS = {
    "download": [[sys.executable, "src/data/download.py"]],
    "split": [[sys.executable, "src/data/clean_split.py"]],
    "features": [[sys.executable, "src/spark/build_features.py"]],
    "retrieval": [[sys.executable, "-m", "src.retrieval.train"]],
    "index": [[sys.executable, "src/index/build_faiss.py"]],
    "ranking": [
        [sys.executable, "-m", "src.ranking.build_ranking_data"],
        [sys.executable, "-m", "src.ranking.train_all"],
    ],
    "evaluate": [
        [sys.executable, "-m", "src.evaluation.eval_retrieval"],
        [sys.executable, "-m", "src.evaluation.eval_ranking"],
    ],
    "ab_sim": [[sys.executable, "-m", "src.evaluation.ab_sim"]],
    "bundle": [[sys.executable, "-m", "src.serving.bundle_export"]],
    "test": [[sys.executable, "-m", "pytest", "-v"]],
    "serve": [[sys.executable, "-m", "uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8080"]],
    "ui": [[sys.executable, "-m", "streamlit", "run", "src/serving/ui.py", "--server.port", "8501"]],
}

ALL_ORDER = ["download", "split", "features", "retrieval", "index", "ranking", "evaluate", "ab_sim", "bundle", "test"]


def run_step(name: str) -> None:
    print(f"\n>>> Running step: {name}")
    for cmd in STEPS[name]:
        print("   $", " ".join(cmd))
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Cross-platform runner for Two-Stage Recommendation Engine")
    parser.add_argument(
        "step",
        nargs="?",
        default="serve",
        choices=[*STEPS.keys(), "all"],
        help="Pipeline step to run (default: 'serve' launches FastAPI + Web App at http://localhost:8080)",
    )
    args = parser.parse_args()

    if args.step == "all":
        for s in ALL_ORDER:
            run_step(s)
    else:
        run_step(args.step)


if __name__ == "__main__":
    main()
