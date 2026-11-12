"""Publish newly trained SavedModels to the next Triton version directory and bump Redis model_version."""
import pathlib
import shutil

from src.serving.cache import Cache

MODELS = ("user_tower", "ranker", "ranker_wd")


def publish_models() -> int:
    repo = pathlib.Path("triton/model_repository")
    next_ver = 1
    for m in MODELS:
        m_dir = repo / m
        if not m_dir.exists():
            continue
        existing_vers = sorted([int(p.name) for p in m_dir.iterdir() if p.is_dir() and p.name.isdigit()])
        if not existing_vers:
            continue
        cur_max = existing_vers[-1]
        next_ver = max(next_ver, cur_max + 1)
        src_sm = m_dir / str(cur_max) / "model.savedmodel"
        dst_sm = m_dir / str(cur_max + 1) / "model.savedmodel"
        if src_sm.exists() and not dst_sm.exists():
            dst_sm.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_sm, dst_sm)

    metrics_src = pathlib.Path("artifacts/metrics.json")
    metrics_prod = pathlib.Path("artifacts/metrics_prod.json")
    if metrics_src.exists():
        shutil.copy2(metrics_src, metrics_prod)

    cache = Cache()
    cache.set_version(next_ver)
    print(f"Published models to Triton version {next_ver} and updated model_version={next_ver}")
    return next_ver


if __name__ == "__main__":
    publish_models()
