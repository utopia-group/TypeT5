from .utils import proj_root, read_file, Path, tqdm
from typing import Optional
import json
import shutil


def get_config_dict() -> dict:
    if (path := proj_root() / "config" / "SPOT.json").exists():
        return json.loads(read_file(path))
    else:
        return {}


def get_config(key: str) -> Optional[str]:
    return get_config_dict().get(key)


def get_dataroot() -> Path:
    if (v := get_config("data_root")) is None:
        return proj_root()
    else:
        return Path(v)


def get_dataset_dir(dataname: str) -> Path:
    if (v := get_config("datasets_root")) is None:
        return get_dataroot() / "datasets" / dataname
    else:
        return Path(v) / dataname


def get_model_dir(trained=True) -> Path:
    post = "trained" if trained else "training"
    return get_dataroot() / "models" / post


def get_eval_dir(dataname: str, modelname: str) -> Path:
    return get_dataroot() / "evaluations" / dataname / modelname


def mk_dataset_from_this_project(name="SPOT-src"):
    dest = get_dataset_dir(name) / "repos" / "test" / "SPOT"
    if dest.exists():
        print("Deleting old dataset at: ", dest)
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    root = proj_root()
    shutil.copytree(root / "src", dest / "src")
    shutil.copytree(root / "tests", dest / "tests")
    return dest
