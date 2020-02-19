import datetime
from pathlib import Path
from typing import Dict
import yaml
from attrdict import AttrDict


def export_params(args,
                  path_prefix=datetime.datetime.now()
                  .isoformat(timespec='minutes')):
    if not isinstance(args["paths"]["log_path"], Path):
        raise KeyError("export path is unknown")
    else:
        path = args["paths"]["log_path"] / path_prefix
        path.mkdir(parents=True, exist_ok=True)
        if (path / "params.yaml").exists():
            raise Exception(
                'Overwrite Exception: Model is already saved in this folder')
        else:
            with open(path / "params.yaml", "w", encoding="utf-8") as f:
                yaml.dump(args, f)
            return path


def load_params(path: Path):
    if not path.exists():
        raise Exception('Model not found Exception')
    with open(Path, 'r', encoding='utf-8') as f:
        params = yaml.load(f, Loader=yaml.Loader)
    return params


def parse_params(parameters: Dict):
    return AttrDict(parameters)
