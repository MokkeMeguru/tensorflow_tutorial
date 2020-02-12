import datetime
from enum import Enum, unique
from pathlib import Path
from typing import Dict
import yaml
from attrdict import AttrDict


@unique
class ImageValType(str, Enum):
    RGB_UNSIGNED = 'unsigned'
    RGB_SIGNED = 'signed'
    RGB_ABSNORMALISED = 'absnormalized'
    RGB_NORMALISED = 'normalized'


@unique
class Optimizer(str, Enum):
    ADAM = 'adam'
    SGD = 'sgd'


@unique
class Activation(str, Enum):
    RELU = "relu"


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
