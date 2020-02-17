from pathlib import Path

from model import Activation, ImageValType, Optimizer

mnist_classification_args = {
    "paths": {
        "model_path": Path(
            "./mnist_cnn/model/"),
        "log_path": Path(
            "./mnist_cnn/log/")
    },
    "hyper_parameters": {
        "learning_rate": 1e-3,
        "step": 5000,
        "optimizer": Optimizer.ADAM,
        "batch_size": 256
    },
    "model_params": {
        "conv_filters": [32, 64],
        "conv_kernels": [[5, 5], [5, 5]],
        "activation": Activation.RELU,
        "pooling_size": [[2, 2], [2, 2]],
        "mid_dense_units": 1024,
        "dropout_rate": 0.4
    },
    "data_params": {
        "image_size": [28, 28, 1],
        "is_flattened": True,
        "has_channel": False,
        "num_class": 10,
        "val_type": ImageValType.RGB_ABSNORMALISED
    }
}
