from __future__ import annotations
import hashlib

import pathlib
import sys

import numpy as np
import scipy.sparse as sparse

from . import wrapper

__all__ = ["Model", "train", "train_with_bin", "load", "make_bin"]


class Model:
    """A model obtained from train or load. Can be saved and used to predict."""

    def __init__(self, n, m, k, W, normalization):
        self.n = n
        self.m = m
        self.k = k
        self.W = W
        self.normalization = normalization

    def save(self, path: str):
        """Save model to path."""
        wrapper.save_model(self.n, self.m, self.k, self.W, self.normalization, path)

    def predict(self, x: sparse.csr_matrix, fields: np.ndarray) -> np.ndarray:
        """Predict the probabilities of x.

        Args:
            x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.
            fields (np.ndarray): A non-negative integral array with dimension number of features.

        Returns:
            np.ndarray: An array with dimension number of instances.
        """
        if x.shape[1] != fields.shape[0]:
            raise ValueError("input matrix shapes do not match")

        fields = fields.astype(np.int32, copy=False)
        return wrapper.predict(
            self.n,
            self.m,
            self.k,
            self.W,
            self.normalization,
            x.shape[0],
            x.shape[1],
            fields,
            x.data,
            x.indices,
            x.indptr,
        )


def train(
    train_x: sparse.csr_matrix,
    train_y: np.ndarray,
    fields: np.ndarray,
    tmpdir: str,
    options: dict[str, int | float | bool] = {},
    valid_x: sparse.csr_matrix = None,
    valid_y: np.ndarray = None,
) -> Model:
    """Trains a model on given data.
    fields is an array used to indicate the field of each feature column in x.
    Refer to ffm_parameter in ffm.h from libffm for the meaning of options.
    Validation data must be provided if auto-stop is used.

    Args:
        train_x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.
        train_y (np.ndarray): A 0/1 array with dimension number of instances.
        fields (np.ndarray): A non-negative integral array with dimension number of features.
        tmpdir (str): A temporary directory used to write internal data structures for training.
        options (dict[str, int  |  float  |  bool], optional): A dictionary of options. Unspecified options will default to the same as libffm. Defaults to {}.
        valid_x (sparse.csr_matrix, optional):  A matrix with dimension number of instances * number of features.. Defaults to None.
        valid_y (np.ndarray, optional): A 0/1 matrix with dimension number of instances.. Defaults to None.

    Returns:
        Model: Trained model. Can be saved and used to predict.
    """
    _check_data(train_x, train_y, fields)

    if options.get("auto_stop", False):
        if valid_x is None or valid_y is None:
            raise ValueError("no validation set provided for auto-stop")
        _check_data(valid_x, valid_y, fields)

    train_path, cached = make_bin(tmpdir, train_x, train_y, fields)
    if cached:
        print(
            f'Using training data cache "{train_path}"',
            file=sys.stderr,
        )

    if options.get("auto_stop", False):
        valid_path, cached = make_bin(tmpdir, valid_x, valid_y, fields)
        if cached:
            print(
                f'Using validation data cache "{valid_path}"',
                file=sys.stderr,
            )
    else:
        valid_path = None

    return train_with_bin(train_path, options, valid_path)


def train_with_bin(
    train_path: str | pathlib.Path,
    options: dict[str, int | float | bool] = {},
    valid_path: pathlib.Path = None,
) -> Model:
    """Trains a model on data created from make_bin.
    Refer to ffm_parameter in ffm.h from libffm for the meaning of options.
    Validation data must be provided if auto-stop is used.

    Args:
        train_path (str | pathlib.Path): Path to training data created from make_bin.
        options (dict[str, int  |  float  |  bool], optional): A dictionary of options. Unspecified options will default to the same as libffm. Defaults to {}.
        valid_path (str | pathlib.Path, optional): Path to validation data created from make_bin.

    Returns:
        Model: Trained model. Can be saved and used to predict.
    """
    default_options = {
        "eta": 0.2,
        "lambda": 0.00002,
        "nr_iters": 15,
        "k": 4,
        "normalization": True,
        "auto_stop": False,
    }

    options = {**default_options, **options}

    if not pathlib.Path(train_path).is_file():
        raise ValueError("training file does not exist")

    if options["auto_stop"]:
        if valid_path is None:
            raise ValueError("no validation set provided for auto-stop")
        if not pathlib.Path(valid_path).is_file():
            raise ValueError("validation file does not exist")
    else:
        valid_path = ""

    model = wrapper.train_on_disk(
        f"{train_path}",
        f"{valid_path}",
        options["eta"],
        options["lambda"],
        options["nr_iters"],
        options["k"],
        options["normalization"],
        options["auto_stop"],
    )

    return Model(*model)


def load(path: str) -> Model:
    """Load model from path."""
    if not pathlib.Path(path).exists():
        raise ValueError("file does not exist")

    model = wrapper.load_model(path)
    return Model(*model)


def _fingerprint(x: sparse.csr_matrix, y: np.ndarray, fields: np.ndarray) -> str:
    mid = x.shape[0] // 2
    fingerprint = str(
        (
            *x.shape,
            *y.shape,
            fields.size,
            x.nnz,
            x[:5].data,
            x[mid : mid + 5].data,
            x[-5:].data,
            y[:20],
            y[mid : mid + 20],
            y[-20:],
            fields[:20],
            fields[fields.size // 2 : fields.size // 2 + 20],
            fields[-20:],
        )
    )
    h = hashlib.sha256()
    h.update(fingerprint.encode("utf-8"))

    return h.hexdigest()[:32]


def make_bin(
    dir: str | pathlib.Path, x: sparse.csr_matrix, y: np.ndarray, fields: np.ndarray
) -> tuple[str, bool]:
    """Create binary files for training. The path names are created from a fingerprint
    of the input data. If another file already exists with the same name, it is assumed
    they are the same data.

    Args:
        dir (str | pathlib.Path): The destination directory.
        x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.
        y (np.ndarray): A 0/1 array with dimension number of instances.
        fields (np.ndarray): A non-negative integral array with dimension number of features.

    Returns:
        tuple[str, bool]: The resulting file path and whether it already exists prior to calling make_bin.
    """
    _check_data(x, y, fields)

    # x is converted to float within the C++ wrapper because x is relatively large
    # y and fields are converted to float here because they are small
    # The C++ wrapper will give an error on wrong array types, so either way, this won't cause hidden bugs
    fields = fields.astype(np.int32, copy=False)
    y = (y > 0).astype(np.float32, copy=False) * 2 - 1

    path = f"{dir}/{_fingerprint(x, y, fields)}"
    if pathlib.Path(path).exists():
        return path, True

    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    wrapper.arr2bin(
        x.shape[0],
        x.shape[1],
        y,
        fields,
        x.data,
        x.indices,
        x.indptr,
        path,
    )

    return path, False


def _check_data(x: sparse.csr_matrix, y: np.ndarray, fields: np.ndarray):
    if y.ndim != 1 or fields.ndim != 1:
        raise ValueError("invalid input matrix shapes")
    if x.shape[0] != y.shape[0] or x.shape[1] != fields.shape[0]:
        raise ValueError("input matrix shapes do not match")
