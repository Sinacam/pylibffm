from __future__ import annotations
import hashlib

import pathlib
import sys
import uuid

import numpy as np
import scipy.sparse as sparse

from . import wrapper

__all__ = ["Model", "train", "load"]


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
    default_options = {
        "eta": 0.2,
        "lambda": 0.00002,
        "nr_iters": 15,
        "k": 4,
        "normalization": True,
        "auto_stop": False,
    }

    options = {**default_options, **options}

    if train_y.ndim != 1 or fields.ndim != 1:
        raise ValueError("invalid input matrix shapes")
    if train_x.shape[0] != train_y.shape[0] or train_x.shape[1] != fields.shape[0]:
        raise ValueError("input matrix shapes do not match")

    if options["auto_stop"]:
        if valid_x is None or valid_y is None:
            raise ValueError("no validation set provided for auto-stop")
        if valid_y.ndim != 1 or fields.ndim != 1:
            raise ValueError("invalid input matrix shapes")
        if valid_x.shape[0] != valid_y.shape[0] or valid_x.shape[1] != fields.shape[0]:
            raise ValueError("input matrix shapes do not match")

    # x is converted to float within the C++ wrapper because x is relatively large
    # y and fields are converted to float here because they are small
    # The C++ wrapper will give an error on wrong array types, so this won't cause hidden bugs
    fields = fields.astype(np.int32, copy=False)
    train_y = (train_y > 0).astype(np.float32, copy=False) * 2 - 1
    if options["auto_stop"]:
        valid_y = (valid_y > 0).astype(np.float32, copy=False) * 2 - 1

    pathlib.Path(tmpdir).mkdir(parents=True, exist_ok=True)
    train_path, cached = _make_bin(tmpdir, train_x, train_y, fields)
    if cached:
        print(
            f'Using training data cache "{train_path}"',
            file=sys.stderr,
        )

    if options["auto_stop"]:
        valid_path, cached = _make_bin(tmpdir, valid_x, valid_y, fields)
        if cached:
            print(
                f'Using validation data cache "{valid_path}"',
                file=sys.stderr,
            )
    else:
        valid_path = ""

    model = wrapper.train_on_disk(
        train_path,
        valid_path,
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


def _make_bin(
    dir: str, x: sparse.csr_matrix, y: np.ndarray, fields: np.ndarray
) -> tuple[str, bool]:
    path = f"{dir}/{_fingerprint(x, y, fields)}"
    if pathlib.Path(path).exists():
        return path, True

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
