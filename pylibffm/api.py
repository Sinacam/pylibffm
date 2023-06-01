from __future__ import annotations
import pathlib

from . import wrapper
import scipy.sparse as sparse
import numpy as np
import uuid

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

        wrapper.predict(
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

    train_path = f"{tmpdir}/{uuid.uuid4().hex}"
    wrapper.arr2bin(
        train_x.shape[0],
        train_x.shape[1],
        train_y,
        fields,
        train_x.data,
        train_x.indices,
        train_x.indptr,
        train_path,
    )

    if options["auto_stop"]:
        valid_path = f"{tmpdir}/{uuid.uuid4().hex}"
        wrapper.arr2bin(
            valid_x.shape[0],
            valid_x.shape[1],
            valid_y,
            fields,
            valid_x.data,
            valid_x.indices,
            valid_x.indptr,
            valid_path,
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
