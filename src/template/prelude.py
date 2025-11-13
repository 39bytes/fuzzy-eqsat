import numpy as np
import json
import keras


def load_params(path: str) -> dict[str, np.ndarray]:
    with open(path) as f:
        params = json.load(f)["params"]
        return {
            k: np.array(v["val"]).reshape(tuple(v["shape"])) for k, v in params.items()
        }


def load_dataset():
    print("Loading MNIST dataset...")
    _, (x_test, y_test) = keras.datasets.mnist.load_data()

    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)

    y_test = keras.utils.to_categorical(y_test, 10)

    return x_test, y_test


params = load_params("parameters.json")


def param(name: str) -> np.ndarray:
    return params[name]


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def softmax(x: np.ndarray):
    x = x - np.max(x, axis=0, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=0, keepdims=True)
