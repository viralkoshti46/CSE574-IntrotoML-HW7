"""CSE 574A Homework 7: MNIST digit classification using backpropagation.

Only NumPy is used for neural-network math. Matplotlib draws plots.
Run: python hw7.py --data-dir /path/to/mnist --epochs 15
Optional experiment: python hw7.py --data-dir /path/to/mnist --hidden 64 --learning-rate 0.05 --output-dir results_64
"""

import argparse
import csv
import json
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_images(path):
    with open(path, "rb") as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError(f"Bad MNIST image header: {path}")
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051 or (rows, cols) != (28, 28):
            raise ValueError(f"Unexpected MNIST image format: {path}")
        raw = f.read()
    pixels = np.frombuffer(raw, dtype=np.uint8)
    if pixels.size != count * rows * cols:
        raise ValueError(f"Image file size does not match header: {path}")
    return pixels.reshape(count, 784).astype(np.float32) / 255.0


def read_labels(path):
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"Bad MNIST label header: {path}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Unexpected MNIST label format: {path}")
        raw = f.read()
    labels = np.frombuffer(raw, dtype=np.uint8)
    if len(labels) != count or np.any(labels > 9):
        raise ValueError(f"Invalid labels or number of labels: {path}")
    return labels.copy()


def file_at(directory, standard_name, alternates=()):
    for name in (standard_name, *alternates):
        p = directory / name
        if p.is_file():
            return p
    return None


class NeuralNetwork:
    def __init__(self, hidden, seed):
        rng = np.random.default_rng(seed)
        # He initialization helps ReLU hidden neurons start with useful scales.
        self.w1 = rng.standard_normal((784, hidden)).astype(np.float32) * np.sqrt(
            2.0 / 784
        )
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = rng.standard_normal((hidden, 10)).astype(np.float32) * np.sqrt(
            2.0 / hidden
        )
        self.b2 = np.zeros(10, dtype=np.float32)

    def forward(self, x):
        z1 = x @ self.w1 + self.b1
        h = np.maximum(z1, 0)
        logits = h @ self.w2 + self.b2
        # Subtract max to prevent numerical overflow in exp().
        exp_values = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return z1, h, probabilities

    def train_batch(self, x, labels, learning_rate, l2):
        z1, h, probabilities = self.forward(x)
        n = len(labels)
        # Derivative of softmax + cross-entropy with respect to logits.
        dlogits = probabilities.copy()
        dlogits[np.arange(n), labels] -= 1.0
        dlogits /= n
        grad_w2 = h.T @ dlogits + l2 * self.w2
        grad_b2 = np.sum(dlogits, axis=0)
        dh = dlogits @ self.w2.T
        dz1 = dh * (z1 > 0)  # ReLU derivative
        grad_w1 = x.T @ dz1 + l2 * self.w1
        grad_b1 = np.sum(dz1, axis=0)
        self.w1 -= learning_rate * grad_w1
        self.b1 -= learning_rate * grad_b1
        self.w2 -= learning_rate * grad_w2
        self.b2 -= learning_rate * grad_b2

    def evaluate(self, x, labels, batch_size=1024):
        errors = 0
        square_error_sum = 0.0
        cross_entropy_sum = 0.0
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            _, _, probs = self.forward(x[start:stop])
            y = labels[start:stop]
            errors += int(np.count_nonzero(np.argmax(probs, axis=1) != y))
            target = np.eye(10, dtype=np.float32)[y]
            # MSE here is averaged over all 10 output units and all images.
            square_error_sum += float(np.sum((target - probs) ** 2))
            cross_entropy_sum += float(
                np.sum(-np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12)))
            )
        return {
            "errors": errors,
            "accuracy_percent": 100.0 * (len(labels) - errors) / len(labels),
            "mse": square_error_sum / (len(labels) * 10),
            "cross_entropy": cross_entropy_sum / len(labels),
        }


def make_plot(history, field, ylabel, filename):
    epochs = [r["epoch"] for r in history]
    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs, [r[f"train_{field}"] for r in history], marker="o", label="Training"
    )
    if history and "test_" + field in history[0]:
        plt.plot(
            epochs, [r[f"test_{field}"] for r in history], marker="s", label="Test"
        )
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Epoch vs. {ylabel}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("hw7_results"))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument(
        "--decay",
        type=float,
        default=0.95,
        help="Multiply learning rate by this factor after each epoch",
    )
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="Optional quick sanity test; final run must use 0",
    )
    args = parser.parse_args()
    if min(args.hidden, args.epochs, args.batch_size) <= 0 or args.learning_rate <= 0:
        parser.error(
            "Hidden size, epochs, batch size, and learning rate must be positive"
        )
    if not 0 < args.decay <= 1 or args.l2 < 0:
        parser.error("Decay must be in (0,1], and L2 must be nonnegative")

    train_images = file_at(
        args.data_dir, "train-images.idx3-ubyte", ("train-images.idx3-ubyte",)
    )
    train_labels = file_at(
        args.data_dir, "train-labels.idx1-ubyte", ("train-labels.idx1-ubyte",)
    )
    test_images = file_at(
        args.data_dir, "t10k-images.idx3-ubyte", ("t10k-images.idx3-ubyte",)
    )
    test_labels = file_at(
        args.data_dir, "t10k-labels.idx1-ubyte", ("t10k-labels.idx1-ubyte",)
    )
    if train_images is None or train_labels is None:
        parser.error("Training images/labels were not found in --data-dir")
    x_train, y_train = read_images(train_images), read_labels(train_labels)
    if len(x_train) != len(y_train):
        raise ValueError("Number of training images and labels differs")
    if args.train_limit:
        x_train, y_train = x_train[: args.train_limit], y_train[: args.train_limit]
    x_test, y_test = None, None
    if (test_images is None) != (test_labels is None):
        raise ValueError("Provide both test files, not just one")
    if test_images is not None:
        x_test, y_test = read_images(test_images), read_labels(test_labels)
        if len(x_test) != len(y_test) or len(x_test) != 10000:
            raise ValueError(
                "The homework test set must contain 10,000 matched examples"
            )
    else:
        print(
            "WARNING: MNIST test files are missing. Test metrics will NOT be fabricated."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nn = NeuralNetwork(args.hidden, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    history = []
    for epoch in range(1, args.epochs + 1):
        lr = args.learning_rate * args.decay ** (epoch - 1)
        indices = rng.permutation(len(y_train))
        for begin in range(0, len(indices), args.batch_size):
            batch_indices = indices[begin : begin + args.batch_size]
            nn.train_batch(x_train[batch_indices], y_train[batch_indices], lr, args.l2)
        train = nn.evaluate(x_train, y_train)
        row = {"epoch": epoch, "learning_rate": lr}
        row.update({f"train_{k}": v for k, v in train.items()})
        if x_test is not None:
            test = nn.evaluate(x_test, y_test)
            row.update({f"test_{k}": v for k, v in test.items()})
        history.append(row)
        display = (
            f"Epoch {epoch:02d}: training accuracy={train['accuracy_percent']:.2f}% "
            f"errors={train['errors']} MSE={train['mse']:.6f}"
        )
        if x_test is not None:
            display += (
                f" | test accuracy={test['accuracy_percent']:.2f}% "
                f"errors={test['errors']} MSE={test['mse']:.6f}"
            )
        print(display, flush=True)

    with open(args.output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    make_plot(
        history,
        "errors",
        "Classification errors (count)",
        args.output_dir / "classification_errors.png",
    )
    make_plot(history, "mse", "Mean squared error (MSE)", args.output_dir / "mse.png")
    np.savez_compressed(
        args.output_dir / "weights.npz", w1=nn.w1, b1=nn.b1, w2=nn.w2, b2=nn.b2
    )
    configuration = {
        "hidden": args.hidden,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "initial_learning_rate": args.learning_rate,
        "decay": args.decay,
        "l2": args.l2,
        "seed": args.seed,
        "train_examples": len(y_train),
        "test_examples": 0 if x_test is None else len(y_test),
    }
    (args.output_dir / "settings.json").write_text(json.dumps(configuration, indent=2))
    print(f"Saved CSV, charts, weights, and settings to {args.output_dir}")


if __name__ == "__main__":
    main()
