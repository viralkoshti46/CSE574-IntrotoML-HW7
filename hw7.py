__Course__ = "CSE 574 - Intro To Machine Learning"
__author__ = "Viral Koshti"
__email__ = "viraljag@buffalo.edu"
__UBPersonNumber__ = "50761354"

import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# MNIST image files start with a 16-byte header.
def read_images(filename):
    with open(filename, "rb") as file:
        header = file.read(16)
        if len(header) != 16:
            raise ValueError("Image file header is incomplete")
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051 or rows != 28 or cols != 28:
            raise ValueError("Invalid MNIST image file")
        pixels = np.frombuffer(file.read(), dtype=np.uint8)

    if pixels.size != count * rows * cols:
        raise ValueError("Image count does not match the file header")

    # Each image becomes a row of 784 values between 0 and 1.
    return pixels.reshape(count, 784).astype(np.float32) / 255.0


def read_labels(filename):
    with open(filename, "rb") as file:
        header = file.read(8)
        if len(header) != 8:
            raise ValueError("Label file header is incomplete")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError("Invalid MNIST label file")
        labels = np.frombuffer(file.read(), dtype=np.uint8)

    if len(labels) != count or np.any(labels > 9):
        raise ValueError("Label count or digit values are invalid")
    return labels.copy()


class NeuralNetwork:
    def __init__(self, hidden, seed):
        random = np.random.default_rng(seed)
        # 784 inputs -> hidden layer -> 10 digit outputs
        # Scale the starting weights based on the number of inputs.
        self.w1 = random.standard_normal((784, hidden)).astype(np.float32) * np.sqrt(
            2.0 / 784
        )
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = random.standard_normal((hidden, 10)).astype(np.float32) * np.sqrt(
            2.0 / hidden
        )
        self.b2 = np.zeros(10, dtype=np.float32)

    def forward(self, x):
        # Do all neurons in a batch together using matrix multiplication.
        z1 = x @ self.w1 + self.b1
        hidden = np.maximum(z1, 0)  # ReLU
        z2 = hidden @ self.w2 + self.b2

        # Softmax converts the ten outputs into probabilities.
        # Subtract the row maximum to keep exp() from overflowing.
        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        return z1, hidden, probs

    def train_batch(self, x, labels, learning_rate, l2):
        z1, hidden, probs = self.forward(x)
        batch_size = len(labels)

        # Start at the output layer and work backward.
        output_error = probs.copy()
        output_error[np.arange(batch_size), labels] -= 1.0
        output_error /= batch_size

        dw2 = hidden.T @ output_error + l2 * self.w2
        db2 = np.sum(output_error, axis=0)
        hidden_error = (output_error @ self.w2.T) * (z1 > 0)
        dw1 = x.T @ hidden_error + l2 * self.w1
        db1 = np.sum(hidden_error, axis=0)

        # Update whole matrices instead of looping over each weight.
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2

    def evaluate(self, images, labels, batch_size=1024):
        total_errors = 0
        total_squared_error = 0.0
        total_cross_entropy = 0.0

        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            actual = labels[start:end]
            _, _, probs = self.forward(images[start:end])

            predicted = np.argmax(probs, axis=1)
            total_errors += int(np.count_nonzero(predicted != actual))
            target = np.eye(10, dtype=np.float32)[actual]
            total_squared_error += float(np.sum((target - probs) ** 2))
            total_cross_entropy += float(
                np.sum(
                    -np.log(np.maximum(probs[np.arange(len(actual)), actual], 1e-12))
                )
            )

        count = len(labels)
        return {
            "errors": total_errors,
            "accuracy_percent": 100.0 * (count - total_errors) / count,
            "mse": total_squared_error / (count * 10),
            "cross_entropy": total_cross_entropy / count,
        }


def save_graph(results, metric, title, filename):
    epochs = [row["epoch"] for row in results]
    training = [row["train_" + metric] for row in results]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training, marker="o", label="Training")
    if "test_" + metric in results[0]:
        testing = [row["test_" + metric] for row in results]
        plt.plot(epochs, testing, marker="s", label="Test")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title("Epoch vs. " + title)
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
    parser.add_argument("--decay", type=float, default=0.95)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=0)
    args = parser.parse_args()

    if min(args.hidden, args.epochs, args.batch_size) <= 0 or args.learning_rate <= 0:
        parser.error(
            "Hidden size, epochs, batch size and learning rate must be positive"
        )
    if not 0 < args.decay <= 1 or args.l2 < 0:
        parser.error("Decay must be between 0 and 1, and L2 cannot be negative")

    folder = args.data_dir
    train_images = folder / "train-images.idx3-ubyte"
    train_labels = folder / "train-labels.idx1-ubyte"
    test_images = folder / "t10k-images.idx3-ubyte"
    test_labels = folder / "t10k-labels.idx1-ubyte"

    if not train_images.is_file() or not train_labels.is_file():
        parser.error("Training images and labels were not found")

    x_train, y_train = read_images(train_images), read_labels(train_labels)
    if len(x_train) != len(y_train):
        raise ValueError("Training images and labels have different counts")
    if args.train_limit:
        x_train, y_train = x_train[: args.train_limit], y_train[: args.train_limit]

    if test_images.is_file() != test_labels.is_file():
        raise ValueError("Both MNIST test files are needed")
    has_test = test_images.is_file()
    if has_test:
        x_test, y_test = read_images(test_images), read_labels(test_labels)
        if len(x_test) != 10000 or len(y_test) != 10000:
            raise ValueError("The test data must have 10,000 images and labels")
    else:
        print("Test files not found. Only training results will be recorded.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    network = NeuralNetwork(args.hidden, args.seed)
    random = np.random.default_rng(args.seed + 1)
    results = []

    for epoch in range(1, args.epochs + 1):
        rate = args.learning_rate * args.decay ** (epoch - 1)
        order = random.permutation(len(y_train))
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            network.train_batch(x_train[batch], y_train[batch], rate, args.l2)

        train = network.evaluate(x_train, y_train)
        row = {"epoch": epoch, "learning_rate": rate}
        row.update({"train_" + key: value for key, value in train.items()})
        message = (
            f"Epoch {epoch:02d}: training accuracy={train['accuracy_percent']:.2f}% "
            f"errors={train['errors']} MSE={train['mse']:.6f}"
        )

        if has_test:
            test = network.evaluate(x_test, y_test)
            row.update({"test_" + key: value for key, value in test.items()})
            message += (
                f" | test accuracy={test['accuracy_percent']:.2f}% "
                f"errors={test['errors']} MSE={test['mse']:.6f}"
            )
        results.append(row)
        print(message, flush=True)

    with open(args.output_dir / "metrics.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    save_graph(
        results,
        "errors",
        "Classification errors (count)",
        args.output_dir / "classification_errors.png",
    )
    save_graph(results, "mse", "Mean squared error (MSE)", args.output_dir / "mse.png")
    np.savez_compressed(
        args.output_dir / "weights.npz",
        w1=network.w1,
        b1=network.b1,
        w2=network.w2,
        b2=network.b2,
    )
    settings = {
        "hidden": args.hidden,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "initial_learning_rate": args.learning_rate,
        "decay": args.decay,
        "l2": args.l2,
        "seed": args.seed,
        "train_examples": len(y_train),
        "test_examples": len(y_test) if has_test else 0,
    }
    (args.output_dir / "settings.json").write_text(json.dumps(settings, indent=2))
    print(f"Saved CSV, charts, weights, and settings to {args.output_dir}")


if __name__ == "__main__":
    main()
