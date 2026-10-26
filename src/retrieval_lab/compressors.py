"""Independent PCA, Gaussian RP and centered linear AE implementations.

No RAE source code is imported. Training uses only unlabeled corpus vectors.
Retrieval normalization belongs to the common evaluation pipeline, not the decoder.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .common import write_json


def select_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Compressor:
    """Fit once on documents; transform queries and documents identically."""

    def __init__(self, method: str, dimension: int, seed: int = 0):
        if method not in {"original", "pca", "rp", "ae"}:
            raise ValueError(f"Unsupported method {method}; RAE/QRAE are not implemented")
        self.method = method
        self.dimension = dimension
        self.seed = seed
        self.arrays: dict[str, np.ndarray] = {}
        self.info = {}

    def fit(self, x: np.ndarray, ae_config: dict | None = None) -> "Compressor":
        x = np.asarray(x, dtype=np.float32)
        n, width = x.shape
        if self.dimension < 1 or self.dimension > width:
            raise ValueError(f"Dimension {self.dimension} must be in [1, {width}]")
        self.info = {
            "method": self.method,
            "input_dimension": width,
            "dimension": self.dimension,
            "seed": self.seed,
            "fit_documents": n,
        }
        start = time.perf_counter()
        if self.method == "original":
            if self.dimension != width:
                raise ValueError("Original must retain the entire embedding dimension")
        elif self.method == "pca":
            if self.dimension > min(n - 1, width):
                raise ValueError("PCA dimension exceeds the available centered sample rank")
            # Float64 fit improves numerical stability; deployed transform is float32.
            x64 = x.astype(np.float64)
            mean = x64.mean(axis=0)
            _, singular, vt = np.linalg.svd(x64 - mean, full_matrices=False)
            self.arrays = {
                "mean": mean.astype(np.float32),
                "encoder": vt[: self.dimension].astype(np.float32),
            }
            self.info["explained_variance_ratio"] = float(
                np.sum(singular[: self.dimension] ** 2) / np.sum(singular**2)
            )
            self.info["centering"] = "corpus mean; no whitening"
        elif self.method == "rp":
            rng = np.random.default_rng(self.seed)
            self.arrays["encoder"] = rng.normal(
                0, 1 / np.sqrt(self.dimension), (self.dimension, width)
            ).astype(np.float32)
            self.info["distribution"] = "independent N(0, 1/output_dimension); no centering"
        else:
            if ae_config is None:
                raise ValueError("AE training configuration is required")
            self._fit_ae(x, ae_config)
        self.info["fit_seconds"] = time.perf_counter() - start
        return self

    def _fit_ae(self, x: np.ndarray, config: dict) -> None:
        # PyTorch/FAISS macOS wheels can bundle incompatible OpenMP runtimes.
        # Never initialize Torch in the FAISS evaluation process.
        from .common import read_json

        with tempfile.TemporaryDirectory(prefix="retrieval-ae-") as directory:
            folder = Path(directory)
            np.save(folder / "inputs.npy", x)
            write_json(
                folder / "request.json",
                {
                    "dimension": self.dimension,
                    "seed": self.seed,
                    "training": config,
                },
            )
            subprocess.run(
                [sys.executable, "-m", "retrieval_lab.worker", "ae", str(folder)], check=True
            )
            with np.load(folder / "trained.npz", allow_pickle=False) as archive:
                self.arrays = {key: archive[key].copy() for key in archive.files}
            self.info.update(read_json(folder / "trained.json"))

    def _fit_ae_in_process(self, x: np.ndarray, config: dict) -> None:
        """Worker-only training implementation. Never invoke inside a FAISS process."""
        import torch

        device = select_device(config["device"])
        torch.set_num_threads(config.get("threads", 4))
        torch.manual_seed(self.seed)
        # Samples are generated on CPU with a private RNG for reproducible minibatch IDs.
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        centered = np.ascontiguousarray(x - mean, dtype=np.float32)
        inputs = torch.from_numpy(centered).to(device)
        encoder = torch.nn.Linear(x.shape[1], self.dimension, bias=False).to(device)
        decoder = torch.nn.Linear(self.dimension, x.shape[1], bias=False).to(device)
        parameters = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(parameters, lr=config["lr"], weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config["steps"] - 1), eta_min=config["min_lr"]
        )
        # Corpus-only probe diagnoses reconstruction; it is not a held-out evaluation set.
        probe_ids = torch.arange(min(1024, len(inputs)), device=device)
        history = []
        for step in range(1, config["steps"] + 1):
            ids = torch.randint(len(inputs), (config["batch_size"],), generator=generator)
            batch = inputs[ids.to(device)]
            optimizer.zero_grad(set_to_none=True)
            prediction = decoder(encoder(batch))
            loss = (prediction - batch).square().sum(dim=1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"AE non-finite loss at step {step}")
            loss.backward()
            optimizer.step()
            used_lr = optimizer.param_groups[0]["lr"]
            if step < config["steps"]:
                scheduler.step()
            if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
                with torch.no_grad():
                    probe = inputs[probe_ids]
                    error = (decoder(encoder(probe)) - probe).square().sum(1).mean().item()
                history.append(
                    {
                        "step": step,
                        "batch_reconstruction": loss.item(),
                        "corpus_probe_reconstruction": error,
                        "learning_rate": used_lr,
                    }
                )
                print(
                    f"  AE seed={self.seed} d={self.dimension} step={step} "
                    f"reconstruction={error:.6f}",
                    flush=True,
                )
        if device == "mps":
            torch.mps.synchronize()
        elif device.startswith("cuda"):
            torch.cuda.synchronize()
        self.arrays = {
            "mean": mean,
            "encoder": encoder.weight.detach().cpu().numpy().copy(),
            "decoder": decoder.weight.detach().cpu().numpy().copy(),
        }
        self.info.update(
            {
                "architecture": "centered, untied linear encoder/decoder; no bias",
                "loss": "mean over samples of summed squared reconstruction error",
                "optimizer": "Adam; weight_decay=0; cosine schedule per update",
                "training": dict(config),
                "actual_device": device,
                "actual_steps": config["steps"],
                "history": history,
                "checkpoint_selection": "last fixed step; no query labels/early stopping",
            }
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.method == "original":
            return x.copy()
        if "mean" in self.arrays:
            x = x - self.arrays["mean"]
        return np.ascontiguousarray(x @ self.arrays["encoder"].T, dtype=np.float32)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        if self.method != "ae":
            raise ValueError("Only the autoencoder has a decoder")
        return self.transform(x) @ self.arrays["decoder"].T + self.arrays["mean"]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        # Deployment artifact excludes AE decoder, optimizer and training-only state.
        deployed = {key: value for key, value in self.arrays.items() if key != "decoder"}
        np.savez(folder / "compressor.npz", **deployed)
        if "decoder" in self.arrays:
            np.savez(folder / "training_decoder.npz", decoder=self.arrays["decoder"])
        write_json(folder / "compressor.json", self.info)
        if self.info.get("history"):
            with (folder / "training.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(self.info["history"][0]))
                writer.writeheader()
                writer.writerows(self.info["history"])

    @classmethod
    def load(cls, folder: Path) -> "Compressor":
        from .common import read_json

        info = read_json(folder / "compressor.json")
        model = cls(info["method"], info["dimension"], info["seed"])
        with np.load(folder / "compressor.npz", allow_pickle=False) as archive:
            model.arrays = {key: archive[key] for key in archive.files}
        model.info = info
        return model
