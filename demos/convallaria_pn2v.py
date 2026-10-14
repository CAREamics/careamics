"""Run the Convallaria PN2V GMM protocol and record PSNR.

Calibration fits a Gaussian mixture on the static diaphragm stack. Bootstrap
trains N2V on the noisy movie and fits the same trainer on that prediction.
Both PN2V runs predict the MMSE image. Metrics are appended to ``metrics.log``.
Sample frames are saved next to the noisy input and the mean ground truth.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from lightning.pytorch import seed_everything

from careamics import CAREamist, NoiseModelTrainer
from careamics.config import create_advanced_n2v_config, create_advanced_pn2v_config
from careamics.config.noise_model import GaussianMixtureNMConfig

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "demos" / "Convallaria_diaphragm"
OUT_DIR = ROOT / "demos" / "convallaria_pn2v_outputs"
RUN_DIR = OUT_DIR / "runs"
LOG_PATH = OUT_DIR / "metrics.log"

NOISY_NAME = "20190520_tl_25um_50msec_05pc_488_130EM_Conv.tif"
CALIBRATION_NAME = "20190726_tl_50um_500msec_wf_130EM_FD.tif"
SAMPLE_FRAMES = (0, 50, 99)
CROP = 512

MODEL_PARAMS = {"depth": 3, "num_channels_init": 64}
TRAINER_PARAMS = {
    "accumulate_grad_batches": 20,
    "accelerator": "mps",
    "devices": 1,
}
SCHEDULER_PARAMS = {"factor": 0.5, "patience": 10}

logger = logging.getLogger("convallaria_pn2v")


def paper_psnr(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """PSNR with data range ``max(gt) - min(gt)``.

    Parameters
    ----------
    ground_truth : np.ndarray
        Reference image.
    prediction : np.ndarray
        Denoised image, same shape as ``ground_truth``.

    Returns
    -------
    float
        PSNR in decibels.
    """
    data_range = float(ground_truth.max() - ground_truth.min())
    error = (ground_truth - prediction) ** 2
    finite = np.isfinite(error)
    mse = float(np.mean(error[finite]))
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def as_syx(array: np.ndarray) -> np.ndarray:
    """Drop a singleton channel axis so the array is ``SYX``.

    Parameters
    ----------
    array : np.ndarray
        Prediction with shape ``SYX`` or ``SCYX``.

    Returns
    -------
    np.ndarray
        Array with shape ``SYX``.
    """
    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0]
    if array.ndim != 3:
        raise ValueError(f"Expected SYX or SCYX, got shape {array.shape}.")
    return array


def sample_scores(prediction: np.ndarray, ground_truth: np.ndarray) -> list[float]:
    """PSNR of each predicted sample against the mean ground truth.

    Parameters
    ----------
    prediction : np.ndarray
        Predicted sample frames, ``SYX``. One row per entry of ``SAMPLE_FRAMES``.
    ground_truth : np.ndarray
        Mean of the top-left crop, ``YX``.

    Returns
    -------
    list of float
        One PSNR per sample frame.
    """
    frames = as_syx(prediction)
    return [paper_psnr(ground_truth, frame) for frame in frames]


def train_gmm(signal: np.ndarray, observation: np.ndarray) -> GaussianMixtureNMConfig:
    """Fit one GMM (K=3, n=2, min_sigma=50).

    Parameters
    ----------
    signal : np.ndarray
        Clean or pseudo-clean stack, ``SYX``.
    observation : np.ndarray
        Noisy stack, same shape as ``signal``.

    Returns
    -------
    GaussianMixtureNMConfig
        Trained single-channel noise model.
    """
    trainer = NoiseModelTrainer(n_gaussian=3, n_coeff=2, min_sigma=50.0)
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SYX",
        n_epochs=2000,
        learning_rate=1e-1,
        batch_size=250000,
    )
    return trainer.get_config().noise_models[0]


def _normalization(noisy: np.ndarray) -> dict[str, list[float]]:
    """Global mean and standard deviation of the noisy movie.

    Parameters
    ----------
    noisy : np.ndarray
        Noisy movie.

    Returns
    -------
    dict
        ``input_means`` and ``input_stds`` for the configuration.
    """
    return {
        "input_means": [float(noisy.mean())],
        "input_stds": [float(noisy.std())],
    }


def train_pn2v(
    noise_model: GaussianMixtureNMConfig,
    name: str,
    train_stack: np.ndarray,
    val_stack: np.ndarray,
    normalization: dict[str, list[float]],
) -> CAREamist:
    """Train PN2V with the Convallaria UNet settings.

    Parameters
    ----------
    noise_model : GaussianMixtureNMConfig
        Trained noise model.
    name : str
        Experiment name.
    train_stack : np.ndarray
        Training frames.
    val_stack : np.ndarray
        Validation frames.
    normalization : dict
        Mean and standard deviation of the full noisy movie.

    Returns
    -------
    CAREamist
        Trained engine.
    """
    # Depth 3 needs a spatial size divisible by 8. Upstream uses 100.
    config = create_advanced_pn2v_config(
        experiment_name=name,
        data_type="array",
        axes="SYX",
        patch_size=[96, 96],
        batch_size=1,
        noise_model=noise_model,
        num_epochs=200,
        num_steps=5,
        num_classes=800,
        n_channels=1,
        roi_size=5,
        masked_pixel_percentage=3.125,
        model_params=MODEL_PARAMS,
        trainer_params=TRAINER_PARAMS,
        optimizer_params={"lr": 1e-3},
        lr_scheduler_params=SCHEDULER_PARAMS,
        normalization_params=normalization,
        n_val_patches=1,
        num_workers=0,
    )
    engine = CAREamist(config, work_dir=RUN_DIR)
    engine.train(train_data=train_stack, val_data=val_stack)
    return engine


def predict_frames(engine: CAREamist, noisy: np.ndarray) -> np.ndarray:
    """Predict the top-left crop of ``SAMPLE_FRAMES`` only.

    Parameters
    ----------
    engine : CAREamist
        Trained engine.
    noisy : np.ndarray
        Noisy movie, ``SYX``.

    Returns
    -------
    np.ndarray
        Predictions with shape ``(len(SAMPLE_FRAMES), 512, 512)``.
    """
    crops = np.stack([noisy[index, :CROP, :CROP] for index in SAMPLE_FRAMES])
    logger.info("Predicting frames %s", SAMPLE_FRAMES)
    predicted, _ = engine.predict(crops)
    return as_syx(predicted[0]).astype(np.float32)


def save_comparisons(
    name: str,
    noisy: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> None:
    """Save sample frames beside the input and the mean ground truth.

    Parameters
    ----------
    name : str
        Mode name used in the file names.
    noisy : np.ndarray
        Noisy movie, ``SYX``.
    prediction : np.ndarray
        Denoised movie, ``SYX``.
    ground_truth : np.ndarray
        Mean of the top-left crop, ``YX``.
    """
    low, high = np.percentile(ground_truth, [1, 99])
    frames = as_syx(prediction)
    for row, index in enumerate(SAMPLE_FRAMES):
        noisy_crop = noisy[index, :CROP, :CROP]
        pred_crop = frames[row]
        stack = np.stack([noisy_crop, pred_crop, ground_truth]).astype(np.float32)
        stem = OUT_DIR / f"{name}_frame{index:03d}"
        tifffile.imwrite(stem.with_suffix(".tif"), stack)

        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, image, title in zip(
            axes,
            (noisy_crop, pred_crop, ground_truth),
            ("input", "MMSE", "mean GT"),
            strict=True,
        ):
            axis.imshow(image, vmin=low, vmax=high, cmap="gray")
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(f"{name} frame {index}, top-left {CROP}")
        figure.savefig(stem.with_suffix(".png"), dpi=120, bbox_inches="tight")
        plt.close(figure)
        logger.info("Saved %s", stem.with_suffix(".png"))


def log_metrics(name: str, target_db: float, scores: list[float]) -> None:
    """Write mean and per-frame PSNR to the log.

    Parameters
    ----------
    name : str
        Mode name.
    target_db : float
        Paper PSNR for this mode. The paper number averages every test frame;
        this log covers ``SAMPLE_FRAMES`` only.
    scores : list of float
        PSNR of each sample frame, in ``SAMPLE_FRAMES`` order.
    """
    mean_db = float(np.mean(scores))
    std_db = float(np.std(scores))
    logger.info(
        "METRIC %s mean_psnr_db=%.4f std_psnr_db=%.4f target_db=%.2f n_frames=%d "
        "frames=%s",
        name,
        mean_db,
        std_db,
        target_db,
        len(scores),
        list(SAMPLE_FRAMES),
    )
    for index, score in zip(SAMPLE_FRAMES, scores, strict=True):
        logger.info("METRIC %s frame=%d psnr_db=%.4f", name, index, score)
    summary_path = OUT_DIR / "metrics.json"
    summary: dict[str, object] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    summary[name] = {
        "mean_psnr_db": mean_db,
        "std_psnr_db": std_db,
        "target_db": target_db,
        "frames": list(SAMPLE_FRAMES),
        "per_frame_psnr_db": scores,
    }
    summary_path.write_text(json.dumps(summary, indent=2))


def configure_logging() -> None:
    """Send log records and process output to ``metrics.log`` and the console."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_PATH.open("a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


class _Tee:
    """Duplicate writes onto several streams.

    Parameters
    ----------
    *streams : object
        File-like objects with ``write`` and ``flush``.
    """

    def __init__(self, *streams: object) -> None:
        """Store the streams.

        Parameters
        ----------
        *streams : object
            File-like objects with ``write`` and ``flush``.
        """
        self.streams = streams

    def write(self, data: str) -> None:
        """Write ``data`` to each stream.

        Parameters
        ----------
        data : str
            Text to write.
        """
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        """Flush every stream."""
        for stream in self.streams:
            stream.flush()


def main() -> None:
    """Fit both noise models, train PN2V, and record Convallaria PSNR."""
    configure_logging()
    seed_everything(42)
    started = time.perf_counter()
    logger.info("Loading %s", DATA_DIR)

    noisy = tifffile.imread(DATA_DIR / NOISY_NAME).astype(np.float32)
    calibration = tifffile.imread(DATA_DIR / CALIBRATION_NAME).astype(np.float32)
    logger.info("noisy %s calibration %s", noisy.shape, calibration.shape)
    train_stack = noisy[:-5]
    # CAREamist runs every validation patch and ignores limit_val_batches.
    # Keep the held-out frames, but only their top-left 192 so an epoch stays short.
    val_stack = noisy[-5:, :192, :192]
    normalization = _normalization(noisy)
    ground_truth = noisy[:, :CROP, :CROP].mean(axis=0)

    logger.info("Fitting calibration GMM")
    calibration_signal = np.repeat(
        calibration.mean(axis=0, keepdims=True), calibration.shape[0], axis=0
    )
    calibration_nm = train_gmm(calibration_signal, calibration)
    logger.info("Training calibration PN2V")
    calibration_engine = train_pn2v(
        calibration_nm,
        "convallaria_pn2v_gmm",
        train_stack,
        val_stack,
        normalization,
    )
    calibration_prediction = predict_frames(calibration_engine, noisy)
    calibration_scores = sample_scores(calibration_prediction, ground_truth)
    log_metrics("calibration", 36.47, calibration_scores)
    save_comparisons("calibration", noisy, calibration_prediction, ground_truth)

    logger.info("Training bootstrap N2V")
    n2v_config = create_advanced_n2v_config(
        experiment_name="convallaria_n2v",
        data_type="array",
        axes="SYX",
        patch_size=[96, 96],
        batch_size=1,
        num_epochs=200,
        num_steps=10,
        n_channels=1,
        roi_size=5,
        masked_pixel_percentage=3.125,
        model_params=MODEL_PARAMS,
        trainer_params=TRAINER_PARAMS,
        optimizer_params={"lr": 1e-3},
        lr_scheduler_params=SCHEDULER_PARAMS,
        normalization_params=normalization,
        n_val_patches=1,
        num_workers=0,
    )
    n2v_engine = CAREamist(n2v_config, work_dir=RUN_DIR)
    n2v_engine.train(train_data=train_stack, val_data=val_stack)
    n2v_prediction = predict_frames(n2v_engine, noisy)

    logger.info("Fitting bootstrap GMM on the sample frames")
    sample_observations = np.stack(
        [noisy[index, :CROP, :CROP] for index in SAMPLE_FRAMES]
    )
    bootstrap_nm = train_gmm(n2v_prediction, sample_observations)
    logger.info("Training bootstrap PN2V")
    bootstrap_engine = train_pn2v(
        bootstrap_nm,
        "convallaria_pn2v_boot_gmm",
        train_stack,
        val_stack,
        normalization,
    )
    bootstrap_prediction = predict_frames(bootstrap_engine, noisy)
    bootstrap_scores = sample_scores(bootstrap_prediction, ground_truth)
    log_metrics("bootstrap", 36.70, bootstrap_scores)
    save_comparisons("bootstrap", noisy, bootstrap_prediction, ground_truth)

    elapsed_min = (time.perf_counter() - started) / 60.0
    logger.info("Finished in %.1f min. Outputs in %s", elapsed_min, OUT_DIR)


def predict_only(checkpoint: Path, name: str, target_db: float) -> None:
    """Predict ``SAMPLE_FRAMES`` from a saved checkpoint and log PSNR.

    Parameters
    ----------
    checkpoint : Path
        Lightning checkpoint written by a finished training run.
    name : str
        Name used in the metrics log and output files.
    target_db : float
        Paper PSNR for this mode.
    """
    configure_logging()
    noisy = tifffile.imread(DATA_DIR / NOISY_NAME).astype(np.float32)
    ground_truth = noisy[:, :CROP, :CROP].mean(axis=0)
    logger.info("Predicting %s from %s", name, checkpoint)
    engine = CAREamist(checkpoint_path=checkpoint, work_dir=RUN_DIR)
    prediction = predict_frames(engine, noisy)
    log_metrics(name, target_db, sample_scores(prediction, ground_truth))
    save_comparisons(name, noisy, prediction, ground_truth)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--predict-only":
        predict_only(Path(sys.argv[2]), sys.argv[3], float(sys.argv[4]))
    else:
        main()
