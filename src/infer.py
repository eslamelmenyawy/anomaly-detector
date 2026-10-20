"""
Run inference with an exported Anomalib model on one image or a whole folder.

The backend is picked from the weights file:
    model.xml -> OpenVINO (optimised CPU / Intel GPU inference)
    model.pt  -> PyTorch (TorchInferencer)

Examples:
    python src/infer.py --weights weights/cflow/weights/openvino/model.xml --source data/hazelnut_toy/crack
    python src/infer.py --weights weights/padim/weights/torch/model.pt --source sample.jpg --threshold 0.4
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anomaly detection inference (OpenVINO / PyTorch)")
    p.add_argument("--weights", required=True, help="Path to model.xml (OpenVINO) or model.pt (PyTorch)")
    p.add_argument("--source", required=True, help="Image file or folder of images")
    p.add_argument("--device", default="CPU", help="OpenVINO device: CPU, GPU or AUTO")
    p.add_argument("--threshold", type=float, default=None,
                   help="Custom score threshold (score > threshold = anomalous). "
                        "Default: use the model's exported threshold")
    p.add_argument("--save-dir", default="outputs", help="Where annotated images are written")
    return p.parse_args()


def load_inferencer(weights: Path, device: str):
    if weights.suffix == ".xml":
        from anomalib.deploy import OpenVINOInferencer

        metadata = weights.parent / "metadata.json"
        return OpenVINOInferencer(
            path=weights,
            metadata=metadata if metadata.exists() else None,
            device=device,
        ), "OpenVINO"
    if weights.suffix in {".pt", ".pth"}:
        from anomalib.deploy import TorchInferencer

        return TorchInferencer(path=weights), "PyTorch"
    raise ValueError(f"Unsupported weights format: {weights.suffix} (expected .xml or .pt)")


def collect_images(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [source]


def to_float(x) -> float:
    return float(x.item() if hasattr(x, "item") else x)


def annotate(image_path: Path, score: float, anomalous: bool) -> np.ndarray:
    img = cv2.imread(str(image_path))
    label = "ANOMALOUS" if anomalous else "NORMAL"
    color = (0, 0, 255) if anomalous else (0, 200, 0)
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(img, f"{label}  score={score:.3f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img


def main() -> None:
    args = parse_args()
    inferencer, backend = load_inferencer(Path(args.weights), args.device)
    images = collect_images(Path(args.source))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backend: {backend} | Images: {len(images)}\n")
    print(f"{'image':<30} {'score':>8} {'label':>10} {'ms':>8}")
    print("-" * 60)

    latencies = []
    for path in images:
        t0 = time.perf_counter()
        result = inferencer.predict(image=str(path))
        latencies.append((time.perf_counter() - t0) * 1000)

        score = to_float(result.pred_score)
        if args.threshold is not None:
            anomalous = score > args.threshold
        else:
            anomalous = bool(to_float(result.pred_label))

        print(f"{path.name:<30} {score:>8.4f} {'ANOMALOUS' if anomalous else 'NORMAL':>10} "
              f"{latencies[-1]:>8.1f}")
        cv2.imwrite(str(save_dir / path.name), annotate(path, score, anomalous))

        heat_map = getattr(result, "heat_map", None)
        if heat_map is not None:
            cv2.imwrite(str(save_dir / f"{path.stem}_heatmap.png"),
                        cv2.cvtColor(np.asarray(heat_map), cv2.COLOR_RGB2BGR))

    if latencies:
        # First call includes model warm-up, so report the median as well.
        print(f"\nMedian latency: {np.median(latencies):.1f} ms/image | Results saved to {save_dir}/")


if __name__ == "__main__":
    main()
