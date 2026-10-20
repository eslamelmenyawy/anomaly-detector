"""
Train an unsupervised anomaly detection model with Anomalib and export it for deployment.

Only normal images are used for training. Abnormal images (optional) are used for evaluation.

Examples:
    # CFlow, exported to OpenVINO for fast CPU inference
    python src/train.py --model cflow --data-root data/hazelnut_toy --export openvino

    # PaDiM (fast, 1 epoch), exported to TorchScript-style .pt
    python src/train.py --model padim --data-root data/hazelnut_toy --export torch
"""

import argparse
from pathlib import Path

import torch
from anomalib import TaskType
from anomalib.data import Folder
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Cflow, Padim
from lightning.pytorch import seed_everything

MODELS = {
    # name: (constructor, default epochs)
    "cflow": (lambda: Cflow(), 25),
    "padim": (lambda: Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"],
                            pre_trained=True, n_features=100), 1),
}

EXPORTS = {"openvino": ExportType.OPENVINO, "torch": ExportType.TORCH}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and export an Anomalib anomaly detector")
    p.add_argument("--model", choices=MODELS, default="cflow", help="Anomaly model to train")
    p.add_argument("--data-root", required=True, help="Dataset root folder")
    p.add_argument("--normal-dir", default="good", help="Sub-folder with normal images")
    p.add_argument("--abnormal-dir", default="crack",
                   help="Sub-folder with defect images (evaluation only). Use '' if none")
    p.add_argument("--image-size", type=int, default=512, help="Square input size")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=None, help="Override the model's default epochs")
    p.add_argument("--export", choices=EXPORTS, default="openvino", help="Deployment format")
    p.add_argument("--output", default="weights", help="Export directory")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    datamodule = Folder(
        name=Path(args.data_root).name,
        root=args.data_root,
        normal_dir=args.normal_dir,
        abnormal_dir=args.abnormal_dir or None,
        normal_split_ratio=0.1,
        image_size=(args.image_size, args.image_size),
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        task=TaskType.CLASSIFICATION,
    )

    build_model, default_epochs = MODELS[args.model]
    model = build_model()

    engine = Engine(
        task=TaskType.CLASSIFICATION,
        threshold="F1AdaptiveThreshold",
        image_metrics=["AUROC", "F1Score"],
        max_epochs=args.epochs or default_epochs,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
    )

    engine.fit(model=model, datamodule=datamodule)

    if args.abnormal_dir:
        engine.test(model=model, datamodule=datamodule)

    exported = engine.export(
        model=model,
        export_type=EXPORTS[args.export],
        export_root=str(Path(args.output) / args.model),
    )
    print(f"\nModel exported to: {exported}")


if __name__ == "__main__":
    main()
