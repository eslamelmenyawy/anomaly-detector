
# Anomaly Detector

This project uses [Anomalib](https://github.com/openvinotoolkit/anomalib) for training and inference of anomaly detection models such as Patchcore, Padim, and FastFlow.

## Features
- Training on folder-based datasets
- Export to OpenVINO format
- Inference with TorchInferencer and OpenVINOInferencer
- Support for classification and segmentation tasks

## How to Use
1. Install dependencies
2. Train using `Padim` or `Patchcore`
3. Export using `export_to_openvino()`
4. Run inference on new images

## Dataset
- Hazelnut toy dataset (structured with `good/` and `crack/` folders)


