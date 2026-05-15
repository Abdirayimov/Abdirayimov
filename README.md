<h1 align="center">Khusan Abdirayimov</h1>

<p align="center">
  <b>Computer vision engineer</b> &middot; production CV/ML on NVIDIA DeepStream, TensorRT, and CUDA<br>
  <a href="mailto:khusanabdirayimov@gmail.com">khusanabdirayimov@gmail.com</a>
</p>

---

## What I work on

I build production multi-camera video analytics: detection,
tracking, pose, recognition, and the unglamorous engineering
between them - batched TRT inference, INT8 calibration, DeepStream
pipelines, FAISS-scale vector search, custom plugins.

My pinned repositories below are clean-room reference
implementations of patterns I have shipped. The code is original,
the algorithms are public (SCRFD, ArcFace, YOLOv8, RTMPose, ST-GCN,
BYTETrack, OSNet), and only public datasets and synthetic test
streams are used.

## Featured projects

### [multi-stream-face-recognition](https://github.com/Abdirayimov/multi-stream-face-recognition)
**C++ &middot; DeepStream &middot; TensorRT &middot; FAISS GPU**

Multi-camera face recognition pipeline: SCRFD detector + ArcFace
embedder + FAISS GPU index, all wired through a DeepStream
GStreamer pipeline with thread-safe source add/remove. Includes an
adaptive IVF-Flat / IVF-PQ index strategy that auto-selects based
on enrollment size, and a probe chain that batches detections
across cameras before encoding.

### [skeleton-action-recognition](https://github.com/Abdirayimov/skeleton-action-recognition)
**C++ &middot; PyTorch &middot; DeepStream &middot; ST-GCN**

Two-stage skeleton-based action recognition: YOLOv8 person
detection + RTMPose 17-keypoint estimation + ST-GCN classifier,
running through a NvDCF-tracked DeepStream pipeline. Per-track
skeleton buffer with forward-fill, centroid normalisation, and
configurable stride for re-classification. Ships with a PyTorch
Lightning training pipeline for the 10-class NTU-RGBD-60 subset.

### [multi-camera-person-tracking](https://github.com/Abdirayimov/multi-camera-person-tracking)
**C++ &middot; BYTETrack &middot; OSNet &middot; Hungarian matching**

Cross-camera person tracking with three pluggable single-camera
backends (BYTETrack, IoU baseline, NvDCF) behind one interface; a
rolling per-track OSNet appearance gallery; and a global identity
matcher that gates ReID similarity by zone topology and a
spatial-temporal window, then assigns optimal pairings via the
Hungarian algorithm.

### [tensorrt-optimization-toolkit](https://github.com/Abdirayimov/tensorrt-optimization-toolkit)
**C++ &middot; TensorRT &middot; INT8 calibration &middot; plugins**

The unglamorous parts of shipping TRT in production: ONNX -> engine
compilation with FP32/FP16/INT8, IInt8EntropyCalibrator2 with
on-disk cache, dynamic shape profiles, a reference custom plugin
(`GeluPlugin` via `IPluginV2DynamicExt`), polygraphy-style
inspectors for both `.onnx` and `.engine` files, latency /
throughput / memory probes, and a two-engine accuracy differ for
spotting INT8 regressions.

### [edge-anomaly-detection](https://github.com/Abdirayimov/edge-anomaly-detection)
**C++ &middot; PaDiM &middot; OpenCV &middot; TensorRT &middot; edge**

Real-time anomaly detection on industrial / perimeter video.
Three orthogonal detectors fused under temporal smoothing and ROI
gating: MOG2 background subtraction, scene-change (HSV histogram +
DCT pHash + edge density), and PaDiM per-position Mahalanobis on
pretrained ResNet18 features. Polygonal zones with severity
multipliers and ignore masks. Pluggable alert sinks. Ships with a
Python scaffold to fit the PaDiM Gaussian statistics on a folder
of "normal" images.

## Stack I reach for

|                | |
|----------------|--|
| **Inference** | NVIDIA DeepStream 7.x / 8.x, TensorRT 8.6+ / 10, CUDA 12, FAISS GPU |
| **Models** | YOLOv8/v11, SCRFD, ArcFace (insightface), RTMPose, ST-GCN, OSNet |
| **Training** | PyTorch 2.x, PyTorch Lightning, Hydra, mmpose-style toolchains |
| **C++** | C++17, CMake 3.22+, Eigen 3.4, spdlog, yaml-cpp, GStreamer |
| **Serving** | gRPC, FastAPI, Redis, PostgreSQL + pgvector |
| **Ops** | Docker multi-stage, Linux (Ubuntu 22.04), CUDA Container Toolkit |

## How I work

- I prefer **measure-then-optimise**. Every project I ship has a
  benchmark CLI that I keep honest with `cudaEvent`-based timing
  and percentile latency, not means.
- I **gate INT8 conversions against accuracy diffs** rather than
  trusting that the conversion "looked fine".
- For deployment, **Docker + a single `cmake --install` step**
  beats every other distribution mechanism I have tried.
- Configuration is **YAML, validated at load time**, not flag
  spaghetti.

## Open to

- Production CV consulting (DeepStream / TensorRT / multi-camera)
- Inference-optimisation contracts (INT8, plugins, throughput)
- ML systems work where the bottleneck is engineering, not
  modelling

## Contact

- Email: [khusanabdirayimov@gmail.com](mailto:khusanabdirayimov@gmail.com)
- GitHub: this is it
