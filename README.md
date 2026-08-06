<h1 align="center">Khusan Abdirayimov</h1>

<p align="center">
  <b>Computer vision &amp; agentic AI engineer</b> — production video analytics and auditable LLM systems<br>
  NVIDIA DeepStream · TensorRT · CUDA · C++ · RAG · RL
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white" alt="C++17">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/TensorRT-8.6%2B%20%2F%2010%2F11-76B900?logo=nvidia&logoColor=white" alt="TensorRT">
  <img src="https://img.shields.io/badge/DeepStream-7.x%20%2F%208.x-76B900?logo=nvidia&logoColor=white" alt="DeepStream">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Docker-multi--stage-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

---

I build **agentic AI systems** — hybrid retrieval (RAG), tool-use orchestration,
and RL policies with audit trails and human-in-the-loop gates. I also build
production multi-camera video analytics — detection, tracking, pose,
recognition, and the unglamorous engineering between them: batched TRT
inference, INT8 calibration, DeepStream pipelines, FAISS-scale vector search,
custom plugins.

The repositories below are **clean-room reference implementations** of patterns
I have shipped. The code is original, the algorithms are public, and every demo
is the **actual binary / CLI running** (the C++ vision demos rendered on an
RTX 4080) — not a mock-up.

## Featured work

### [agentic-fiscal-rag](https://github.com/Abdirayimov/agentic-fiscal-rag) &nbsp;·&nbsp; `Python · BGE-M3 RAG · RL (Q-learning) · audit + HITL`

[![CI](https://github.com/Abdirayimov/agentic-fiscal-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdirayimov/agentic-fiscal-rag/actions/workflows/ci.yml)

<a href="https://github.com/Abdirayimov/agentic-fiscal-rag">
  <img src="https://raw.githubusercontent.com/Abdirayimov/agentic-fiscal-rag/main/docs/assets/afr_demo.png" width="560" alt="agentic-fiscal-rag answering a tax question with effective-date versioning and an audited computation">
</a>

An agentic RAG system over fiscal/tax law: BGE-M3 hybrid retrieval over an
article-level index, a deterministic **audited** compute layer, and an RL policy
that learns *when* to retrieve, compute, answer — or escalate to a human. The
same question on two dates returns two article versions; every computed answer
cites the article, version and inputs it used. DB-less and reproducible.

### [multi-stream-face-recognition](https://github.com/Abdirayimov/multi-stream-face-recognition) &nbsp;·&nbsp; `C++ · DeepStream · TensorRT · FAISS GPU`

[![CI](https://github.com/Abdirayimov/multi-stream-face-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdirayimov/multi-stream-face-recognition/actions/workflows/ci.yml)

<a href="https://github.com/Abdirayimov/multi-stream-face-recognition">
  <img src="https://raw.githubusercontent.com/Abdirayimov/multi-stream-face-recognition/main/docs/assets/face_detect_demo.jpg" width="470" alt="SCRFD detecting all 29 faces in the 1927 Solvay Conference photo">
</a>

SCRFD + ArcFace + FAISS GPU through a DeepStream pipeline, with adaptive
IVF-Flat / IVF-PQ index selection. The detector resolves its outputs by shape,
so it runs the stock insightface export directly — here finding all 29 faces in
the 1927 Solvay photo. 101 unit tests cover the algorithmic stages.

### [eduvision-cqi](https://github.com/Abdirayimov/eduvision-cqi) &nbsp;·&nbsp; `Python · IEEE Access · fuzzy inference · GRU`

[![CI](https://github.com/Abdirayimov/eduvision-cqi/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdirayimov/eduvision-cqi/actions/workflows/ci.yml)

Code accompanying an IEEE Access submission (revised manuscript under review) on classroom process quality
assessment from valence–arousal trajectories. A Mamdani fuzzy inference layer
sits on top of reconstructed VA signals to produce a continuous quality index.
43 tests verify the reference implementation against the printed tables, so a
reviewer can check the code against the manuscript mechanically. No weights or
corpus are released — the README explains why.

### [multi-camera-person-tracking](https://github.com/Abdirayimov/multi-camera-person-tracking) &nbsp;·&nbsp; `C++ · YOLO11 · BYTETrack · OSNet · Hungarian`

[![CI](https://github.com/Abdirayimov/multi-camera-person-tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdirayimov/multi-camera-person-tracking/actions/workflows/ci.yml)

<a href="https://github.com/Abdirayimov/multi-camera-person-tracking">
  <img src="https://raw.githubusercontent.com/Abdirayimov/multi-camera-person-tracking/main/docs/assets/mctrack_demo.gif" width="470" alt="YOLO11 + BYTETrack tracking a busy concourse">
</a>

YOLO11 detection → BYTETrack (two in-process backends, NvDCF via DeepStream) → OSNet ReID → a
cross-camera identity matcher that keeps a **global id** per person across
views (ReID cosine, gated by zone topology + a spatial-temporal window, solved
with Hungarian assignment).

### [skeleton-action-recognition](https://github.com/Abdirayimov/skeleton-action-recognition) &nbsp;·&nbsp; `C++ · PyTorch · ST-GCN · TensorRT`

[![CI](https://github.com/Abdirayimov/skeleton-action-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdirayimov/skeleton-action-recognition/actions/workflows/ci.yml)

<a href="https://github.com/Abdirayimov/skeleton-action-recognition">
  <img src="https://raw.githubusercontent.com/Abdirayimov/skeleton-action-recognition/main/docs/assets/skeleton_demo.gif" width="300" alt="ST-GCN classifying NTU-RGB+D skeleton clips">
</a>

YOLOv8 + RTMPose + ST-GCN two-stage action recognition. The ST-GCN here was
**trained from scratch on public NTU-RGB+D** (76% cross-subject acc) and runs
as a TensorRT engine — the GIF shows it classifying held-out clips, 10/10.

### [tensorrt-optimization-toolkit](https://github.com/Abdirayimov/tensorrt-optimization-toolkit) &nbsp;·&nbsp; `C++ · TensorRT · INT8 · plugins`

[![CI](https://github.com/Abdirayimov/tensorrt-optimization-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdirayimov/tensorrt-optimization-toolkit/actions/workflows/ci.yml)

<a href="https://github.com/Abdirayimov/tensorrt-optimization-toolkit">
  <img src="https://raw.githubusercontent.com/Abdirayimov/tensorrt-optimization-toolkit/main/docs/assets/trt_toolkit_demo.png" width="560" alt="trt-toolkit build / inspect / benchmark on a ResNet18 FP16 engine">
</a>

ONNX → engine compilation (FP32/FP16/INT8 with entropy calibration), dynamic
shape profiles, a reference custom plugin, polygraphy-style inspectors, and a
two-engine accuracy differ for catching INT8 regressions.

### [edge-anomaly-detection](https://github.com/Abdirayimov/edge-anomaly-detection) &nbsp;·&nbsp; `C++ · PaDiM · OpenCV · TensorRT`

[![CI](https://github.com/Abdirayimov/edge-anomaly-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdirayimov/edge-anomaly-detection/actions/workflows/ci.yml)

<a href="https://github.com/Abdirayimov/edge-anomaly-detection">
  <img src="https://raw.githubusercontent.com/Abdirayimov/edge-anomaly-detection/main/docs/assets/eanom_demo.gif" width="470" alt="Real-time anomaly detection with motion + scene-change + ROI gating">
</a>

Real-time anomaly detection for industrial / perimeter video. Three orthogonal
detectors (MOG2 motion, scene-change, PaDiM feature distance) fused under
temporal smoothing, ROI zones, and severity-laddered alerts.

> The C++ repositories' CI runs clang-format, cppcheck and — where the code
> allows it — the CPU-only unit tests. GitHub's runners carry no CUDA,
> TensorRT or DeepStream, so a green check there does not mean the GPU
> pipeline was built; each README says so explicitly.

## Stack I reach for

| | |
|----------------|--|
| **LLM / agents** | RAG (BGE-M3 hybrid · RRF) · tool-use orchestration · RL (Q-learning) · audit + HITL |
| **Inference** | NVIDIA DeepStream 7.x / 8.x · TensorRT 8.6+ / 10 / 11 · CUDA 12 · FAISS GPU |
| **Models** | YOLOv8 / YOLO11 · SCRFD · ArcFace · RTMPose · ST-GCN · OSNet · PaDiM |
| **Training** | PyTorch 2.x · PyTorch Lightning · Hydra |
| **C++** | C++17 · CMake · Eigen · spdlog · yaml-cpp · GStreamer · GoogleTest |
| **Serving / data** | gRPC · FastAPI · Redis · PostgreSQL + pgvector |
| **Ops** | Docker multi-stage · Linux · NVIDIA Container Toolkit · GitHub Actions |

## How I work

- **Measure, then optimise.** Every project ships a benchmark CLI kept honest
  with `cudaEvent` timing and percentile latency — not means.
- **Gate INT8 against accuracy diffs**, never "it looked fine".
- **Config is YAML, validated at load**, not flag spaghetti.
- **Deploy as Docker + one `cmake --install`.**

## Open to

**Agentic AI / RAG systems** (retrieval, tool-use, auditable outputs) ·
production CV consulting (DeepStream / TensorRT / multi-camera) · inference
optimisation (INT8, plugins, throughput) · ML-systems work where the bottleneck
is engineering, not modelling.

<p align="center">
  <a href="mailto:khusanabdirayimov@gmail.com"><img src="https://img.shields.io/badge/Email-khusanabdirayimov%40gmail.com-D14836?logo=gmail&logoColor=white" alt="Email"></a>
</p>

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=Abdirayimov&layout=compact&langs_count=8&theme=github_dark&hide_border=true" alt="Top languages">
</p>
