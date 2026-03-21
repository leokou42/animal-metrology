# Animal Eye Metrology

Image segmentation + metrology pipeline that detects animals in images, locates their eyes, and measures inter-ocular and inter-animal eye distances with three layers of precision.

![Architecture](docs/architecture.png)

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API Framework | FastAPI | REST API with Swagger UI |
| Instance Segmentation | YOLO11n-seg (ultralytics) | Detect animal contours + bounding boxes |
| Eye Keypoint Detection | RTMPose-m + AP-10K (rtmlib/ONNX) | Locate left/right eye keypoints |
| Metric Depth Estimation | Apple Depth Pro (primary) / DA V2 (fallback) | Per-pixel depth in meters + focal length |
| Visualization | OpenCV | Annotated result images |
| Containerization | Docker + docker-compose | One-command deployment |

## Three-Layer Measurement Architecture

### Layer 1: 2D Pixel Distance

Basic Euclidean distance in image pixel space.

- **Formula**: `d = sqrt((x2-x1)^2 + (y2-y1)^2)`
- **Unit**: pixels
- **Limitation**: No physical meaning — closer animals appear larger.

### Layer 2: Metric Distance via Depth Pro

Uses Apple Depth Pro (ICLR 2025) to estimate metric depth (meters) and focal length, then projects pixel coordinates to 3D camera space:

```
X = (x_pixel - cx) * Z / focal_length
Y = (y_pixel - cy) * Z / focal_length
d = sqrt((X2-X1)^2 + (Y2-Y1)^2 + (Z2-Z1)^2)
```

- **Unit**: meters
- **Advantage**: True 3D distance without camera calibration
- **Note**: Metric depth has estimation error — not lab-grade precision

### Layer 3: Sanity Check

Cross-validates metric IOD against known biological ranges:

| Animal | Expected IOD |
|--------|-------------|
| Cat | 5-6 cm |
| Dog | 6-10 cm |
| Giraffe | 18-22 cm |
| Sheep | 6-8 cm |

Results: **PASS** (within range), **WARNING** (within 50% tolerance), **FAIL** (unreasonable)

## Quick Start

### Local Setup

```bash
git clone <repo-url>
cd animal-metrology
pip install -r requirements.txt

# For metric depth (optional — ~1.8GB model):
pip install git+https://github.com/apple/ml-depth-pro.git
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('apple/DepthPro', 'depth_pro.pt', local_dir='weights/')"

cp .env.example .env
python -m scripts.run_demo    # Process test images
python -m app.main            # Start API server
```

### Docker

```bash
docker compose up --build
```

API available at `http://localhost:8000`

## API Documentation

Swagger UI: `http://localhost:8000/docs`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/coco/animals` | Browse COCO images with multiple animals |
| POST | `/api/v1/analyze/{image_id}` | Run pipeline on a COCO image |
| POST | `/api/v1/analyze/upload` | Run pipeline on an uploaded image |

### Query Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `steps` | `segment`, `eyes`, `full` | `full` | Pipeline depth |
| `visualize` | `true`, `false` | `true` | Generate annotated image |

### Examples

```bash
# Full pipeline (3 layers + visualization)
curl -X POST "http://localhost:8000/api/v1/analyze/287545"

# Segmentation only, JSON only
curl -X POST "http://localhost:8000/api/v1/analyze/287545?steps=segment&visualize=false"

# Eye detection with visualization
curl -X POST "http://localhost:8000/api/v1/analyze/287545?steps=eyes"
```

## Test Images

| Image ID | Animals | Description |
|----------|---------|-------------|
| 287545 | 2 giraffes | Large animals, different depths |
| 402473 | 2 cats | Small animals, similar depth |
| 547383 | 2 sheep | Medium animals, different depths |

## Model Selection Rationale

### YOLO11n-seg
- **Why**: COCO pretrained, detection + segmentation in one pass, nano variant for CPU
- **Metrics**: mAP@50 = 73.4 on COCO val2017
- **Evaluation**: IoU (Intersection over Union) measures segmentation mask quality

### RTMPose-m + AP-10K
- **Why**: AP-10K is the standard animal pose benchmark (17 keypoints, 23 species)
- **Metrics**: AP@0.5 = 72.2 on AP-10K
- **Evaluation**: OKS (Object Keypoint Similarity) measures keypoint accuracy
- **Delivery**: ONNX format via rtmlib — avoids heavy mmpose/mmcv dependency

### Apple Depth Pro
- **Why**: Outputs metric depth (meters) + estimated focal length — no camera calibration needed
- **Paper**: "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second" (ICLR 2025)
- **Evaluation**: AbsRel, RMSE on standard benchmarks (NYU, KITTI)
- **Design**: Optional — pipeline gracefully falls back to pixel-only if unavailable

## Verification Methods

1. **Visual**: Annotated images with eye markers, distance lines, and 3-layer labels
2. **Sanity Check**: Metric IOD cross-validated against known animal biology
3. **Unit Tests**: 27 tests covering distance computation, metric projection, sanity check logic
4. **Demo Script**: `python -m scripts.run_demo` processes all test images end-to-end

## Limitations

- **Metric depth accuracy**: Depth Pro is zero-shot — error varies by scene. Sanity checks help flag outliers.
- **Eye detection**: RTMPose may fail on profile views (eyes set to None, distances skipped)
- **YOLO threshold**: 0.5 confidence may miss partially occluded animals
- **CPU inference**: Depth Pro takes ~30-60s per image on CPU; GPU recommended for production
- **No camera calibration**: Depth Pro estimates focal length, but true calibration would be more accurate

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
animal-metrology/
├── app/
│   ├── main.py                     # FastAPI entry point
│   ├── config.py                   # Pydantic settings
│   ├── models/schemas.py           # Response schemas (3-layer)
│   ├── routers/analyze.py          # API endpoints
│   ├── services/
│   │   ├── coco_filter.py          # COCO dataset filtering
│   │   ├── segmentation.py         # YOLO instance segmentation
│   │   ├── eye_detection.py        # RTMPose eye keypoints
│   │   ├── depth_estimation.py     # Depth Pro / DA V2
│   │   └── measurement.py          # 3-layer distance computation
│   └── utils/visualization.py      # Result visualization
├── tests/                          # Unit + integration tests
├── data/
│   ├── test_images/                # 3 pre-selected COCO images
│   ├── test_annotations/           # Minimal COCO annotation JSON
│   └── sample_results.csv          # Pipeline output CSV
├── scripts/
│   ├── run_demo.py                 # One-command demo
│   └── generate_architecture.py    # Architecture diagram
├── docs/architecture.png
├── Dockerfile, docker-compose.yml
├── requirements.txt, .env.example
└── README.md
```
