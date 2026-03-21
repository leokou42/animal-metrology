# Animal Eye Metrology

Image segmentation + metrology pipeline that detects animals in images, locates their eyes, and measures inter-ocular and inter-animal eye distances.

![Architecture](docs/architecture.png)

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API Framework | FastAPI | REST API with Swagger UI |
| Instance Segmentation | YOLO11n-seg (ultralytics) | Detect animal contours + bounding boxes |
| Eye Keypoint Detection | RTMPose-m + AP-10K (rtmlib/ONNX) | Locate left/right eye keypoints |
| Depth Estimation | Depth Anything V2 Small (transformers) | Monocular depth for distance correction |
| Visualization | OpenCV | Annotated result images |
| Containerization | Docker + docker-compose | One-command deployment |

## Pipeline

```
Input Image → YOLO Segmentation → RTMPose Eye Detection → Depth Estimation → Distance Measurement → Visualization
```

1. **Segmentation**: YOLO11n-seg identifies animals and outputs bounding boxes + binary masks
2. **Eye Detection**: RTMPose (AP-10K pretrained) runs top-down pose estimation per bbox, extracting left_eye and right_eye keypoints
3. **Depth Estimation**: Depth Anything V2 estimates per-pixel relative depth (optional)
4. **Measurement**: Euclidean distances computed at two layers:
   - **Layer 1 (2D Pixel)**: `d = sqrt((x2-x1)^2 + (y2-y1)^2)` in pixel space
   - **Layer 2 (Depth-Corrected)**: `corrected = pixel_dist * avg_depth / min(depth_a, depth_b)`
5. **Visualization**: Annotated image with masks, eye markers, and distance lines

## Quick Start

### Local Setup

```bash
# Clone and install
git clone <repo-url>
cd animal-metrology
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Run demo (processes test images, generates CSV + annotated images)
python -m scripts.run_demo

# Start API server
python -m app.main
```

### Docker

```bash
docker compose up --build
```

API available at `http://localhost:8000`

## API Documentation

Interactive Swagger UI: `http://localhost:8000/docs`

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
# Full pipeline with visualization
curl -X POST "http://localhost:8000/api/v1/analyze/287545"

# Segmentation only, no image
curl -X POST "http://localhost:8000/api/v1/analyze/287545?steps=segment&visualize=false"

# Eye detection only
curl -X POST "http://localhost:8000/api/v1/analyze/287545?steps=eyes"
```

## Test Images

| Image ID | Animals | Description |
|----------|---------|-------------|
| 287545 | 2 giraffes | Large animals, different depths |
| 402473 | 2 cats | Small animals, similar depth |
| 547383 | 2 sheep | Medium animals, different depths |

Pre-selected images are in `data/test_images/`. Run `python -m scripts.run_demo` to process all three.

## Three-Layer Measurement Architecture

### Layer 1: 2D Pixel Distance (Implemented)

Basic Euclidean distance in image pixel space.

- **Intra-animal**: Distance between an animal's left and right eyes
- **Inter-animal**: Distance between right eyes of all animal pairs
- **Formula**: `d = sqrt((x2-x1)^2 + (y2-y1)^2)`
- **Unit**: pixels
- **Limitation**: No physical meaning. Closer animals appear larger, inflating pixel distances.

### Layer 2: Depth-Corrected Distance (Implemented)

Uses Depth Anything V2 to estimate relative depth per pixel, then corrects inter-animal distances for perspective distortion.

- **Formula**: `corrected = pixel_dist * avg_depth / min(depth_a, depth_b)`
- **Unit**: depth-corrected pixels (relative, not metric)
- **Effect**: If animal B is farther than A, the raw pixel distance underestimates the real separation. Depth correction compensates for this.

### Layer 3: Metric Distance (Not Implemented — Explained)

Converting to real-world units (cm/mm) requires either:

**Option A — Camera Calibration**: Use focal length (f) and metric depth to project pixel coordinates to 3D:
```
X = (x_pixel - cx) * Z / f
Y = (y_pixel - cy) * Z / f
d = sqrt((X2-X1)^2 + (Y2-Y1)^2 + (Z2-Z1)^2)
```

**Option B — Known Reference Object**: Use known animal dimensions (e.g., adult cat binocular distance ~5cm) to compute pixel-to-cm scale factor.

Both require information not available in the COCO dataset (no camera parameters, no species-specific metadata). This limitation is inherent to the dataset, not the pipeline.

## Model Selection Rationale

### YOLO11n-seg (Instance Segmentation)

- **Why**: COCO pretrained, simultaneous detection + segmentation, nano variant runs on CPU
- **Metrics**: mAP@50 = 73.4 on COCO val2017
- **Alternative considered**: Mask R-CNN — higher accuracy but 10x slower, impractical for CPU inference

### RTMPose-m + AP-10K (Eye Detection)

- **Why**: AP-10K is the standard animal pose benchmark (10,015 images, 23 species, 17 keypoints including eyes)
- **Metrics**: AP@0.5 = 72.2 on AP-10K, OKS-based evaluation
- **Delivery**: ONNX format via rtmlib, avoids heavy mmpose/mmcv dependency chain
- **Alternative considered**: Full mmpose stack — same model but requires mmengine + mmcv (~500MB+ install)

### Depth Anything V2 Small (Depth Estimation)

- **Why**: State-of-the-art monocular depth estimation, zero-shot, auto-downloads via HuggingFace
- **Metrics**: AbsRel = 0.261 on NYU Depth V2
- **Design**: Optional — pipeline gracefully falls back to pixel-only distances if torch/transformers not installed

## Verification Methods

1. **Visual Verification**: Annotated images show eye markers overlaid on the original image. Misplaced markers are immediately visible.
2. **Sanity Checks**: Giraffe binocular distance (~7px) < Cat binocular distance (~50px) — expected since giraffes were photographed from farther away.
3. **Depth Correction Validation**: For image 547383 (two sheep at different depths), pixel distance = 175.6px vs depth-corrected = 265.0px — the correction is significant because one sheep is much closer to the camera.
4. **Unit Tests**: 26 tests covering distance computation, edge cases (missing eyes, empty lists), and API endpoints.

## Limitations

- **Pixel-space distances** have no physical meaning without camera calibration
- **Eye detection accuracy** depends on animal pose — profile views may fail (eyes set to None)
- **Depth correction** uses relative depth, not metric — corrected values are fairer comparisons but not centimeters
- **YOLO confidence threshold** (0.5) may miss partially occluded animals
- **CPU inference only** — GPU would significantly speed up all three models

## Project Structure

```
animal-metrology/
├── app/
│   ├── main.py                     # FastAPI entry point
│   ├── config.py                   # Pydantic settings
│   ├── models/schemas.py           # Response schemas
│   ├── routers/analyze.py          # API endpoints
│   ├── services/
│   │   ├── coco_filter.py          # COCO dataset filtering
│   │   ├── segmentation.py         # YOLO instance segmentation
│   │   ├── eye_detection.py        # RTMPose eye keypoints
│   │   ├── depth_estimation.py     # Depth Anything V2
│   │   └── measurement.py          # Distance computation
│   └── utils/visualization.py      # Result visualization
├── tests/
│   ├── test_pipeline.py            # API + integration tests
│   └── test_measurement.py         # Measurement unit tests
├── data/
│   ├── test_images/                # 3 pre-selected COCO images
│   ├── test_annotations/           # Minimal COCO annotation JSON
│   └── sample_results.csv          # Pipeline output CSV
├── scripts/
│   ├── run_demo.py                 # One-command demo
│   ├── generate_architecture.py    # Architecture diagram generator
│   └── extract_test_annotations.py # COCO annotation extractor
├── docs/architecture.png           # System architecture diagram
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Running Tests

```bash
python -m pytest tests/ -v
```
