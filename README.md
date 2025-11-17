# Face Detection & Frame Extraction using GPU Acceleration

### Input
- Video file (MP4, AVI, MOV, etc.)
- 3-5 reference images of the character/person to find

## What it does

1. Loads reference images of a specific person/character
2. Processes video frames using GPU-accelerated face detection
3. Compares detected faces against reference images using facial embeddings
4. Extracts and saves frames where the target character appears
5. Generates a timestamped log of all appearances


## Requirements
- Python 3.8+
- NVIDIA GPU with CUDA Support (Tested on CUDA 12.6 with GTX 1660+)
- cuDNN 9.x for CUDA 11.x/12.x
- 4GB+ VRAM recommended

## Setup

### 1. Install Dependencies
```bash
uv sync
```

### 2. Install cuDNN
- Download cuDNN v9.x from NVIDIA website based on your CUDA version
- Used for `onnx-runtime`


### 3. Setup Reference Images
```bash
mkdir reference_images
# Add 3-5 clear photos of the target person
```

## Usage

### Basic Usage
```bash
uv run main.py
```

### Configuration
Edit settings in `face_finder_gpu.py`:

```python
# In main() function:
video_file = "your_video.mp4"         # Your video file
sample_folder = "reference_images"     # Reference images folder

frame_skip = 5                         # Process every Nth frame (lower = more thorough)
similarity_threshold = 0.4             # Matching strictness (0.3-0.5 typical)

save_frames = True                     # Save matched frames as images
output_folder = "matched_frames"       # Output directory
```

**Pro Tip:** For best results, always do a quick test run with `frame_skip=30` first to validate your reference images work, then do a thorough pass with `frame_skip=5` once confirmed.


## Similarity Threshold Guidelines

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.3 | Very lenient | Catches more matches but more false positives |
| **0.4** | **Balanced (default)** | **Good balance of accuracy and recall** |
| 0.5 | Strict | Fewer false positives but might miss some appearances |
| 0.6+ | Very strict | Only very confident matches |

## Model Information

**InsightFace Buffalo_L Model:**
- Face detection: RetinaFace (10G parameters)
- Face recognition: ArcFace (R50 backbone)
- Additional: Gender/age estimation, facial landmarks
- Embedding size: 512 dimensions
- Accuracy: 99.8% on LFW benchmark

## System Requirements by Video Length

| Video Length | Minimum RAM | Recommended VRAM | frame_skip |
|--------------|-------------|------------------|------------|
| < 5 minutes | 8GB | 2GB | 3-5 |
| 5-15 minutes | 8GB | 4GB | 5-10 |
| 15-30 minutes | 16GB | 4GB | 5-10 |
| 30+ minutes | 16GB | 6GB+ | 10-15 |
| 2+ hours | 16GB+ | 6GB+ | 15-20 |

## License

Free to use for personal projects. For commercial use, check licenses:
- InsightFace (MIT)
- ONNX Runtime (MIT)
- OpenCV (Apache 2.0)

## Credits

- Face detection: [InsightFace](https://github.com/deepinsight/insightface)
- ONNX Runtime: [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime)

---

