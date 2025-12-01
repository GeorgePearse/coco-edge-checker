# coco-edge-checker

A fast Rust tool to validate COCO segmentation annotations against edge detection. Identifies annotations that don't align well with actual image edges - candidates for quality improvement.

## Features

- **Fast parallel processing** using Rayon
- **Multiple edge detectors**: Canny, Sobel, or combined
- **Handles both polygon and RLE segmentations**
- **Quality metrics per annotation**:
  - Boundary precision: % of mask boundary points near image edges
  - Boundary recall: % of image edges captured by the mask
  - Alignment score: F1-score combining both
- **JSON report output** with detailed per-annotation scores
- **Optional visualizations**: side-by-side comparison images

## Installation

```bash
cargo build --release
```

## Usage

```bash
# Basic usage
./target/release/coco-edge-checker \
  -a /path/to/annotations.json \
  -i /path/to/images/

# With visualizations, limited to 100 images
./target/release/coco-edge-checker \
  -a annotations.json \
  -i images/ \
  --visualize \
  -l 100

# Tune edge detection parameters
./target/release/coco-edge-checker \
  -a annotations.json \
  -i images/ \
  -m canny \
  --canny-low 30 \
  --canny-high 80 \
  --distance-threshold 5
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-a, --annotations` | Path to COCO annotations JSON | required |
| `-i, --images` | Path to images directory | required |
| `-o, --output` | Output directory | `output` |
| `-m, --method` | Edge detection: `canny`, `sobel`, `both` | `both` |
| `--canny-low` | Canny low threshold | `50.0` |
| `--canny-high` | Canny high threshold | `100.0` |
| `--distance-threshold` | Pixel distance for alignment | `3` |
| `-l, --limit` | Max images to process (0 = all) | `0` |
| `--visualize` | Generate visualization images | off |

## Output

- `output/quality_report.json` - Detailed quality metrics per annotation
- `output/vis_*.png` - Visualization images (with `--visualize`)

### Report Format

```json
{
  "total_images": 100,
  "total_annotations": 523,
  "avg_alignment_score": 0.72,
  "low_quality_count": 45,
  "images": [
    {
      "image_id": 123,
      "file_name": "image.jpg",
      "avg_alignment_score": 0.68,
      "annotations": [
        {
          "annotation_id": 456,
          "category": "person",
          "edge_alignment_score": 0.65,
          "boundary_precision": 0.70,
          "boundary_recall": 0.61,
          "issues": ["Many image edges near object are not captured by mask"]
        }
      ]
    }
  ]
}
```

## How It Works

1. Loads COCO annotations and corresponding images
2. Applies edge detection (Canny/Sobel) to each image
3. Extracts mask boundaries from polygon/RLE annotations
4. Compares annotation boundaries against detected edges:
   - **Precision**: Do mask boundaries follow actual edges?
   - **Recall**: Are all relevant edges captured by the mask?
5. Generates quality scores and identifies problematic annotations

## License

MIT
