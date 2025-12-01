# coco-edge-checker

A fast Rust tool to validate COCO segmentation annotations against edge detection. Identifies annotations that don't align well with actual image edges - candidates for quality improvement.

## Features

- **Fast parallel processing** using Rayon
- **Distance transform** for precise edge-to-boundary distance measurement
- **Multiple edge detectors**: Canny, Sobel, or combined
- **Handles both polygon and RLE segmentations**
- **Comprehensive quality metrics**:
  - Distance-based scoring with exponential decay
  - Mean/median/percentile distance statistics
  - Confidence indicators for edge case handling
- **Smart edge case handling**:
  - Low-contrast boundaries (flags uncertainty vs bad annotation)
  - Textured objects (filters internal edges from recall)
  - Image boundary truncation detection
  - Small annotation reliability warnings
- **JSON report output** with detailed per-annotation analysis
- **Color-coded visualizations**: green (good) to red (bad)

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

# Tune scoring parameters
./target/release/coco-edge-checker \
  -a annotations.json \
  -i images/ \
  --scoring-sigma 3.0 \
  --gradient-threshold 20
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
| `--scoring-sigma` | Sigma for distance scoring (higher = lenient) | `2.0` |
| `--gradient-threshold` | Min gradient for edge confidence (0-255) | `30` |
| `--min-boundary-points` | Min points for reliable stats | `32` |
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
  "avg_distance_score": 0.68,
  "low_quality_count": 45,
  "low_confidence_count": 23,
  "images": [
    {
      "image_id": 123,
      "file_name": "image.jpg",
      "avg_alignment_score": 0.68,
      "avg_distance_score": 0.65,
      "annotations": [
        {
          "annotation_id": 456,
          "category": "cat",

          "edge_alignment_score": 0.35,
          "boundary_precision": 0.28,
          "boundary_recall": 0.45,

          "distance_score": 0.72,
          "distance_stats": {
            "mean": 2.1,
            "median": 1.5,
            "p90": 4.5,
            "pct_within_1px": 0.45,
            "pct_within_2px": 0.68
          },

          "adjusted_precision": 0.75,
          "adjusted_recall": 0.69,

          "edge_confidence_ratio": 0.40,
          "is_textured": true,
          "is_low_contrast": true,
          "is_truncated": false,
          "is_small": false,
          "evaluation_confidence": "Medium",

          "issues": [
            "Low edge contrast on 60% of boundary",
            "Textured object (892 internal vs 156 boundary edges)"
          ]
        }
      ]
    }
  ]
}
```

## Key Metrics Explained

### Distance Score
Uses exponential decay based on actual pixel distance to nearest edge:
- Score = exp(-d²/2σ²) averaged over all boundary points
- At σ=2.0: 1px→0.88, 2px→0.61, 3px→0.32, 5px→0.04

### Evaluation Confidence
- **High**: No edge cases, metrics are reliable
- **Medium**: One edge case factor present
- **Low**: Multiple factors (textured + low contrast + small + truncated)

### Interpreting Results
A **low `edge_alignment_score` with reasonable `distance_score` and `is_low_contrast: true`** means "annotation is probably fine, we just can't detect edges to verify it" - NOT "bad annotation".

## How It Works

1. Loads COCO annotations and corresponding images
2. Computes gradient magnitude and edge detection per image
3. Builds distance transform (distance from each pixel to nearest edge)
4. For each annotation boundary:
   - Measures actual distance to nearest edge at each point
   - Computes distance statistics and exponential decay score
   - Analyzes edge confidence (gradient strength at boundary)
   - Detects edge cases (texture, low contrast, truncation, size)
   - Generates adjusted metrics excluding low-confidence regions
5. Outputs comprehensive quality report

## License

MIT
