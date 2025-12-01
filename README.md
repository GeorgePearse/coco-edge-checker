# coco-edge-checker

A fast Rust tool to auto-correct COCO segmentation annotations by snapping polygon vertices to detected image edges. Improves mask quality by aligning boundaries with actual edge features in images.

## Features

- **Fast parallel processing** using Rayon
- **Edge snapping** - automatically moves polygon vertices to nearby strong edges
- **Multiple edge detectors**: Canny, Sobel, or combined
- **Configurable thresholds** for snap distance, gradient improvement, and area change
- **Topology validation** - rejects corrections that change polygon area too much
- **Handles polygon segmentations** (RLE segmentations are passed through unchanged)
- **Correction report** - optional detailed JSON report of all modifications

## Installation

```bash
cargo build --release
```

## Usage

```bash
# Basic usage - correct annotations
./target/release/coco-edge-checker \
  -a annotations.json \
  -i images/ \
  -o corrected_annotations.json

# With custom thresholds
./target/release/coco-edge-checker \
  -a annotations.json \
  -i images/ \
  -o corrected.json \
  --snap-distance 3 \
  --min-gradient-improvement 30 \
  --max-area-change 0.05

# With corrections report
./target/release/coco-edge-checker \
  -a annotations.json \
  -i images/ \
  -o corrected.json \
  --report corrections_report.json
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-a, --annotations` | Path to COCO annotations JSON | required |
| `-i, --images` | Path to images directory | required |
| `-o, --output` | Output corrected COCO JSON file | required |
| `-m, --method` | Edge detection: `canny`, `sobel`, `both` | `both` |
| `--canny-low` | Canny low threshold | `50.0` |
| `--canny-high` | Canny high threshold | `100.0` |
| `--gradient-threshold` | Min gradient for edge detection (0-255) | `30` |
| `--snap-distance` | Max distance to snap vertices (pixels) | `5` |
| `--min-gradient-improvement` | Min gradient improvement to trigger snap (0-255) | `20` |
| `--max-area-change` | Max allowed area change (fraction, e.g. 0.1 = 10%) | `0.1` |
| `-l, --limit` | Max images to process (0 = all) | `0` |
| `--report` | Output corrections report JSON file | none |

## How It Works

1. Loads COCO annotations and corresponding images
2. Computes gradient magnitude and edge detection per image
3. For each polygon annotation:
   - Examines each vertex
   - Searches for stronger edges within `snap_distance` pixels
   - If a significantly stronger edge is found (gradient improvement > threshold), snaps the vertex
   - Validates the corrected polygon doesn't change area too much
4. Outputs corrected COCO JSON with updated segmentation coordinates

### Snapping Algorithm

For each polygon vertex at position (x, y):
1. Search all pixels within `snap_distance` radius
2. Find the pixel with highest gradient that exceeds `current_gradient + min_gradient_improvement`
3. Score candidates by `gradient / (1 + distance)` to prefer close, strong edges
4. If a valid target is found, snap the vertex to that position

### Safety Checks

- **Area validation**: If snapping would change the polygon area by more than `max_area_change`, the correction is rejected and the original polygon is kept
- **RLE passthrough**: RLE segmentations are left unchanged (only polygons are corrected)

## Output

### Corrected COCO JSON
Same format as input, but with updated `segmentation` coordinates for polygon annotations.

### Corrections Report (optional)
```json
{
  "total_annotations": 523,
  "annotations_modified": 156,
  "annotations_unchanged": 367,
  "annotations_skipped_rle": 0,
  "avg_points_snapped": 12.3,
  "avg_snap_distance": 2.1,
  "corrections": [
    {
      "annotation_id": 456,
      "image_id": 123,
      "category": "cat",
      "was_modified": true,
      "polygon_stats": {
        "points_snapped": 15,
        "points_unchanged": 42,
        "avg_snap_distance": 1.8,
        "max_snap_distance": 3.2,
        "area_change_pct": 2.3
      }
    }
  ]
}
```

## License

MIT
