use anyhow::{Context, Result};
use clap::Parser;
use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use imageproc::distance_transform::{distance_transform, Norm};
use imageproc::edges::canny;
use imageproc::gradients::{horizontal_sobel, vertical_sobel};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "coco-edge-checker")]
#[command(about = "Check COCO segmentation annotations against edge detection")]
struct Args {
    /// Path to COCO annotations JSON file
    #[arg(short, long)]
    annotations: PathBuf,

    /// Path to images directory
    #[arg(short, long)]
    images: PathBuf,

    /// Output directory for reports/visualizations
    #[arg(short, long, default_value = "output")]
    output: PathBuf,

    /// Edge detection method: canny, sobel, both
    #[arg(short, long, default_value = "both")]
    method: String,

    /// Canny low threshold
    #[arg(long, default_value = "50.0")]
    canny_low: f32,

    /// Canny high threshold
    #[arg(long, default_value = "100.0")]
    canny_high: f32,

    /// Distance threshold for edge-mask alignment (pixels)
    #[arg(long, default_value = "3")]
    distance_threshold: u32,

    /// Sigma for distance scoring (higher = more lenient)
    #[arg(long, default_value = "2.0")]
    scoring_sigma: f64,

    /// Gradient threshold for edge confidence (0-255)
    #[arg(long, default_value = "30")]
    gradient_threshold: u8,

    /// Minimum boundary points for reliable statistics
    #[arg(long, default_value = "32")]
    min_boundary_points: usize,

    /// Number of images to process (0 = all)
    #[arg(short, long, default_value = "0")]
    limit: usize,

    /// Generate visualization images
    #[arg(long)]
    visualize: bool,
}

// COCO Format Structures
#[derive(Debug, Deserialize, Serialize, Clone)]
struct CocoDataset {
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnnotation>,
    categories: Vec<CocoCategory>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct CocoImage {
    id: u64,
    file_name: String,
    width: u32,
    height: u32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct CocoAnnotation {
    id: u64,
    image_id: u64,
    category_id: u64,
    segmentation: Segmentation,
    area: f64,
    bbox: Vec<f64>,
    iscrowd: u8,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
enum Segmentation {
    Polygon(Vec<Vec<f64>>),
    Rle(RleSegmentation),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct RleSegmentation {
    counts: RleCounts,
    size: Vec<u32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
enum RleCounts {
    Encoded(String),
    Raw(Vec<u32>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct CocoCategory {
    id: u64,
    name: String,
}

// Edge data with both binary edges and gradient magnitude
struct EdgeData {
    binary_edges: GrayImage,
    gradient_magnitude: GrayImage,
    distance_map: GrayImage,  // Distance to nearest edge (u8, capped at 255)
}

// Distance statistics
#[derive(Debug, Serialize, Clone, Default)]
struct DistanceStats {
    mean: f64,
    median: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    p90: f64,
    pct_within_1px: f64,
    pct_within_2px: f64,
    pct_within_threshold: f64,
    sample_size: usize,
}

// Quality metrics for an annotation
#[derive(Debug, Serialize)]
struct AnnotationQuality {
    annotation_id: u64,
    image_id: u64,
    category: String,

    // Original metrics (backward compatible)
    edge_alignment_score: f64,
    boundary_precision: f64,
    boundary_recall: f64,

    // Distance-based metrics
    distance_score: f64,
    distance_stats: DistanceStats,

    // Adjusted metrics (accounting for edge cases)
    adjusted_precision: f64,
    adjusted_recall: f64,

    // Confidence indicators
    edge_confidence_ratio: f64,
    avg_edge_strength: f64,
    is_textured: bool,
    is_low_contrast: bool,
    is_truncated: bool,
    is_small: bool,
    evaluation_confidence: String,

    issues: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ImageQuality {
    image_id: u64,
    file_name: String,
    avg_alignment_score: f64,
    avg_distance_score: f64,
    annotation_count: usize,
    annotations: Vec<AnnotationQuality>,
}

#[derive(Debug, Serialize)]
struct Report {
    total_images: usize,
    total_annotations: usize,
    avg_alignment_score: f64,
    avg_distance_score: f64,
    low_quality_count: usize,
    low_confidence_count: usize,
    images: Vec<ImageQuality>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading COCO annotations from {:?}...", args.annotations);
    let file = File::open(&args.annotations)
        .with_context(|| format!("Failed to open annotations file: {:?}", args.annotations))?;
    let reader = BufReader::new(file);
    let dataset: CocoDataset = serde_json::from_reader(reader)
        .with_context(|| "Failed to parse COCO JSON")?;

    println!(
        "Loaded {} images, {} annotations, {} categories",
        dataset.images.len(),
        dataset.annotations.len(),
        dataset.categories.len()
    );

    let category_map: HashMap<u64, String> = dataset
        .categories
        .iter()
        .map(|c| (c.id, c.name.clone()))
        .collect();

    let mut annotations_by_image: HashMap<u64, Vec<CocoAnnotation>> = HashMap::new();
    for ann in dataset.annotations.iter() {
        annotations_by_image
            .entry(ann.image_id)
            .or_default()
            .push(ann.clone());
    }

    std::fs::create_dir_all(&args.output)?;

    let images_to_process: Vec<_> = if args.limit > 0 {
        dataset.images.iter().take(args.limit).collect()
    } else {
        dataset.images.iter().collect()
    };

    let pb = ProgressBar::new(images_to_process.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("#>-"),
    );

    let image_results: Vec<Option<ImageQuality>> = images_to_process
        .par_iter()
        .map(|img| {
            let result = process_image(
                img,
                &args,
                &category_map,
                annotations_by_image.get(&img.id).map(|v| v.as_slice()),
            );
            pb.inc(1);
            result.ok()
        })
        .collect();

    pb.finish_with_message("Processing complete");

    let valid_results: Vec<ImageQuality> = image_results.into_iter().flatten().collect();

    let total_annotations: usize = valid_results.iter().map(|r| r.annotation_count).sum();
    let avg_score: f64 = if !valid_results.is_empty() {
        valid_results.iter().map(|r| r.avg_alignment_score).sum::<f64>() / valid_results.len() as f64
    } else {
        0.0
    };
    let avg_distance_score: f64 = if !valid_results.is_empty() {
        valid_results.iter().map(|r| r.avg_distance_score).sum::<f64>() / valid_results.len() as f64
    } else {
        0.0
    };

    let low_quality_count = valid_results
        .iter()
        .flat_map(|r| &r.annotations)
        .filter(|a| a.distance_score < 0.5)
        .count();

    let low_confidence_count = valid_results
        .iter()
        .flat_map(|r| &r.annotations)
        .filter(|a| a.evaluation_confidence == "Low")
        .count();

    let report = Report {
        total_images: valid_results.len(),
        total_annotations,
        avg_alignment_score: avg_score,
        avg_distance_score,
        low_quality_count,
        low_confidence_count,
        images: valid_results,
    };

    let report_path = args.output.join("quality_report.json");
    let report_file = File::create(&report_path)?;
    serde_json::to_writer_pretty(report_file, &report)?;

    println!("\n=== Quality Report ===");
    println!("Images processed: {}", report.total_images);
    println!("Annotations checked: {}", report.total_annotations);
    println!("Average alignment score: {:.2}%", report.avg_alignment_score * 100.0);
    println!("Average distance score: {:.2}%", report.avg_distance_score * 100.0);
    println!("Low quality (<50% distance score): {}", report.low_quality_count);
    println!("Low confidence (uncertain evaluation): {}", report.low_confidence_count);
    println!("Report saved to: {:?}", report_path);

    Ok(())
}

fn process_image(
    img_info: &CocoImage,
    args: &Args,
    category_map: &HashMap<u64, String>,
    annotations: Option<&[CocoAnnotation]>,
) -> Result<ImageQuality> {
    let img_path = args.images.join(&img_info.file_name);
    let img = image::open(&img_path)
        .with_context(|| format!("Failed to load image: {:?}", img_path))?;
    let gray = img.to_luma8();

    // Compute edge data with gradient magnitude
    let edge_data = compute_edge_data(&gray, &args);

    let annotations = annotations.unwrap_or(&[]);
    let mut annotation_qualities = Vec::new();

    for ann in annotations.iter().filter(|a| a.iscrowd == 0) {
        let category = category_map
            .get(&ann.category_id)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        let quality = evaluate_annotation(
            ann,
            &edge_data,
            img_info.width,
            img_info.height,
            args,
            &category,
        );
        annotation_qualities.push(quality);
    }

    if args.visualize && !annotation_qualities.is_empty() {
        let vis = create_visualization(&img.to_rgb8(), &edge_data, annotations, &annotation_qualities, img_info);
        let vis_path = args.output.join(format!("vis_{}", img_info.file_name));
        vis.save(&vis_path)?;
    }

    let avg_score = if !annotation_qualities.is_empty() {
        annotation_qualities.iter().map(|a| a.edge_alignment_score).sum::<f64>()
            / annotation_qualities.len() as f64
    } else {
        1.0
    };

    let avg_distance_score = if !annotation_qualities.is_empty() {
        annotation_qualities.iter().map(|a| a.distance_score).sum::<f64>()
            / annotation_qualities.len() as f64
    } else {
        1.0
    };

    Ok(ImageQuality {
        image_id: img_info.id,
        file_name: img_info.file_name.clone(),
        avg_alignment_score: avg_score,
        avg_distance_score,
        annotation_count: annotation_qualities.len(),
        annotations: annotation_qualities,
    })
}

fn compute_edge_data(gray: &GrayImage, args: &Args) -> EdgeData {
    // Compute gradient magnitude (preserve strength)
    let gx = horizontal_sobel(gray);
    let gy = vertical_sobel(gray);
    let (width, height) = gray.dimensions();

    let mut gradient_magnitude = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let gx_val = gx.get_pixel(x, y)[0] as f64;
            let gy_val = gy.get_pixel(x, y)[0] as f64;
            let magnitude = ((gx_val * gx_val + gy_val * gy_val).sqrt()).min(255.0) as u8;
            gradient_magnitude.put_pixel(x, y, Luma([magnitude]));
        }
    }

    // Compute binary edges
    let binary_edges = match args.method.as_str() {
        "canny" => canny(gray, args.canny_low, args.canny_high),
        "sobel" => {
            let mut edges = gradient_magnitude.clone();
            for pixel in edges.pixels_mut() {
                pixel[0] = if pixel[0] > args.gradient_threshold { 255 } else { 0 };
            }
            edges
        }
        "both" | _ => {
            let canny_edges = canny(gray, args.canny_low, args.canny_high);
            let mut sobel_edges = gradient_magnitude.clone();
            for pixel in sobel_edges.pixels_mut() {
                pixel[0] = if pixel[0] > args.gradient_threshold { 255 } else { 0 };
            }
            combine_edges(&canny_edges, &sobel_edges)
        }
    };

    // Compute distance transform (distance to nearest edge)
    // Invert: we want distance TO edges, not FROM edges
    let mut inverted = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let edge_val = binary_edges.get_pixel(x, y)[0];
            inverted.put_pixel(x, y, Luma([if edge_val > 0 { 0 } else { 255 }]));
        }
    }
    let distance_map = distance_transform(&inverted, Norm::L2);

    EdgeData {
        binary_edges,
        gradient_magnitude,
        distance_map,
    }
}

fn combine_edges(canny: &GrayImage, sobel: &GrayImage) -> GrayImage {
    let (width, height) = canny.dimensions();
    let mut result = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let c = canny.get_pixel(x, y)[0];
            let s = sobel.get_pixel(x, y)[0];
            result.put_pixel(x, y, Luma([c.max(s)]));
        }
    }

    result
}

fn evaluate_annotation(
    ann: &CocoAnnotation,
    edge_data: &EdgeData,
    width: u32,
    height: u32,
    args: &Args,
    category: &str,
) -> AnnotationQuality {
    let mut issues = Vec::new();

    let mask_boundary = extract_boundary_from_annotation(ann, width, height);

    if mask_boundary.is_empty() {
        return AnnotationQuality {
            annotation_id: ann.id,
            image_id: ann.image_id,
            category: category.to_string(),
            edge_alignment_score: 0.0,
            boundary_precision: 0.0,
            boundary_recall: 0.0,
            distance_score: 0.0,
            distance_stats: DistanceStats::default(),
            adjusted_precision: 0.0,
            adjusted_recall: 0.0,
            edge_confidence_ratio: 0.0,
            avg_edge_strength: 0.0,
            is_textured: false,
            is_low_contrast: false,
            is_truncated: false,
            is_small: true,
            evaluation_confidence: "Low".to_string(),
            issues: vec!["Could not extract boundary from annotation".to_string()],
        };
    }

    // Collect distances from each boundary point to nearest edge
    let distances: Vec<f64> = mask_boundary
        .iter()
        .map(|&(x, y)| edge_data.distance_map.get_pixel(x, y)[0] as f64)
        .collect();

    // Compute distance statistics
    let distance_stats = compute_distance_stats(&distances, args.distance_threshold as f64);

    // Compute distance-based score using exponential decay
    let distance_score = compute_distance_score(&distances, args.scoring_sigma);

    // Analyze edge confidence (how many boundary points have detectable gradient)
    let (edge_confidence_ratio, avg_edge_strength, high_conf_points, _low_conf_points) =
        analyze_edge_confidence(&mask_boundary, &edge_data.gradient_magnitude, args.gradient_threshold);

    // Detect if annotation is at image boundary (truncated)
    let (is_truncated, at_image_edge_count) =
        analyze_image_boundary(&mask_boundary, width, height, args.distance_threshold);

    // Detect texture (many edges inside the bounding box relative to boundary)
    let (is_textured, boundary_edge_count, internal_edge_count) =
        analyze_texture(ann, &mask_boundary, &edge_data.binary_edges, args.distance_threshold);

    // Size reliability
    let is_small = mask_boundary.len() < args.min_boundary_points;

    // Low contrast detection
    let is_low_contrast = edge_confidence_ratio < 0.4;

    // Original binary precision (for backward compatibility)
    let boundary_near_edge = distances.iter().filter(|&&d| d <= args.distance_threshold as f64).count();
    let precision = boundary_near_edge as f64 / mask_boundary.len() as f64;

    // Adjusted precision: only count high-confidence points (exclude low-gradient and image-edge points)
    let adjusted_precision = if high_conf_points > 0 {
        let adjusted_near = mask_boundary.iter().enumerate()
            .filter(|(i, &(x, y))| {
                let gradient = edge_data.gradient_magnitude.get_pixel(x, y)[0];
                let at_edge = is_at_image_boundary(x, y, width, height, args.distance_threshold);
                gradient >= args.gradient_threshold && !at_edge && distances[*i] <= args.distance_threshold as f64
            })
            .count();
        adjusted_near as f64 / high_conf_points as f64
    } else {
        precision
    };

    // Original recall
    let edge_points = get_edge_points_near_bbox(&edge_data.binary_edges, &ann.bbox, args.distance_threshold);
    let edges_near_boundary = edge_points.iter()
        .filter(|&&(ex, ey)| is_near_boundary(ex, ey, &mask_boundary, args.distance_threshold))
        .count();
    let recall = if !edge_points.is_empty() {
        edges_near_boundary as f64 / edge_points.len() as f64
    } else {
        1.0
    };

    // Adjusted recall: only count boundary-relevant edges (not internal texture)
    let adjusted_recall = if boundary_edge_count > 0 {
        let boundary_edges_captured = edge_points.iter()
            .filter(|&&(ex, ey)| {
                is_near_boundary(ex, ey, &mask_boundary, args.distance_threshold) &&
                min_distance_to_points(ex, ey, &mask_boundary) <= (args.distance_threshold * 2) as f64
            })
            .count();
        (boundary_edges_captured as f64 / boundary_edge_count as f64).min(1.0)
    } else {
        recall
    };

    // F1-like alignment score
    let alignment_score = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    // Determine evaluation confidence
    let confidence_factors = [is_textured, is_low_contrast, is_truncated, is_small];
    let factor_count = confidence_factors.iter().filter(|&&f| f).count();
    let evaluation_confidence = match factor_count {
        0 => "High",
        1 => "Medium",
        _ => "Low",
    }.to_string();

    // Generate issues
    if is_low_contrast {
        issues.push(format!(
            "Low edge contrast on {:.0}% of boundary - edges may not be detectable",
            (1.0 - edge_confidence_ratio) * 100.0
        ));
    }
    if is_textured {
        issues.push(format!(
            "Textured object ({} internal vs {} boundary edges) - recall uses boundary edges only",
            internal_edge_count, boundary_edge_count
        ));
    }
    if is_truncated {
        issues.push(format!(
            "Annotation touches image boundary ({} points) - may be truncated",
            at_image_edge_count
        ));
    }
    if is_small {
        issues.push(format!(
            "Small annotation ({} boundary points < {}) - high statistical variance",
            mask_boundary.len(), args.min_boundary_points
        ));
    }
    if distance_stats.median > args.distance_threshold as f64 {
        issues.push(format!(
            "Median distance to edge is {:.1}px (threshold: {}px)",
            distance_stats.median, args.distance_threshold
        ));
    }
    if distance_stats.p90 > (args.distance_threshold * 3) as f64 {
        issues.push(format!(
            "10% of boundary points are >{:.1}px from edges",
            distance_stats.p90
        ));
    }

    AnnotationQuality {
        annotation_id: ann.id,
        image_id: ann.image_id,
        category: category.to_string(),
        edge_alignment_score: alignment_score,
        boundary_precision: precision,
        boundary_recall: recall,
        distance_score,
        distance_stats,
        adjusted_precision,
        adjusted_recall,
        edge_confidence_ratio,
        avg_edge_strength,
        is_textured,
        is_low_contrast,
        is_truncated,
        is_small,
        evaluation_confidence,
        issues,
    }
}

fn compute_distance_stats(distances: &[f64], threshold: f64) -> DistanceStats {
    if distances.is_empty() {
        return DistanceStats::default();
    }

    let mut sorted = distances.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let sum: f64 = sorted.iter().sum();
    let mean = sum / n as f64;

    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };

    let variance: f64 = sorted.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    let percentile = |p: f64| -> f64 {
        let idx = p * (n - 1) as f64;
        let lower = idx.floor() as usize;
        let upper = (lower + 1).min(n - 1);
        let frac = idx - lower as f64;
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    };

    let count_within = |max_dist: f64| -> f64 {
        sorted.iter().filter(|&&d| d <= max_dist).count() as f64 / n as f64
    };

    DistanceStats {
        mean,
        median,
        std_dev,
        min: sorted[0],
        max: sorted[n - 1],
        p90: percentile(0.90),
        pct_within_1px: count_within(1.0),
        pct_within_2px: count_within(2.0),
        pct_within_threshold: count_within(threshold),
        sample_size: n,
    }
}

fn compute_distance_score(distances: &[f64], sigma: f64) -> f64 {
    if distances.is_empty() {
        return 0.0;
    }

    let scores: Vec<f64> = distances
        .iter()
        .map(|&d| (-d.powi(2) / (2.0 * sigma.powi(2))).exp())
        .collect();

    scores.iter().sum::<f64>() / scores.len() as f64
}

fn analyze_edge_confidence(
    boundary: &[(u32, u32)],
    gradient: &GrayImage,
    threshold: u8,
) -> (f64, f64, usize, usize) {
    let mut high_conf = 0;
    let mut low_conf = 0;
    let mut total_strength: f64 = 0.0;

    for &(x, y) in boundary {
        let strength = gradient.get_pixel(x, y)[0];
        total_strength += strength as f64;
        if strength >= threshold {
            high_conf += 1;
        } else {
            low_conf += 1;
        }
    }

    let ratio = if !boundary.is_empty() {
        high_conf as f64 / boundary.len() as f64
    } else {
        0.0
    };

    let avg_strength = if !boundary.is_empty() {
        total_strength / boundary.len() as f64
    } else {
        0.0
    };

    (ratio, avg_strength, high_conf, low_conf)
}

fn analyze_image_boundary(
    boundary: &[(u32, u32)],
    width: u32,
    height: u32,
    margin: u32,
) -> (bool, usize) {
    let at_edge_count = boundary
        .iter()
        .filter(|&&(x, y)| is_at_image_boundary(x, y, width, height, margin))
        .count();

    let is_truncated = at_edge_count as f64 / boundary.len() as f64 > 0.1;
    (is_truncated, at_edge_count)
}

fn is_at_image_boundary(x: u32, y: u32, width: u32, height: u32, margin: u32) -> bool {
    x < margin || x >= width.saturating_sub(margin) ||
    y < margin || y >= height.saturating_sub(margin)
}

fn analyze_texture(
    ann: &CocoAnnotation,
    mask_boundary: &[(u32, u32)],
    edges: &GrayImage,
    threshold: u32,
) -> (bool, usize, usize) {
    let all_edges = get_edge_points_near_bbox(edges, &ann.bbox, threshold);

    let boundary_zone = threshold * 2;
    let mut boundary_edges = 0;
    let mut internal_edges = 0;

    for &(ex, ey) in &all_edges {
        let dist = min_distance_to_points(ex, ey, mask_boundary);
        if dist <= boundary_zone as f64 {
            boundary_edges += 1;
        } else {
            internal_edges += 1;
        }
    }

    let texture_ratio = if boundary_edges > 0 {
        internal_edges as f64 / boundary_edges as f64
    } else {
        0.0
    };

    let is_textured = texture_ratio > 2.0;
    (is_textured, boundary_edges, internal_edges)
}

fn min_distance_to_points(x: u32, y: u32, points: &[(u32, u32)]) -> f64 {
    points
        .iter()
        .map(|&(px, py)| {
            let dx = x as f64 - px as f64;
            let dy = y as f64 - py as f64;
            (dx * dx + dy * dy).sqrt()
        })
        .fold(f64::MAX, f64::min)
}

fn extract_boundary_from_annotation(ann: &CocoAnnotation, width: u32, height: u32) -> Vec<(u32, u32)> {
    match &ann.segmentation {
        Segmentation::Polygon(polygons) => {
            let mut boundary_points = Vec::new();
            for polygon in polygons {
                let points: Vec<(f64, f64)> = polygon
                    .chunks(2)
                    .filter(|c| c.len() == 2)
                    .map(|c| (c[0], c[1]))
                    .collect();

                for i in 0..points.len() {
                    let (x1, y1) = points[i];
                    let (x2, y2) = points[(i + 1) % points.len()];

                    let dist = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
                    let steps = (dist as usize).max(1);

                    for step in 0..steps {
                        let t = step as f64 / steps as f64;
                        let x = (x1 + t * (x2 - x1)).round() as i32;
                        let y = (y1 + t * (y2 - y1)).round() as i32;

                        if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                            boundary_points.push((x as u32, y as u32));
                        }
                    }
                }
            }
            boundary_points
        }
        Segmentation::Rle(rle) => {
            let mask = decode_rle(rle, width, height);
            extract_boundary_from_mask(&mask)
        }
    }
}

fn decode_rle(rle: &RleSegmentation, width: u32, height: u32) -> GrayImage {
    let mut mask = GrayImage::new(width, height);

    let counts: Vec<u32> = match &rle.counts {
        RleCounts::Raw(v) => v.clone(),
        RleCounts::Encoded(s) => decode_rle_string(s),
    };

    let mut pos = 0u32;
    let mut value = 0u8;

    for count in counts {
        for _ in 0..count {
            let x = pos / height;
            let y = pos % height;
            if x < width && y < height {
                mask.put_pixel(x, y, Luma([value]));
            }
            pos += 1;
        }
        value = if value == 0 { 255 } else { 0 };
    }

    mask
}

fn decode_rle_string(s: &str) -> Vec<u32> {
    let mut counts = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let mut count = 0u32;
        let mut shift = 0;

        loop {
            if i >= bytes.len() {
                break;
            }
            let b = bytes[i] as u32 - 48;
            i += 1;
            count |= (b & 0x1f) << shift;
            shift += 5;
            if b < 32 {
                break;
            }
        }

        counts.push(count);
    }

    counts
}

fn extract_boundary_from_mask(mask: &GrayImage) -> Vec<(u32, u32)> {
    let (width, height) = mask.dimensions();
    let mut boundary = Vec::new();

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            if mask.get_pixel(x, y)[0] > 0 {
                let is_boundary = mask.get_pixel(x - 1, y)[0] == 0
                    || mask.get_pixel(x + 1, y)[0] == 0
                    || mask.get_pixel(x, y - 1)[0] == 0
                    || mask.get_pixel(x, y + 1)[0] == 0;

                if is_boundary {
                    boundary.push((x, y));
                }
            }
        }
    }

    boundary
}

fn is_near_boundary(x: u32, y: u32, boundary: &[(u32, u32)], threshold: u32) -> bool {
    let t_sq = (threshold * threshold) as i64;

    for &(bx, by) in boundary {
        let dx = x as i64 - bx as i64;
        let dy = y as i64 - by as i64;
        if dx * dx + dy * dy <= t_sq {
            return true;
        }
    }

    false
}

fn get_edge_points_near_bbox(edges: &GrayImage, bbox: &[f64], padding: u32) -> Vec<(u32, u32)> {
    if bbox.len() < 4 {
        return Vec::new();
    }

    let (width, height) = edges.dimensions();
    let x1 = (bbox[0] as i32 - padding as i32).max(0) as u32;
    let y1 = (bbox[1] as i32 - padding as i32).max(0) as u32;
    let x2 = ((bbox[0] + bbox[2]) as u32 + padding).min(width - 1);
    let y2 = ((bbox[1] + bbox[3]) as u32 + padding).min(height - 1);

    let mut points = Vec::new();
    for y in y1..=y2 {
        for x in x1..=x2 {
            if edges.get_pixel(x, y)[0] > 0 {
                points.push((x, y));
            }
        }
    }

    points
}

fn create_visualization(
    img: &RgbImage,
    edge_data: &EdgeData,
    annotations: &[CocoAnnotation],
    qualities: &[AnnotationQuality],
    _img_info: &CocoImage,
) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut vis: RgbImage = ImageBuffer::new(width * 2, height);

    // Left side: original with color-coded boundaries
    for y in 0..height {
        for x in 0..width {
            vis.put_pixel(x, y, *img.get_pixel(x, y));
        }
    }

    // Draw annotation boundaries color-coded by distance score
    for (ann, quality) in annotations.iter().zip(qualities.iter()) {
        let boundary = extract_boundary_from_annotation(ann, width, height);
        let color = score_to_color(quality.distance_score);
        for (x, y) in boundary {
            vis.put_pixel(x, y, color);
        }
    }

    // Right side: gradient magnitude with edges overlay
    for y in 0..height {
        for x in 0..width {
            let grad = edge_data.gradient_magnitude.get_pixel(x, y)[0];
            let edge = edge_data.binary_edges.get_pixel(x, y)[0];
            if edge > 0 {
                vis.put_pixel(width + x, y, Rgb([255, 255, 255]));
            } else {
                vis.put_pixel(width + x, y, Rgb([grad / 2, grad / 2, grad / 2]));
            }
        }
    }

    // Overlay annotations in colors on edge view
    for (ann, quality) in annotations.iter().zip(qualities.iter()) {
        let boundary = extract_boundary_from_annotation(ann, width, height);
        let color = score_to_color(quality.distance_score);
        for (x, y) in boundary {
            vis.put_pixel(width + x, y, color);
        }
    }

    vis
}

fn score_to_color(score: f64) -> Rgb<u8> {
    if score >= 0.8 {
        Rgb([0, 255, 0])      // Green: excellent
    } else if score >= 0.6 {
        Rgb([128, 255, 0])    // Yellow-green: good
    } else if score >= 0.4 {
        Rgb([255, 255, 0])    // Yellow: moderate
    } else if score >= 0.2 {
        Rgb([255, 128, 0])    // Orange: poor
    } else {
        Rgb([255, 0, 0])      // Red: bad
    }
}
