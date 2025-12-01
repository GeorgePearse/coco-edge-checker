use anyhow::{Context, Result};
use clap::Parser;
use image::{GrayImage, Luma};
use imageproc::distance_transform::{distance_transform, Norm};
use imageproc::edges::canny;
use imageproc::gradients::{horizontal_sobel, vertical_sobel};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "coco-edge-checker")]
#[command(about = "Auto-correct COCO segmentation annotations by snapping to detected edges")]
struct Args {
    /// Path to COCO annotations JSON file
    #[arg(short, long)]
    annotations: PathBuf,

    /// Path to images directory
    #[arg(short, long)]
    images: PathBuf,

    /// Output corrected COCO JSON file
    #[arg(short, long)]
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

    /// Gradient threshold for edge detection (0-255)
    #[arg(long, default_value = "30")]
    gradient_threshold: u8,

    /// Max distance to snap polygon vertices (pixels)
    #[arg(long, default_value = "5")]
    snap_distance: u32,

    /// Min gradient improvement to trigger snap (0-255)
    #[arg(long, default_value = "20")]
    min_gradient_improvement: u8,

    /// Max allowed area change (fraction, e.g. 0.1 = 10%)
    #[arg(long, default_value = "0.1")]
    max_area_change: f64,

    /// Number of images to process (0 = all)
    #[arg(short, long, default_value = "0")]
    limit: usize,

    /// Output corrections report JSON file
    #[arg(long)]
    report: Option<PathBuf>,
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
    #[allow(dead_code)]
    binary_edges: GrayImage,
    gradient_magnitude: GrayImage,
    #[allow(dead_code)]
    distance_map: GrayImage,
}

// Snapping configuration
#[derive(Clone)]
struct SnapConfig {
    snap_distance: u32,
    min_gradient_improvement: u8,
    max_area_change: f64,
}

// Statistics for a single polygon correction
#[derive(Debug, Serialize, Clone, Default)]
struct PolygonCorrection {
    points_snapped: usize,
    points_unchanged: usize,
    avg_snap_distance: f64,
    max_snap_distance: f64,
    area_change_pct: f64,
}

// Correction info for a single annotation
#[derive(Debug, Serialize)]
struct AnnotationCorrection {
    annotation_id: u64,
    image_id: u64,
    category: String,
    was_modified: bool,
    polygon_stats: PolygonCorrection,
}

// Full correction report
#[derive(Debug, Serialize)]
struct CorrectionReport {
    total_annotations: usize,
    annotations_modified: usize,
    annotations_unchanged: usize,
    annotations_skipped_rle: usize,
    avg_points_snapped: f64,
    avg_snap_distance: f64,
    corrections: Vec<AnnotationCorrection>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading COCO annotations from {:?}...", args.annotations);
    let file = File::open(&args.annotations)
        .with_context(|| format!("Failed to open annotations file: {:?}", args.annotations))?;
    let reader = BufReader::new(file);
    let mut dataset: CocoDataset = serde_json::from_reader(reader)
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

    let _image_map: HashMap<u64, &CocoImage> = dataset
        .images
        .iter()
        .map(|img| (img.id, img))
        .collect();

    let snap_config = SnapConfig {
        snap_distance: args.snap_distance,
        min_gradient_improvement: args.min_gradient_improvement,
        max_area_change: args.max_area_change,
    };

    // Process images and build edge data cache
    let images_to_process: Vec<_> = if args.limit > 0 {
        dataset.images.iter().take(args.limit).collect()
    } else {
        dataset.images.iter().collect()
    };

    println!("Computing edge data for {} images...", images_to_process.len());
    let pb = ProgressBar::new(images_to_process.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("#>-"),
    );

    // Build edge data for each image (in parallel)
    let edge_data_map: HashMap<u64, EdgeData> = images_to_process
        .par_iter()
        .filter_map(|img| {
            let img_path = args.images.join(&img.file_name);
            match image::open(&img_path) {
                Ok(loaded_img) => {
                    let gray = loaded_img.to_luma8();
                    let edge_data = compute_edge_data(&gray, &args);
                    pb.inc(1);
                    Some((img.id, edge_data))
                }
                Err(_) => {
                    pb.inc(1);
                    None
                }
            }
        })
        .collect();

    pb.finish_with_message("Edge detection complete");

    // Process annotations
    println!("Correcting annotations...");
    let mut corrections = Vec::new();
    let mut annotations_modified = 0;
    let mut annotations_unchanged = 0;
    let mut annotations_skipped_rle = 0;
    let mut total_points_snapped = 0;
    let mut total_snap_distances = Vec::new();

    for ann in dataset.annotations.iter_mut() {
        let category = category_map
            .get(&ann.category_id)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        // Skip if we don't have edge data for this image
        let edge_data = match edge_data_map.get(&ann.image_id) {
            Some(ed) => ed,
            None => {
                annotations_unchanged += 1;
                continue;
            }
        };

        // Try to correct the segmentation
        match correct_annotation_segmentation(&ann.segmentation, edge_data, &snap_config) {
            Some((corrected_seg, stats)) => {
                let was_modified = stats.points_snapped > 0;

                if was_modified {
                    ann.segmentation = corrected_seg;
                    annotations_modified += 1;
                    total_points_snapped += stats.points_snapped;
                    if stats.avg_snap_distance > 0.0 {
                        total_snap_distances.push(stats.avg_snap_distance);
                    }
                } else {
                    annotations_unchanged += 1;
                }

                corrections.push(AnnotationCorrection {
                    annotation_id: ann.id,
                    image_id: ann.image_id,
                    category,
                    was_modified,
                    polygon_stats: stats,
                });
            }
            None => {
                // RLE segmentation - skip
                annotations_skipped_rle += 1;
                corrections.push(AnnotationCorrection {
                    annotation_id: ann.id,
                    image_id: ann.image_id,
                    category,
                    was_modified: false,
                    polygon_stats: PolygonCorrection::default(),
                });
            }
        }
    }

    // Write corrected COCO JSON
    println!("Writing corrected annotations to {:?}...", args.output);
    let output_file = File::create(&args.output)
        .with_context(|| format!("Failed to create output file: {:?}", args.output))?;
    let writer = BufWriter::new(output_file);
    serde_json::to_writer_pretty(writer, &dataset)
        .with_context(|| "Failed to write corrected COCO JSON")?;

    // Write report if requested
    if let Some(report_path) = &args.report {
        let avg_points_snapped = if annotations_modified > 0 {
            total_points_snapped as f64 / annotations_modified as f64
        } else {
            0.0
        };

        let avg_snap_distance = if !total_snap_distances.is_empty() {
            total_snap_distances.iter().sum::<f64>() / total_snap_distances.len() as f64
        } else {
            0.0
        };

        let report = CorrectionReport {
            total_annotations: dataset.annotations.len(),
            annotations_modified,
            annotations_unchanged,
            annotations_skipped_rle,
            avg_points_snapped,
            avg_snap_distance,
            corrections,
        };

        let report_file = File::create(report_path)
            .with_context(|| format!("Failed to create report file: {:?}", report_path))?;
        serde_json::to_writer_pretty(report_file, &report)
            .with_context(|| "Failed to write report JSON")?;
        println!("Report saved to: {:?}", report_path);
    }

    println!("\n=== Correction Summary ===");
    println!("Total annotations: {}", dataset.annotations.len());
    println!("Annotations modified: {}", annotations_modified);
    println!("Annotations unchanged: {}", annotations_unchanged);
    println!("Annotations skipped (RLE): {}", annotations_skipped_rle);
    if annotations_modified > 0 {
        println!(
            "Average points snapped per annotation: {:.1}",
            total_points_snapped as f64 / annotations_modified as f64
        );
        if !total_snap_distances.is_empty() {
            println!(
                "Average snap distance: {:.2} px",
                total_snap_distances.iter().sum::<f64>() / total_snap_distances.len() as f64
            );
        }
    }
    println!("Output saved to: {:?}", args.output);

    Ok(())
}

fn compute_edge_data(gray: &GrayImage, args: &Args) -> EdgeData {
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

// ============= EDGE SNAPPING FUNCTIONS =============

/// Compute signed area of polygon using shoelace formula
fn polygon_area(coords: &[f64]) -> f64 {
    if coords.len() < 6 {
        return 0.0;
    }

    let n = coords.len() / 2;
    let mut area = 0.0;

    for i in 0..n {
        let x1 = coords[i * 2];
        let y1 = coords[i * 2 + 1];
        let x2 = coords[((i + 1) % n) * 2];
        let y2 = coords[((i + 1) % n) * 2 + 1];
        area += x1 * y2 - x2 * y1;
    }

    area / 2.0
}

/// Find the best edge pixel to snap to within max_distance
fn find_snap_target(
    x: f64,
    y: f64,
    edge_data: &EdgeData,
    max_distance: u32,
    min_gradient_improvement: u8,
) -> Option<(f64, f64, f64)> {
    let (width, height) = edge_data.gradient_magnitude.dimensions();
    let ix = x.round() as i32;
    let iy = y.round() as i32;

    let current_grad = if ix >= 0 && ix < width as i32 && iy >= 0 && iy < height as i32 {
        edge_data.gradient_magnitude.get_pixel(ix as u32, iy as u32)[0]
    } else {
        0
    };

    let mut best_target: Option<(f64, f64, u8)> = None;
    let mut best_score = 0.0f64;

    let search_range = max_distance as i32;

    for dy in -search_range..=search_range {
        for dx in -search_range..=search_range {
            let nx = ix + dx;
            let ny = iy + dy;

            if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                continue;
            }

            let dist_sq = (dx * dx + dy * dy) as f64;
            if dist_sq > (max_distance * max_distance) as f64 {
                continue;
            }

            let grad = edge_data.gradient_magnitude.get_pixel(nx as u32, ny as u32)[0];

            if grad < current_grad.saturating_add(min_gradient_improvement) {
                continue;
            }

            let dist = dist_sq.sqrt();
            let score = grad as f64 / (1.0 + dist);

            if score > best_score {
                best_score = score;
                best_target = Some((nx as f64, ny as f64, grad));
            }
        }
    }

    best_target.map(|(tx, ty, g)| (tx, ty, g as f64))
}

/// Correct a polygon by snapping vertices to nearby strong edges
fn correct_polygon(
    coords: &[f64],
    edge_data: &EdgeData,
    config: &SnapConfig,
) -> (Vec<f64>, PolygonCorrection) {
    if coords.len() < 6 {
        return (coords.to_vec(), PolygonCorrection::default());
    }

    let original_area = polygon_area(coords).abs();
    let mut corrected = coords.to_vec();
    let mut snapped_count = 0;
    let mut snap_distances = Vec::new();

    let n_points = coords.len() / 2;

    for i in 0..n_points {
        let x = coords[i * 2];
        let y = coords[i * 2 + 1];

        if let Some((new_x, new_y, _grad)) = find_snap_target(
            x, y,
            edge_data,
            config.snap_distance,
            config.min_gradient_improvement,
        ) {
            let dist = ((new_x - x).powi(2) + (new_y - y).powi(2)).sqrt();
            corrected[i * 2] = new_x;
            corrected[i * 2 + 1] = new_y;
            snapped_count += 1;
            snap_distances.push(dist);
        }
    }

    let new_area = polygon_area(&corrected).abs();
    let area_change = if original_area > 0.0 {
        ((new_area - original_area) / original_area).abs()
    } else {
        0.0
    };

    if area_change > config.max_area_change {
        return (coords.to_vec(), PolygonCorrection {
            points_snapped: 0,
            points_unchanged: n_points,
            avg_snap_distance: 0.0,
            max_snap_distance: 0.0,
            area_change_pct: 0.0,
        });
    }

    let avg_snap = if !snap_distances.is_empty() {
        snap_distances.iter().sum::<f64>() / snap_distances.len() as f64
    } else {
        0.0
    };

    let max_snap = snap_distances.iter().cloned().fold(0.0f64, f64::max);

    (corrected, PolygonCorrection {
        points_snapped: snapped_count,
        points_unchanged: n_points - snapped_count,
        avg_snap_distance: avg_snap,
        max_snap_distance: max_snap,
        area_change_pct: area_change * 100.0,
    })
}

/// Correct all polygons in an annotation's segmentation
fn correct_annotation_segmentation(
    segmentation: &Segmentation,
    edge_data: &EdgeData,
    config: &SnapConfig,
) -> Option<(Segmentation, PolygonCorrection)> {
    match segmentation {
        Segmentation::Polygon(polygons) => {
            let mut corrected_polygons = Vec::new();
            let mut total_stats = PolygonCorrection::default();
            let mut total_points = 0;
            let mut total_snapped = 0;
            let mut all_distances = Vec::new();

            for polygon in polygons {
                let (corrected, stats) = correct_polygon(polygon, edge_data, config);
                corrected_polygons.push(corrected);
                total_snapped += stats.points_snapped;
                total_points += stats.points_snapped + stats.points_unchanged;
                if stats.max_snap_distance > total_stats.max_snap_distance {
                    total_stats.max_snap_distance = stats.max_snap_distance;
                }
                if stats.points_snapped > 0 {
                    all_distances.push(stats.avg_snap_distance);
                }
            }

            total_stats.points_snapped = total_snapped;
            total_stats.points_unchanged = total_points - total_snapped;
            total_stats.avg_snap_distance = if !all_distances.is_empty() {
                all_distances.iter().sum::<f64>() / all_distances.len() as f64
            } else {
                0.0
            };

            Some((Segmentation::Polygon(corrected_polygons), total_stats))
        }
        Segmentation::Rle(_) => None,
    }
}
