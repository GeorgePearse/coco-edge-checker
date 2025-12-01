use anyhow::{Context, Result};
use clap::Parser;
use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
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

    /// Number of images to process (0 = all)
    #[arg(short, long, default_value = "0")]
    limit: usize,

    /// Generate visualization images
    #[arg(long)]
    visualize: bool,
}

// COCO Format Structures
#[derive(Debug, Deserialize)]
struct CocoDataset {
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnnotation>,
    categories: Vec<CocoCategory>,
}

#[derive(Debug, Deserialize, Clone)]
struct CocoImage {
    id: u64,
    file_name: String,
    width: u32,
    height: u32,
}

#[derive(Debug, Deserialize, Clone)]
struct CocoAnnotation {
    id: u64,
    image_id: u64,
    category_id: u64,
    segmentation: Segmentation,
    area: f64,
    bbox: Vec<f64>,
    iscrowd: u8,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum Segmentation {
    Polygon(Vec<Vec<f64>>),
    Rle(RleSegmentation),
}

#[derive(Debug, Deserialize, Clone)]
struct RleSegmentation {
    counts: RleCounts,
    size: Vec<u32>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum RleCounts {
    Encoded(String),
    Raw(Vec<u32>),
}

#[derive(Debug, Deserialize, Clone)]
struct CocoCategory {
    id: u64,
    name: String,
}

// Quality metrics for an annotation
#[derive(Debug, Serialize)]
struct AnnotationQuality {
    annotation_id: u64,
    image_id: u64,
    category: String,
    edge_alignment_score: f64,
    boundary_precision: f64,
    boundary_recall: f64,
    issues: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ImageQuality {
    image_id: u64,
    file_name: String,
    avg_alignment_score: f64,
    annotation_count: usize,
    annotations: Vec<AnnotationQuality>,
}

#[derive(Debug, Serialize)]
struct Report {
    total_images: usize,
    total_annotations: usize,
    avg_alignment_score: f64,
    low_quality_count: usize,
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

    // Build lookup maps
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

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    // Select images to process
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

    // Process images in parallel
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

    // Compile report
    let valid_results: Vec<ImageQuality> = image_results.into_iter().flatten().collect();

    let total_annotations: usize = valid_results.iter().map(|r| r.annotation_count).sum();
    let avg_score: f64 = if !valid_results.is_empty() {
        valid_results.iter().map(|r| r.avg_alignment_score).sum::<f64>() / valid_results.len() as f64
    } else {
        0.0
    };

    let low_quality_count = valid_results
        .iter()
        .flat_map(|r| &r.annotations)
        .filter(|a| a.edge_alignment_score < 0.5)
        .count();

    let report = Report {
        total_images: valid_results.len(),
        total_annotations,
        avg_alignment_score: avg_score,
        low_quality_count,
        images: valid_results,
    };

    // Write report
    let report_path = args.output.join("quality_report.json");
    let report_file = File::create(&report_path)?;
    serde_json::to_writer_pretty(report_file, &report)?;

    println!("\n=== Quality Report ===");
    println!("Images processed: {}", report.total_images);
    println!("Annotations checked: {}", report.total_annotations);
    println!("Average alignment score: {:.2}%", report.avg_alignment_score * 100.0);
    println!("Low quality annotations (<50%): {}", report.low_quality_count);
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

    // Run edge detection
    let edges = match args.method.as_str() {
        "canny" => detect_edges_canny(&gray, args.canny_low, args.canny_high),
        "sobel" => detect_edges_sobel(&gray),
        "both" | _ => {
            let canny_edges = detect_edges_canny(&gray, args.canny_low, args.canny_high);
            let sobel_edges = detect_edges_sobel(&gray);
            combine_edges(&canny_edges, &sobel_edges)
        }
    };

    let annotations = annotations.unwrap_or(&[]);
    let mut annotation_qualities = Vec::new();

    for ann in annotations.iter().filter(|a| a.iscrowd == 0) {
        let category = category_map
            .get(&ann.category_id)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        let quality = evaluate_annotation(
            ann,
            &edges,
            img_info.width,
            img_info.height,
            args.distance_threshold,
            &category,
        );
        annotation_qualities.push(quality);
    }

    // Generate visualization if requested
    if args.visualize && !annotation_qualities.is_empty() {
        let vis = create_visualization(&img.to_rgb8(), &edges, annotations, img_info);
        let vis_path = args.output.join(format!("vis_{}", img_info.file_name));
        vis.save(&vis_path)?;
    }

    let avg_score = if !annotation_qualities.is_empty() {
        annotation_qualities
            .iter()
            .map(|a| a.edge_alignment_score)
            .sum::<f64>()
            / annotation_qualities.len() as f64
    } else {
        1.0
    };

    Ok(ImageQuality {
        image_id: img_info.id,
        file_name: img_info.file_name.clone(),
        avg_alignment_score: avg_score,
        annotation_count: annotation_qualities.len(),
        annotations: annotation_qualities,
    })
}

fn detect_edges_canny(gray: &GrayImage, low: f32, high: f32) -> GrayImage {
    canny(gray, low, high)
}

fn detect_edges_sobel(gray: &GrayImage) -> GrayImage {
    let gx = horizontal_sobel(gray);
    let gy = vertical_sobel(gray);
    let (width, height) = gray.dimensions();
    let mut result = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let gx_val = gx.get_pixel(x, y)[0] as f64;
            let gy_val = gy.get_pixel(x, y)[0] as f64;
            let magnitude = ((gx_val * gx_val + gy_val * gy_val).sqrt()).min(255.0) as u8;
            result.put_pixel(x, y, Luma([magnitude]));
        }
    }

    // Apply threshold to create binary edge map
    let threshold = 30u8;
    for pixel in result.pixels_mut() {
        pixel[0] = if pixel[0] > threshold { 255 } else { 0 };
    }

    result
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
    edges: &GrayImage,
    width: u32,
    height: u32,
    distance_threshold: u32,
    category: &str,
) -> AnnotationQuality {
    let mut issues = Vec::new();

    // Extract mask boundary from annotation
    let mask_boundary = extract_boundary_from_annotation(ann, width, height);

    if mask_boundary.is_empty() {
        return AnnotationQuality {
            annotation_id: ann.id,
            image_id: ann.image_id,
            category: category.to_string(),
            edge_alignment_score: 0.0,
            boundary_precision: 0.0,
            boundary_recall: 0.0,
            issues: vec!["Could not extract boundary from annotation".to_string()],
        };
    }

    // Calculate precision: what fraction of mask boundary points are near image edges?
    let mut boundary_near_edge = 0;
    for &(x, y) in &mask_boundary {
        if is_near_edge(x, y, edges, distance_threshold) {
            boundary_near_edge += 1;
        }
    }
    let precision = boundary_near_edge as f64 / mask_boundary.len() as f64;

    // Calculate recall: what fraction of image edges near the mask are captured?
    let edge_points = get_edge_points_near_bbox(edges, &ann.bbox, distance_threshold);
    let mut edges_near_boundary = 0;
    for &(ex, ey) in &edge_points {
        if is_near_boundary(ex, ey, &mask_boundary, distance_threshold) {
            edges_near_boundary += 1;
        }
    }
    let recall = if !edge_points.is_empty() {
        edges_near_boundary as f64 / edge_points.len() as f64
    } else {
        1.0 // No edges means nothing to miss
    };

    // F1-like alignment score
    let alignment_score = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    // Identify specific issues
    if precision < 0.3 {
        issues.push("Many boundary points don't align with image edges".to_string());
    }
    if recall < 0.3 {
        issues.push("Many image edges near object are not captured by mask".to_string());
    }
    if ann.area < 100.0 {
        issues.push("Very small annotation - may need manual review".to_string());
    }

    AnnotationQuality {
        annotation_id: ann.id,
        image_id: ann.image_id,
        category: category.to_string(),
        edge_alignment_score: alignment_score,
        boundary_precision: precision,
        boundary_recall: recall,
        issues,
    }
}

fn extract_boundary_from_annotation(ann: &CocoAnnotation, width: u32, height: u32) -> Vec<(u32, u32)> {
    match &ann.segmentation {
        Segmentation::Polygon(polygons) => {
            let mut boundary_points = Vec::new();
            for polygon in polygons {
                // Polygon is [x1, y1, x2, y2, ...]
                let points: Vec<(f64, f64)> = polygon
                    .chunks(2)
                    .filter(|c| c.len() == 2)
                    .map(|c| (c[0], c[1]))
                    .collect();

                // Sample points along polygon edges
                for i in 0..points.len() {
                    let (x1, y1) = points[i];
                    let (x2, y2) = points[(i + 1) % points.len()];

                    // Interpolate along edge
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
            // For RLE, create mask and extract boundary
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
    // COCO uses a modified LEB128 encoding
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
            let b = bytes[i] as u32 - 48; // ASCII offset
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
                // Check if this is a boundary pixel
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

fn is_near_edge(x: u32, y: u32, edges: &GrayImage, threshold: u32) -> bool {
    let (width, height) = edges.dimensions();
    let t = threshold as i32;

    for dy in -t..=t {
        for dx in -t..=t {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;

            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                if edges.get_pixel(nx as u32, ny as u32)[0] > 0 {
                    return true;
                }
            }
        }
    }

    false
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
    edges: &GrayImage,
    annotations: &[CocoAnnotation],
    _img_info: &CocoImage,
) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut vis: RgbImage = ImageBuffer::new(width * 2, height);

    // Left side: original with annotations
    for y in 0..height {
        for x in 0..width {
            vis.put_pixel(x, y, *img.get_pixel(x, y));
        }
    }

    // Draw annotation boundaries in green
    for ann in annotations {
        let boundary = extract_boundary_from_annotation(ann, width, height);
        for (x, y) in boundary {
            vis.put_pixel(x, y, Rgb([0, 255, 0]));
        }
    }

    // Right side: edges with annotations overlaid
    for y in 0..height {
        for x in 0..width {
            let edge_val = edges.get_pixel(x, y)[0];
            vis.put_pixel(width + x, y, Rgb([edge_val, edge_val, edge_val]));
        }
    }

    // Overlay annotations in red on edges
    for ann in annotations {
        let boundary = extract_boundary_from_annotation(ann, width, height);
        for (x, y) in boundary {
            vis.put_pixel(width + x, y, Rgb([255, 0, 0]));
        }
    }

    vis
}
