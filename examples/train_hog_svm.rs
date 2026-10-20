use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use fourcast::hog::{extract_hog, HOG_DIM};
use fourcast::image::Image;
use fourcast::svm::{LinearSvm, SvmTrainConfig};

#[derive(Debug, Deserialize)]
struct ManifestEntry {
    id: String,
    file: String,
    label: String,
    split: String,
}

fn load_jpeg_pixels(path: &Path) -> Image {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    // Decode JPEG
    let mut decoder = jpeg_decoder::Decoder::new(&bytes[..]);
    let raw_pixels = decoder.decode().expect("Failed to decode JPEG");
    let info = decoder.info().expect("Failed to read JPEG header");
    let (w, h) = (info.width as u32, info.height as u32);

    // Convert to RGBA for preprocess
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    if info.pixel_format == jpeg_decoder::PixelFormat::RGB24 {
        for chunk in raw_pixels.chunks_exact(3) {
            rgba.push(chunk[0]);
            rgba.push(chunk[1]);
            rgba.push(chunk[2]);
            rgba.push(255);
        }
    } else if info.pixel_format == jpeg_decoder::PixelFormat::L8 {
        for &p in &raw_pixels {
            rgba.push(p);
            rgba.push(p);
            rgba.push(p);
            rgba.push(255);
        }
    } else {
        panic!("Unsupported pixel format: {:?}", info.pixel_format);
    }

    fourcast::preprocess::preprocess(&rgba, w, h)
}

fn main() {
    let manifest_path = Path::new("data/manifest.json");
    if !manifest_path.exists() {
        eprintln!("Error: data/manifest.json not found. Run ./build_dataset.sh first.");
        std::process::exit(1);
    }

    let file = File::open(manifest_path).expect("Cannot open manifest.json");
    let entries: Vec<ManifestEntry> =
        serde_json::from_reader(BufReader::new(file)).expect("Invalid manifest.json");

    println!("Loaded manifest with {} entries.", entries.len());

    let mut train_features = Vec::new();
    let mut train_labels = Vec::new();
    let mut val_features = Vec::new();
    let mut val_labels = Vec::new();
    let mut holdout_features = Vec::new();
    let mut holdout_labels = Vec::new();

    println!("Extracting HOG features from images...");
    let start_extract = Instant::now();

    for (i, entry) in entries.iter().enumerate() {
        if i % 200 == 0 && i > 0 {
            println!("  Processed {} / {} images...", i, entries.len());
        }

        let img_path = Path::new(&entry.file);
        let image = load_jpeg_pixels(img_path);
        let (hog, _grid) = extract_hog(&image);
        let label_num = if entry.label == "cat" { 0u8 } else { 1u8 };

        match entry.split.as_str() {
            "train" => {
                train_features.push(hog);
                train_labels.push(label_num);
            }
            "val" => {
                val_features.push(hog);
                val_labels.push(label_num);
            }
            "holdout" => {
                holdout_features.push(hog);
                holdout_labels.push(label_num);
            }
            _ => {}
        }
    }

    println!(
        "HOG extraction finished in {:.2}s. Splits: Train={}, Val={}, Holdout={}",
        start_extract.elapsed().as_secs_f32(),
        train_features.len(),
        val_features.len(),
        holdout_features.len()
    );

    let config = SvmTrainConfig {
        lambda: 1e-3,
        epochs: 50,
        batch_size: 32,
        lr0: 0.05,
        seed: 42,
    };

    println!("\nTraining Linear SVM (Pegasos/mini-batch SGD on Hinge Loss)...");
    let start_train = Instant::now();
    let (svm, loss_history) = LinearSvm::train(&train_features, &train_labels, &config);
    println!(
        "Training completed in {:.2}ms. Initial loss: {:.4}, Final loss: {:.4}",
        start_train.elapsed().as_secs_f32() * 1000.0,
        loss_history.first().copied().unwrap_or(0.0),
        loss_history.last().copied().unwrap_or(0.0)
    );

    // Evaluate on all splits
    let eval_split = |name: &str, feats: &[Vec<f32>], labels: &[u8]| {
        let mut correct = 0;
        let mut cat_correct = 0;
        let mut cat_total = 0;
        let mut dog_correct = 0;
        let mut dog_total = 0;

        for (x, &y) in feats.iter().zip(labels.iter()) {
            let pred = svm.predict(x);
            let pred_label = if pred.decision_value >= 0.0 { 0u8 } else { 1u8 };
            let is_correct = pred_label == y;
            if is_correct {
                correct += 1;
            }
            if y == 0 {
                cat_total += 1;
                if is_correct {
                    cat_correct += 1;
                }
            } else {
                dog_total += 1;
                if is_correct {
                    dog_correct += 1;
                }
            }
        }

        let acc = (correct as f32 / feats.len() as f32) * 100.0;
        let cat_acc = (cat_correct as f32 / cat_total as f32) * 100.0;
        let dog_acc = (dog_correct as f32 / dog_total as f32) * 100.0;
        println!(
            "[{}] Accuracy: {:.2}% ({}/{}). Cat: {:.1}%, Dog: {:.1}%",
            name, acc, correct, feats.len(), cat_acc, dog_acc
        );
    };

    println!("\n=== Evaluation Results ===");
    eval_split("TRAIN  ", &train_features, &train_labels);
    eval_split("VAL    ", &val_features, &val_labels);
    eval_split("HOLDOUT", &holdout_features, &holdout_labels);

    // Save model to data/hog_svm_model.json
    let out_path = Path::new("data/hog_svm_model.json");
    let serialized = serde_json::to_string_pretty(&svm).expect("Failed to serialize SVM model");
    std::fs::write(out_path, serialized).expect("Failed to write model file");
    println!("\nSaved trained model weights to {}", out_path.display());
}
