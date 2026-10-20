use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

pub mod hog;
pub mod image;
pub mod preprocess;
pub mod svm;

use hog::{extract_hog, HogCellGrid, HOG_DIM};
use image::Image;
use svm::{LinearSvm, SvmTrainConfig};

/// Preprocesses raw RGBA bytes into a 64x64 grayscale buffer.
#[wasm_bindgen(js_name = preprocess)]
pub fn run_preprocess(rgba: Vec<u8>, width: u32, height: u32) -> Vec<u8> {
    preprocess::preprocess(&rgba, width, height).into_pixels()
}

/// Result of HOG extraction exported to JS.
#[derive(Serialize, Deserialize)]
pub struct HogResult {
    pub descriptor: Vec<f32>,
    pub cell_grid: HogCellGrid,
}

/// Extracts HOG features from a 64x64 grayscale image.
#[wasm_bindgen(js_name = extractHog)]
pub fn run_extract_hog(pixels: Vec<u8>) -> Result<JsValue, JsValue> {
    let image = Image::new(pixels);
    let (descriptor, cell_grid) = extract_hog(&image);
    let result = HogResult {
        descriptor,
        cell_grid,
    };
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// JS-facing handle for a trained Linear SVM.
#[wasm_bindgen]
pub struct SvmModelHandle {
    model: LinearSvm,
}

#[wasm_bindgen]
impl SvmModelHandle {
    /// Creates an SVM handle directly from pre-trained weights and bias.
    #[wasm_bindgen(constructor)]
    pub fn new(weights: Vec<f32>, bias: f32) -> Self {
        Self {
            model: LinearSvm::new(weights, bias),
        }
    }

    /// Evaluates the SVM given a 1764-dimensional HOG vector.
    #[wasm_bindgen]
    pub fn predict(&self, hog_features: Vec<f32>) -> Result<JsValue, JsValue> {
        let pred = self.model.predict(&hog_features);
        serde_wasm_bindgen::to_value(&pred).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Predicts directly from a 64x64 grayscale image (computes HOG + SVM).
    #[wasm_bindgen(js_name = predictImage)]
    pub fn predict_image(&self, pixels: Vec<u8>) -> Result<JsValue, JsValue> {
        let image = Image::new(pixels);
        let (hog, _grid) = extract_hog(&image);
        let pred = self.model.predict(&hog);
        serde_wasm_bindgen::to_value(&pred).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Returns weights and bias as a JS object for inspection / visualization.
    #[wasm_bindgen(js_name = getModelData)]
    pub fn get_model_data(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.model).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// In-browser training function: trains Linear SVM on extracted HOG descriptors.
/// `flat_hogs`: length N * 1764
/// `labels`: length N (0 = cat, 1 = dog)
#[wasm_bindgen(js_name = trainSvm)]
pub fn train_svm(flat_hogs: Vec<f32>, labels: Vec<u8>, epochs: usize, lambda: f32) -> Result<SvmModelHandle, JsValue> {
    let n = labels.len();
    if flat_hogs.len() != n * HOG_DIM {
        return Err(JsValue::from_str(&format!(
            "flat_hogs length ({}) != labels length ({}) * HOG_DIM ({})",
            flat_hogs.len(),
            n,
            HOG_DIM
        )));
    }

    let mut features = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * HOG_DIM;
        let end = start + HOG_DIM;
        features.push(flat_hogs[start..end].to_vec());
    }

    let mut config = SvmTrainConfig::default();
    if epochs > 0 {
        config.epochs = epochs;
    }
    if lambda > 0.0 {
        config.lambda = lambda;
    }

    let (model, _loss) = LinearSvm::train(&features, &labels, &config);
    Ok(SvmModelHandle { model })
}
