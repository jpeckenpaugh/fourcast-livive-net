use serde::{Deserialize, Serialize};

/// Trained Linear SVM model parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearSvm {
    /// Dimension equal to feature size (e.g. 1764 for HOG)
    pub weights: Vec<f32>,
    pub bias: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SvmPrediction {
    /// Decision value: w^T x + bias. > 0 indicates Cat (label 0), < 0 indicates Dog (label 1).
    pub decision_value: f32,
    /// Probability of Cat via Platt sigmoid: 1 / (1 + exp(-decision_value))
    pub p_cat: f32,
    /// Probability of Dog
    pub p_dog: f32,
    /// Predicted label: "cat" or "dog"
    pub label: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SvmTrainConfig {
    pub lambda: f32,       // L2 regularization parameter (e.g. 1e-4)
    pub epochs: usize,     // Passes over dataset (e.g. 30)
    pub batch_size: usize, // Mini-batch size (e.g. 32)
    pub lr0: f32,          // Initial learning rate (e.g. 0.05)
    pub seed: u64,
}

impl Default for SvmTrainConfig {
    fn default() -> Self {
        Self {
            lambda: 1e-3,
            epochs: 40,
            batch_size: 32,
            lr0: 0.05,
            seed: 42,
        }
    }
}

impl LinearSvm {
    pub fn new(weights: Vec<f32>, bias: f32) -> Self {
        Self { weights, bias }
    }

    /// Predicts cat probability and decision margin for a given feature vector.
    pub fn predict(&self, features: &[f32]) -> SvmPrediction {
        assert_eq!(
            self.weights.len(),
            features.len(),
            "Feature dimension mismatch: expected {}, got {}",
            self.weights.len(),
            features.len()
        );

        let mut dot = 0.0f32;
        for (w, x) in self.weights.iter().zip(features.iter()) {
            dot += w * x;
        }
        let decision_value = dot + self.bias;

        // Sigmoid mapping for smooth probability display in UI
        let p_cat = 1.0 / (1.0 + (-decision_value).exp());
        let p_dog = 1.0 - p_cat;
        let label = if decision_value >= 0.0 {
            "cat".to_string()
        } else {
            "dog".to_string()
        };

        SvmPrediction {
            decision_value,
            p_cat,
            p_dog,
            label,
        }
    }

    /// Trains a linear SVM using mini-batch subgradient descent on hinge loss with L2 regularization.
    /// Labels: 0 = cat (+1 in SVM margin), 1 = dog (-1 in SVM margin).
    pub fn train(
        features: &[Vec<f32>],
        labels: &[u8],
        config: &SvmTrainConfig,
    ) -> (Self, Vec<f32>) {
        let n = features.len();
        assert_eq!(n, labels.len(), "Number of samples and labels must match");
        assert!(n > 0, "Training set cannot be empty");
        let dim = features[0].len();

        let mut weights = vec![0.0f32; dim];
        let mut bias = 0.0f32;
        let mut loss_history = Vec::with_capacity(config.epochs);

        // Simple Xorshift PRNG for reproducible shuffling
        let mut rng_state = if config.seed == 0 { 0xdeadbeef } else { config.seed };
        let mut next_u32 = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            rng_state as u32
        };

        let mut indices: Vec<usize> = (0..n).collect();

        for epoch in 0..config.epochs {
            // Fisher-Yates shuffle
            for i in (1..n).rev() {
                let j = (next_u32() as usize) % (i + 1);
                indices.swap(i, j);
            }

            // Learning rate decay: lr = lr0 / sqrt(1 + epoch)
            let lr = config.lr0 / (1.0 + epoch as f32 * 0.1).sqrt();

            for chunk in indices.chunks(config.batch_size) {
                let batch_len = chunk.len() as f32;
                let mut grad_w = vec![0.0f32; dim];
                let mut grad_b = 0.0f32;

                for &idx in chunk {
                    let x = &features[idx];
                    let y = if labels[idx] == 0 { 1.0f32 } else { -1.0f32 };

                    let mut dot = 0.0f32;
                    for (w, val) in weights.iter().zip(x.iter()) {
                        dot += w * val;
                    }
                    let margin = y * (dot + bias);

                    // Hinge loss gradient: if margin < 1, subgradient is -y * x
                    if margin < 1.0 {
                        let scale = -y;
                        for (gw, val) in grad_w.iter_mut().zip(x.iter()) {
                            *gw += scale * val;
                        }
                        grad_b += scale;
                    }
                }

                // Update weights with L2 weight decay: w <- w - lr * (lambda * w + grad_w / batch_len)
                for (w, gw) in weights.iter_mut().zip(grad_w.iter()) {
                    let total_grad = config.lambda * (*w) + (*gw) / batch_len;
                    *w -= lr * total_grad;
                }
                bias -= lr * (grad_b / batch_len);
            }

            // Compute epoch hinge loss + regularization for monitoring
            let mut total_hinge = 0.0f32;
            for (x, &l) in features.iter().zip(labels.iter()) {
                let y = if l == 0 { 1.0f32 } else { -1.0f32 };
                let mut dot = 0.0f32;
                for (w, val) in weights.iter().zip(x.iter()) {
                    dot += w * val;
                }
                let margin = y * (dot + bias);
                total_hinge += (1.0 - margin).max(0.0);
            }

            let mut w_sq = 0.0f32;
            for w in &weights {
                w_sq += w * w;
            }
            let epoch_loss = 0.5 * config.lambda * w_sq + (total_hinge / n as f32);
            loss_history.push(epoch_loss);
        }

        (Self { weights, bias }, loss_history)
    }
}
