use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

use crate::image::{Image, IMAGE_SIDE};

pub const CELL_SIZE: usize = 8;
pub const NUM_BINS: usize = 9;
pub const CELLS_PER_AXIS: usize = IMAGE_SIDE / CELL_SIZE; // 64 / 8 = 8
pub const BLOCKS_PER_AXIS: usize = CELLS_PER_AXIS - 1; // 8 - 1 = 7
pub const CELLS_PER_BLOCK: usize = 2 * 2; // 4
pub const BLOCK_DIM: usize = CELLS_PER_BLOCK * NUM_BINS; // 36
pub const HOG_DIM: usize = BLOCKS_PER_AXIS * BLOCKS_PER_AXIS * BLOCK_DIM; // 7 * 7 * 36 = 1764

/// Cell-level histogram of oriented gradients (8x8 cells, each having 9 orientation bins).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HogCellGrid {
    pub cells_per_axis: usize,
    pub num_bins: usize,
    /// Length: CELLS_PER_AXIS * CELLS_PER_AXIS * NUM_BINS (8 * 8 * 9 = 576 floats)
    pub histograms: Vec<f32>,
}

/// Computes cell-level orientation histograms and returns the full HOG descriptor (1764 floats)
/// along with the cell histogram grid for interactive UI inspection.
pub fn extract_hog(image: &Image) -> (Vec<f32>, HogCellGrid) {
    let pixels = image.pixels();

    // 1. Compute gradients Ix and Iy for all pixels using [-1, 0, 1] kernel
    // Interior: gx = p(x+1, y) - p(x-1, y), gy = p(x, y+1) - p(x, y-1)
    // Edges use forward/backward differences
    let mut grad_mag = [0.0f32; IMAGE_SIDE * IMAGE_SIDE];
    let mut grad_ori = [0.0f32; IMAGE_SIDE * IMAGE_SIDE];

    for y in 0..IMAGE_SIDE {
        for x in 0..IMAGE_SIDE {
            let idx = y * IMAGE_SIDE + x;

            let gx = if x == 0 {
                pixels[idx + 1] as f32 - pixels[idx] as f32
            } else if x == IMAGE_SIDE - 1 {
                pixels[idx] as f32 - pixels[idx - 1] as f32
            } else {
                (pixels[idx + 1] as f32 - pixels[idx - 1] as f32) * 0.5
            };

            let gy = if y == 0 {
                pixels[(y + 1) * IMAGE_SIDE + x] as f32 - pixels[idx] as f32
            } else if y == IMAGE_SIDE - 1 {
                pixels[idx] as f32 - pixels[(y - 1) * IMAGE_SIDE + x] as f32
            } else {
                (pixels[(y + 1) * IMAGE_SIDE + x] as f32 - pixels[(y - 1) * IMAGE_SIDE + x] as f32) * 0.5
            };

            let mag = (gx * gx + gy * gy).sqrt();
            // Unsigned orientation in radians: [0, PI)
            let mut angle = gy.atan2(gx);
            if angle < 0.0 {
                angle += PI;
            }
            if angle >= PI {
                angle = PI - 1e-6;
            }

            grad_mag[idx] = mag;
            grad_ori[idx] = angle;
        }
    }

    // 2. Accumulate into 8x8 spatial cells with 9 bins
    // Bin width is PI / 9
    let bin_width = PI / (NUM_BINS as f32);
    let mut cell_histograms = vec![0.0f32; CELLS_PER_AXIS * CELLS_PER_AXIS * NUM_BINS];

    for cy in 0..CELLS_PER_AXIS {
        for cx in 0..CELLS_PER_AXIS {
            let cell_offset = (cy * CELLS_PER_AXIS + cx) * NUM_BINS;

            for dy in 0..CELL_SIZE {
                let py = cy * CELL_SIZE + dy;
                for dx in 0..CELL_SIZE {
                    let px = cx * CELL_SIZE + dx;
                    let idx = py * IMAGE_SIDE + px;

                    let mag = grad_mag[idx];
                    if mag == 0.0 {
                        continue;
                    }
                    let ori = grad_ori[idx];

                    // Linear interpolation between two adjacent orientation bins
                    let continuous_bin = ori / bin_width;
                    let bin_low = (continuous_bin.floor() as usize) % NUM_BINS;
                    let bin_high = (bin_low + 1) % NUM_BINS;
                    let frac = continuous_bin - continuous_bin.floor();

                    cell_histograms[cell_offset + bin_low] += mag * (1.0 - frac);
                    cell_histograms[cell_offset + bin_high] += mag * frac;
                }
            }
        }
    }

    // 3. Block normalization (2x2 cells per block, stride of 1 cell)
    // 7x7 blocks, each block is 4 cells * 9 bins = 36 values
    let mut descriptor = Vec::with_capacity(HOG_DIM);

    for by in 0..BLOCKS_PER_AXIS {
        for bx in 0..BLOCKS_PER_AXIS {
            // Collect the 4 cells in this block: (by, bx), (by, bx+1), (by+1, bx), (by+1, bx+1)
            let mut block_vec = [0.0f32; BLOCK_DIM];
            let mut block_sq_norm = 0.0f32;

            for (c_idx, &(dy, dx)) in [(0, 0), (0, 1), (1, 0), (1, 1)].iter().enumerate() {
                let cy = by + dy;
                let cx = bx + dx;
                let cell_offset = (cy * CELLS_PER_AXIS + cx) * NUM_BINS;

                for b in 0..NUM_BINS {
                    let val = cell_histograms[cell_offset + b];
                    let out_idx = c_idx * NUM_BINS + b;
                    block_vec[out_idx] = val;
                    block_sq_norm += val * val;
                }
            }

            // L2-Hys normalization (L2 norm -> clip to 0.2 -> re-normalize)
            let eps = 1e-4;
            let norm = (block_sq_norm + eps).sqrt();
            let mut second_sq_norm = 0.0f32;

            for val in &mut block_vec {
                *val = (*val / norm).min(0.2);
                second_sq_norm += *val * *val;
            }

            let second_norm = (second_sq_norm + eps).sqrt();
            for val in &mut block_vec {
                *val /= second_norm;
                descriptor.push(*val);
            }
        }
    }

    let grid = HogCellGrid {
        cells_per_axis: CELLS_PER_AXIS,
        num_bins: NUM_BINS,
        histograms: cell_histograms,
    };

    (descriptor, grid)
}
