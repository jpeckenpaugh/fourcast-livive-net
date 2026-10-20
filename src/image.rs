use serde::{Deserialize, Serialize};

pub const IMAGE_SIDE: usize = 64;
pub const IMAGE_LEN: usize = IMAGE_SIDE * IMAGE_SIDE;

/// A 64x64 row-major 8-bit grayscale image.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Image {
    pixels: Vec<u8>,
}

impl Image {
    pub fn new(pixels: Vec<u8>) -> Self {
        assert_eq!(
            pixels.len(),
            IMAGE_LEN,
            "Image must be exactly {} bytes (64x64), got {}",
            IMAGE_LEN,
            pixels.len()
        );
        Self { pixels }
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub fn into_pixels(self) -> Vec<u8> {
        self.pixels
    }
}
