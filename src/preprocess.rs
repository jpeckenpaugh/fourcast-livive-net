use crate::image::{Image, IMAGE_LEN, IMAGE_SIDE};

/// Preprocesses RGBA buffer to 64x64 grayscale via visible integer arithmetic.
pub fn preprocess(rgba: &[u8], width: u32, height: u32) -> Image {
    let width = width as u64;
    let height = height as u64;

    let expected_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .expect("preprocess: width * height * 4 overflows u64");

    assert_eq!(
        rgba.len() as u64,
        expected_len,
        "preprocess: rgba length must be width * height * 4 = {expected_len}, got {}",
        rgba.len()
    );
    assert!(width >= 64, "preprocess: width must be at least 64, got {width}");
    assert!(height >= 64, "preprocess: height must be at least 64, got {height}");

    let mut out = vec![0u8; IMAGE_LEN];
    let width_usize = width as usize;
    let side = IMAGE_SIDE as u64;

    for oy in 0..side {
        let y0 = oy * height / side;
        let y1 = (oy + 1) * height / side;

        for ox in 0..side {
            let x0 = ox * width / side;
            let x1 = (ox + 1) * width / side;

            let mut sum: u64 = 0;
            for y in y0..y1 {
                let row = y as usize * width_usize;
                for x in x0..x1 {
                    let i = (row + x as usize) * 4;
                    let r = rgba[i] as u64;
                    let g = rgba[i + 1] as u64;
                    let b = rgba[i + 2] as u64;
                    // Alpha is ignored
                    let gray = (299 * r + 587 * g + 114 * b + 500) / 1000;
                    sum += gray;
                }
            }

            let count = (x1 - x0) * (y1 - y0);
            let value = (sum + count / 2) / count;
            out[oy as usize * IMAGE_SIDE + ox as usize] = value as u8;
        }
    }

    Image::new(out)
}
