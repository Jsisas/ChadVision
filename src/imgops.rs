//! Shared image geometry. A "crop" is the item cell rectangle padded by 1/8 slot on each side
//! (RatEye's `TryAddIcon` padding). "Base" is the 1080p size: 63 px per slot.
use fast_image_resize::{images::Image, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::{imageops, Rgb, RgbImage};

pub const PITCH: f32 = 63.0;
pub const GRID_RGB: Rgb<u8> = Rgb([73, 81, 84]);

pub fn resize(img: &RgbImage, w: u32, h: u32) -> RgbImage {
    let (w, h) = (w.max(1), h.max(1));
    if img.dimensions() == (w, h) {
        return img.clone();
    }
    let src = Image::from_vec_u8(img.width(), img.height(), img.as_raw().clone(), PixelType::U8x3)
        .expect("valid source buffer");
    let mut dst = Image::new(w, h, PixelType::U8x3);
    Resizer::new()
        .resize(&src, &mut dst, &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3)))
        .expect("resize");
    RgbImage::from_raw(w, h, dst.into_vec()).expect("dst buffer")
}

/// UI scale implied by a padded crop of an item with on-screen size `slots`.
pub fn crop_scale(crop: &RgbImage, slots: (u32, u32)) -> f32 {
    let sx = crop.width() as f32 / (PITCH * (slots.0 as f32 + 0.25));
    let sy = crop.height() as f32 / (PITCH * (slots.1 as f32 + 0.25));
    (sx + sy) / 2.0
}

/// Padded crop rescaled to base scale (≈ 63·(w+¼) × 63·(h+¼)).
pub fn crop_to_base_padded(crop: &RgbImage, slots: (u32, u32)) -> RgbImage {
    let s = crop_scale(crop, slots);
    resize(crop, (crop.width() as f32 / s).round() as u32, (crop.height() as f32 / s).round() as u32)
}

/// Cell content at base scale: exactly 63·w × 63·h, cut from the centre of the padded crop.
pub fn crop_to_base(crop: &RgbImage, slots: (u32, u32)) -> RgbImage {
    let p = crop_to_base_padded(crop, slots);
    let (bw, bh) = (63 * slots.0, 63 * slots.1);
    let (pw, ph) = (p.width().max(bw), p.height().max(bh));
    let p = if (pw, ph) != p.dimensions() { resize(&p, pw, ph) } else { p };
    imageops::crop_imm(&p, (pw - bw) / 2, (ph - bh) / 2, bw, bh).to_image()
}

/// Letterbox onto a square grid-coloured canvas.
pub fn letterbox(img: &RgbImage) -> RgbImage {
    let side = img.width().max(img.height());
    let mut canvas = RgbImage::from_pixel(side, side, GRID_RGB);
    imageops::overlay(&mut canvas, img, ((side - img.width()) / 2) as i64, ((side - img.height()) / 2) as i64);
    canvas
}
