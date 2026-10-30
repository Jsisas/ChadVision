//! C ABI for the icon identifier with hover-tooltip name correction (see `CvCandidate`).
use crate::catalog::Catalog;
use crate::methods::{corrected::Corrected, embed::Embed, fusion::Fusion, ocr::Ocr, Identifier, Input, Tooltip};
use crate::refs;
use image::RgbImage;
use std::cell::RefCell;
use std::ffi::{c_char, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;

/// `id` is a NUL-padded ASCII tarkov.dev item id.
#[repr(C)]
pub struct CvCandidate {
    pub id: [u8; 32],
    pub score: f32,
}

pub struct CvEngine {
    #[allow(dead_code)]
    catalog: Arc<Catalog>,
    corrected: Corrected,
}

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(msg: String) {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg);
}

fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic".into())
}

fn open(dir: &Path) -> anyhow::Result<CvEngine> {
    // Explicit path: a bare "onnxruntime.dll" could resolve to an older copy in System32.
    #[cfg(windows)]
    {
        let exe_dir = std::env::current_exe()?.parent().map(Path::to_path_buf).unwrap_or_default();
        ort::init_from(exe_dir.join("onnxruntime.dll").to_string_lossy()).commit()?;
    }
    let refs_dir = dir.join("refs");
    let models = dir.join("models");
    let catalog = Arc::new(Catalog::load(&refs_dir)?);
    let bases = refs::load_bases(&refs_dir, &catalog);
    let embed = Arc::new(Embed::new(catalog.clone(), &bases, &models.join("dinov2-small.onnx"), &dir.join("index"))?);
    let ocr = Arc::new(Ocr::new(
        catalog.clone(),
        &models.join("ppocrv5_latin_rec.onnx"),
        &models.join("ppocrv5_latin_dict.txt"),
    )?);
    drop(bases);
    let fusion = Arc::new(Fusion::new(catalog.clone(), embed, ocr.clone()));
    let corrected = Corrected::new(catalog.clone(), fusion, ocr);
    Ok(CvEngine { catalog, corrected })
}

/// Returns null on error; see `cv_last_error`.
#[no_mangle]
pub extern "C" fn cv_open(data_dir: *const c_char) -> *mut CvEngine {
    if data_dir.is_null() {
        set_error("data_dir is null".into());
        return std::ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let dir = unsafe { CStr::from_ptr(data_dir) }.to_str()?;
        open(Path::new(dir))
    }));
    match result {
        Ok(Ok(e)) => Box::into_raw(Box::new(e)),
        Ok(Err(e)) => {
            set_error(format!("{e:#}"));
            std::ptr::null_mut()
        }
        Err(p) => {
            set_error(panic_message(p));
            std::ptr::null_mut()
        }
    }
}

/// `bgr`: crop rows of B,G,R; `screen_bgr` (nullable): full scan screenshot, cursor at (`mouse_x`,`mouse_y`),
/// used to read EFT's hover tooltip name. Returns candidate count or -1.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn cv_identify(
    e: *const CvEngine,
    bgr: *const u8,
    width: u32,
    height: u32,
    stride: u32,
    slot_w: u32,
    slot_h: u32,
    screen_bgr: *const u8,
    screen_width: u32,
    screen_height: u32,
    screen_stride: u32,
    mouse_x: i32,
    mouse_y: i32,
    scale: f32,
    out: *mut CvCandidate,
    cap: u32,
) -> i32 {
    if e.is_null() || bgr.is_null() || out.is_null() || width == 0 || height == 0 || stride < width.saturating_mul(3) {
        set_error("invalid argument".into());
        return -1;
    }
    if !screen_bgr.is_null() && (screen_width == 0 || screen_height == 0 || screen_stride < screen_width.saturating_mul(3)) {
        set_error("invalid screen argument".into());
        return -1;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let engine = unsafe { &*e };
        let crop = unsafe { bgr_to_rgb(bgr, width, height, stride) };
        let tooltip = (!screen_bgr.is_null())
            .then(|| {
                let screen = unsafe { bgr_to_rgb(screen_bgr, screen_width, screen_height, screen_stride) };
                let (image, mouse) = crate::dataset::tip_window(&screen, (mouse_x as i64, mouse_y as i64), scale);
                (image.width() > 0 && image.height() > 0).then_some(Tooltip { image, mouse, scale })
            })
            .flatten();
        let cands = engine.corrected.identify(&Input { crop: &crop, slots: (slot_w, slot_h), tooltip: tooltip.as_ref() });
        let out = unsafe { std::slice::from_raw_parts_mut(out, cap as usize) };
        let mut n = 0;
        for (slot, c) in out.iter_mut().zip(cands) {
            let mut id = [0u8; 32];
            let bytes = c.item_id.as_bytes();
            let len = bytes.len().min(31);
            id[..len].copy_from_slice(&bytes[..len]);
            *slot = CvCandidate { id, score: c.score };
            n += 1;
        }
        n as i32
    }));
    match result {
        Ok(n) => n,
        Err(p) => {
            set_error(panic_message(p));
            -1
        }
    }
}

/// Copies BGR rows (`stride` bytes apart) into an RGB image. Reads `stride*(h-1) + w*3` bytes.
unsafe fn bgr_to_rgb(bgr: *const u8, width: u32, height: u32, stride: u32) -> RgbImage {
    let (w, h, stride) = (width as usize, height as usize, stride as usize);
    let src = std::slice::from_raw_parts(bgr, stride * (h - 1) + w * 3);
    let mut buf = Vec::with_capacity(w * h * 3);
    for row in 0..h {
        for px in src[row * stride..row * stride + w * 3].chunks_exact(3) {
            buf.extend_from_slice(&[px[2], px[1], px[0]]);
        }
    }
    RgbImage::from_raw(width, height, buf).expect("buffer size")
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct CvCell {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub slot_w: u32,
    pub slot_h: u32,
}

/// Item cell under (`mouse_x`, `mouse_y`) from the stash grid lines in a BGR screenshot.
/// `hint_slot_*` fill in a side whose far edge is not visible. Returns 1 found, 0 not found, -1 error.
#[no_mangle]
pub extern "C" fn cv_locate_cell(
    bgr: *const u8,
    width: u32,
    height: u32,
    stride: u32,
    mouse_x: i32,
    mouse_y: i32,
    scale: f32,
    hint_slot_w: u32,
    hint_slot_h: u32,
    out: *mut CvCell,
) -> i32 {
    if bgr.is_null() || out.is_null() || width == 0 || height == 0 || stride < width.saturating_mul(3) || !(scale > 0.0) {
        set_error("invalid argument".into());
        return -1;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let img = unsafe { bgr_to_rgb(bgr, width, height, stride) };
        crate::grid::locate_cell(&img, (mouse_x, mouse_y), scale, (hint_slot_w, hint_slot_h))
    }));
    match result {
        Ok(Some(c)) => {
            unsafe { *out = CvCell { x: c.x, y: c.y, w: c.w, h: c.h, slot_w: c.slots.0, slot_h: c.slots.1 } };
            1
        }
        Ok(None) => 0,
        Err(p) => {
            set_error(panic_message(p));
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn cv_close(e: *mut CvEngine) {
    if !e.is_null() {
        drop(unsafe { Box::from_raw(e) });
    }
}

/// Copies the calling thread's last error into `buf` (NUL-terminated); returns bytes written excl. NUL.
#[no_mangle]
pub extern "C" fn cv_last_error(buf: *mut c_char, cap: u32) -> u32 {
    if buf.is_null() || cap == 0 {
        return 0;
    }
    LAST_ERROR.with(|e| {
        let e = e.borrow();
        let n = e.len().min(cap as usize - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(e.as_ptr(), buf as *mut u8, n);
            *buf.add(n) = 0;
        }
        n as u32
    })
}
