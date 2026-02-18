// image.rs — Runtime-sized image container, generic over pixel type.
//
// KEY DIFFERENCES FROM lalir::Matrix<M, N>:
// ┌──────────────────────────┬───────────────────────────────────────┐
// │  lalir::Matrix<M, N>     │  Image<T>                             │
// ├──────────────────────────┼───────────────────────────────────────┤
// │  Dimensions at compile   │  Dimensions at runtime                │
// │  time (const generics)   │  (stored as usize fields)             │
// ├──────────────────────────┼───────────────────────────────────────┤
// │  Stack-allocated array   │  Heap-allocated Vec<T>                │
// │  [T; M * N]              │  (size known only at runtime)         │
// ├──────────────────────────┼───────────────────────────────────────┤
// │  Fixed-size → no stride  │  Stride may differ from width         │
// │  concerns                │  (alignment padding for GPU later)    │
// └──────────────────────────┴───────────────────────────────────────┘
//
// New concepts this file introduces:
// - Trait definition + implementation (Pixel)
// - Vec<T> heap allocation
// - Lifetime annotations ('a) on ImageView
// - Send + Sync bounds (marker traits for thread safety)
// - Default trait
// - impl Iterator (return type that hides the concrete iterator type)

use std::fmt;

// ---------------------------------------------------------------------------
// Pixel Trait
// ---------------------------------------------------------------------------
// This is a "trait" — Rust's version of an interface / typeclass.
// Any type that implements Pixel can be stored in an Image.
//
// Trait bounds explained:
//   Copy    — pixel values are trivially copyable (no deep clone needed)
//   Default — can produce a zero/default value (used for new())
//   Send    — safe to transfer between threads
//   Sync    — safe to share references between threads
//   'static — the type itself contains no borrowed references
//             (u8, f32, etc. all satisfy this; it just rules out types
//              like &'a u8 which you'd never want as a pixel anyway)
//
// We also require PartialOrd so we can compare pixel values (needed for
// FAST detector thresholding in Step 3).

/// Trait for types that can serve as pixel values in an Image.
pub trait Pixel: Copy + Default + Send + Sync + PartialOrd + 'static {
    /// Convert this pixel value to f32 (normalized or raw depending on type).
    fn to_f32(self) -> f32;

    /// Construct a pixel from an f32 value (with appropriate clamping/rounding).
    fn from_f32(v: f32) -> Self;
}

// --- Pixel implementations for concrete types ---

impl Pixel for u8 {
    #[inline]
    fn to_f32(self) -> f32 {
        // Raw cast, NOT normalized to [0,1]. This is intentional:
        // algorithms like FAST compare raw intensity values.
        // Use convert::u8_to_normalized_f32() for [0,1] mapping.
        self as f32
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        // Clamp to valid range, then round.
        // .round() returns f32; `as u8` truncates, so we clamp first.
        v.clamp(0.0, 255.0).round() as u8
    }
}

impl Pixel for u16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v.clamp(0.0, 65535.0).round() as u16
    }
}

impl Pixel for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
}

// ---------------------------------------------------------------------------
// Image<T>
// ---------------------------------------------------------------------------
// Row-major, contiguous buffer with explicit stride.
//
// Memory layout (stride = 5, width = 4):
//
//   data index:  0  1  2  3 [4]  5  6  7  8 [9] 10 11 12 13 [14]
//   pixel:       ■  ■  ■  ■  ·   ■  ■  ■  ■  ·   ■  ■  ■  ■  ·
//   row:         |--- row 0 ---|  |--- row 1 ---|  |--- row 2 ---|
//
//   [4], [9], [14] are padding elements (stride - width = 1 padding per row).
//   These exist so that each row starts at an aligned address on GPU.

/// A 2D image with runtime dimensions, generic over pixel type `T`.
pub struct Image<T: Pixel> {
    /// Pixel data in row-major order. Length = height * stride.
    data: Vec<T>,
    /// Image width in pixels.
    width: usize,
    /// Image height in pixels.
    height: usize,
    /// Row stride in *elements* (not bytes). stride >= width.
    /// Pixels for row y start at index y * stride.
    stride: usize,
}

// We implement Clone manually rather than deriving it, because we want to
// document that this is a potentially expensive deep copy of heap data.
impl<T: Pixel> Clone for Image<T> {
    fn clone(&self) -> Self {
        Image {
            data: self.data.clone(),
            width: self.width,
            height: self.height,
            stride: self.stride,
        }
    }
}

impl<T: Pixel> Image<T> {
    // --- Constructors ---

    /// Create a zero-initialized image with the given dimensions.
    /// Stride equals width (no padding). Use `new_with_stride` if you
    /// need alignment padding.
    pub fn new(width: usize, height: usize) -> Self {
        Self::new_with_stride(width, height, width)
    }

    /// Create a zero-initialized image with an explicit stride.
    ///
    /// # Panics
    /// Panics if `stride < width`.
    pub fn new_with_stride(width: usize, height: usize, stride: usize) -> Self {
        assert!(
            stride >= width,
            "stride ({stride}) must be >= width ({width})"
        );
        Image {
            // vec![value; count] allocates `count` copies of `value` on the heap.
            // T::default() gives us the zero value (0 for u8/u16, 0.0 for f32).
            data: vec![T::default(); height * stride],
            width,
            height,
            stride,
        }
    }

    /// Create an image from an existing pixel vector.
    ///
    /// `data` must contain exactly `height * width` elements (no stride padding).
    /// Stride is set equal to width.
    ///
    /// # Panics
    /// Panics if `data.len() != width * height`.
    pub fn from_vec(width: usize, height: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            width * height,
            "data length ({}) must equal width * height ({})",
            data.len(),
            width * height,
        );
        Image {
            data,
            width,
            height,
            stride: width,
        }
    }

    /// Create an image from raw data with explicit stride.
    ///
    /// # Panics
    /// Panics if `data.len() != height * stride` or `stride < width`.
    pub fn from_vec_with_stride(
        width: usize,
        height: usize,
        stride: usize,
        data: Vec<T>,
    ) -> Self {
        assert!(stride >= width, "stride ({stride}) must be >= width ({width})");
        assert_eq!(
            data.len(),
            height * stride,
            "data length ({}) must equal height * stride ({})",
            data.len(),
            height * stride,
        );
        Image {
            data,
            width,
            height,
            stride,
        }
    }

    // --- Accessors ---

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Get the pixel value at (x, y). x is column, y is row.
    ///
    /// # Panics
    /// Panics if (x, y) is out of bounds.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> T {
        self.bounds_check(x, y);
        // SAFETY: bounds_check guarantees the index is valid.
        self.data[y * self.stride + x]
    }

    /// Get pixel value without bounds checking.
    ///
    /// # Safety
    /// Caller must guarantee x < width and y < height.
    /// Used in hot inner loops (convolution, KLT) where bounds are
    /// already validated at the loop level.
    ///
    /// GPU EQUIVALENT: This mirrors GPU texture loads where the hardware
    /// sampler handles addressing — no per-pixel bounds check.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, x: usize, y: usize) -> T {
        debug_assert!(x < self.width && y < self.height,
            "get_unchecked({x},{y}) out of bounds for {}x{}", self.width, self.height);
        *self.data.get_unchecked(y * self.stride + x)
    }

    /// Set pixel value without bounds checking.
    ///
    /// # Safety
    /// Caller must guarantee x < width and y < height.
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, x: usize, y: usize, value: T) {
        debug_assert!(x < self.width && y < self.height);
        *self.data.get_unchecked_mut(y * self.stride + x) = value;
    }

    /// Get a mutable reference to the pixel at (x, y).
    ///
    /// NOTE on return type: `&mut T`
    /// This returns a *reference* to the pixel in the buffer, not a copy.
    /// The caller can write through this reference: `*img.get_mut(x, y) = value;`
    /// The lifetime of the returned reference is tied to `&mut self`.
    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut T {
        self.bounds_check(x, y);
        let idx = y * self.stride + x;
        &mut self.data[idx]
    }

    /// Set the pixel at (x, y) to the given value.
    /// Convenience wrapper around get_mut.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: T) {
        *self.get_mut(x, y) = value;
    }

    /// Borrow a single row as a slice.
    ///
    /// LIFETIME NOTE: The returned `&[T]` borrows from `&self`.
    /// This means you cannot modify the image while the row slice exists.
    /// This is Rust's borrow checker enforcing memory safety — if you could
    /// mutate the Vec while someone holds a reference into it, a reallocation
    /// could leave the reference dangling (like a C++ iterator invalidation).
    #[inline]
    pub fn row(&self, y: usize) -> &[T] {
        assert!(y < self.height, "row {y} out of bounds (height {})", self.height);
        let start = y * self.stride;
        // Only the valid pixel portion, not the stride padding.
        &self.data[start..start + self.width]
    }

    /// Mutable borrow of a single row.
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        assert!(y < self.height, "row {y} out of bounds (height {})", self.height);
        let start = y * self.stride;
        &mut self.data[start..start + self.width]
    }

    /// Borrow a rectangular sub-region as an `ImageView`.
    ///
    /// LIFETIME DEEP DIVE:
    /// The returned `ImageView<'_, T>` borrows from this Image.
    /// The `'_` is an anonymous lifetime — the compiler infers that the
    /// view cannot outlive `&self`. Written explicitly, the signature is:
    ///
    ///   pub fn sub_image<'a>(&'a self, ...) -> ImageView<'a, T>
    ///
    /// This means: "the ImageView lives at most as long as the borrow of self."
    ///
    /// # Panics
    /// Panics if the sub-region extends beyond image bounds.
    pub fn sub_image(&self, x: usize, y: usize, w: usize, h: usize) -> ImageView<'_, T> {
        assert!(
            x + w <= self.width && y + h <= self.height,
            "sub_image region ({x},{y},{w},{h}) exceeds image bounds ({},{})",
            self.width,
            self.height,
        );
        // The view points into the parent's data slice, starting at (x, y).
        // It uses the *parent's* stride to navigate between rows.
        let start = y * self.stride + x;
        // We need enough data to cover h rows at the parent's stride.
        // Last row only needs w elements, but for simplicity we take the
        // full slice from start to the end of the last row's stride.
        let end = if h == 0 {
            start
        } else {
            (y + h - 1) * self.stride + self.width
        };
        ImageView {
            data: &self.data[start..end],
            width: w,
            height: h,
            // The parent's stride lets the view skip over
            // both the parent's padding AND the columns outside the sub-region.
            parent_stride: self.stride,
            // Offset from the start of each row in the data slice to the
            // first pixel of the view. For the first row this is 0 because
            // we already shifted `start` by x.
            x_offset: 0,
        }
    }

    /// Iterate over all pixels as `(x, y, value)` tuples.
    ///
    /// `impl Iterator<...>` is an "opaque return type" — the caller knows
    /// it gets an Iterator but doesn't need to know the concrete type.
    /// This is like returning a Box<dyn Iterator> but with zero overhead
    /// (the type is monomorphized at compile time).
    pub fn pixels(&self) -> impl Iterator<Item = (usize, usize, T)> + '_ {
        // We iterate row by row, skipping stride padding.
        (0..self.height).flat_map(move |y| {
            (0..self.width).map(move |x| (x, y, self.data[y * self.stride + x]))
        })
    }

    /// Access the underlying data as a flat slice.
    /// Includes stride padding. Useful for bulk operations.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable access to the underlying data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Total number of elements in the buffer (including stride padding).
    pub fn buffer_len(&self) -> usize {
        self.data.len()
    }

    // --- Internal helpers ---

    #[inline]
    fn bounds_check(&self, x: usize, y: usize) {
        assert!(
            x < self.width && y < self.height,
            "pixel ({x},{y}) out of bounds for image {}×{}",
            self.width,
            self.height,
        );
    }
}

// Debug formatting — useful for small images in tests.
impl<T: Pixel + fmt::Debug> fmt::Debug for Image<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Image<{}> {{ {}×{}, stride={} }}",
            std::any::type_name::<T>(),
            self.width,
            self.height,
            self.stride,
        )?;
        for y in 0..self.height.min(8) {
            write!(f, "  row {y}: [")?;
            for x in 0..self.width.min(16) {
                if x > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", self.get(x, y))?;
            }
            if self.width > 16 {
                write!(f, ", ...")?;
            }
            writeln!(f, "]")?;
        }
        if self.height > 8 {
            writeln!(f, "  ...")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Index / IndexMut — img[(x, y)] syntax
// ---------------------------------------------------------------------------
// Implementing std::ops::Index lets you use bracket syntax for pixel access.
// For Copy types (u8, f32), Rust auto-derefs &T → T transparently, so
// `img[(x, y)] + img[(x+1, y)]` works without manual dereferencing.

impl<T: Pixel> std::ops::Index<(usize, usize)> for Image<T> {
    type Output = T;

    #[inline]
    fn index(&self, (x, y): (usize, usize)) -> &T {
        self.bounds_check(x, y);
        &self.data[y * self.stride + x]
    }
}

impl<T: Pixel> std::ops::IndexMut<(usize, usize)> for Image<T> {
    #[inline]
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut T {
        self.bounds_check(x, y);
        let idx = y * self.stride + x;
        &mut self.data[idx]
    }
}

// ---------------------------------------------------------------------------
// ImageView<'a, T> — Borrowed sub-region of an Image
// ---------------------------------------------------------------------------
//
// LIFETIME EXPLANATION:
//
// The `'a` in `ImageView<'a, T>` is a lifetime parameter. It tells the
// compiler: "this ImageView contains a reference that borrows data which
// lives for at least the duration `'a`."
//
// In practice, `'a` is the lifetime of the `&Image<T>` you called
// `.sub_image()` on. The borrow checker ensures you cannot:
//   1. Drop or move the parent Image while a view exists.
//   2. Mutate the parent Image while a view exists (& vs &mut exclusivity).
//
// This is Rust's compile-time guarantee against dangling pointers and
// data races — what you'd have to manually verify in C/C++.

/// A borrowed, read-only view into a rectangular region of an `Image<T>`.
///
/// The view does NOT own its data — it borrows from a parent Image.
/// The lifetime `'a` ties the view to the parent's borrow.
pub struct ImageView<'a, T: Pixel> {
    /// Slice of the parent's data buffer, starting from the view's (0,0) pixel.
    data: &'a [T],
    /// Width of the viewed region.
    width: usize,
    /// Height of the viewed region.
    height: usize,
    /// Stride of the *parent* image (elements per row in the parent buffer).
    parent_stride: usize,
    /// Column offset within each row's slice. For top-level sub_image this is 0
    /// because we shift the data pointer. Kept for potential nested views.
    x_offset: usize,
}

impl<'a, T: Pixel> ImageView<'a, T> {
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the pixel at (x, y) within the view's coordinate system.
    ///
    /// (0, 0) is the top-left of the view, NOT the parent image.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> T {
        assert!(
            x < self.width && y < self.height,
            "ImageView pixel ({x},{y}) out of bounds for view {}×{}",
            self.width,
            self.height,
        );
        self.data[y * self.parent_stride + self.x_offset + x]
    }

    /// Iterate over all pixels in the view as `(x, y, value)`.
    pub fn pixels(&self) -> impl Iterator<Item = (usize, usize, T)> + '_ {
        (0..self.height).flat_map(move |y| {
            (0..self.width).map(move |x| {
                (x, y, self.data[y * self.parent_stride + self.x_offset + x])
            })
        })
    }

    /// Copy the view's pixels into a new owned Image.
    /// Useful when you need to pass a sub-region to a function that takes Image<T>.
    pub fn to_owned_image(&self) -> Image<T> {
        let mut img = Image::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                img.set(x, y, self.get(x, y));
            }
        }
        img
    }
}

impl<'a, T: Pixel + fmt::Debug> fmt::Debug for ImageView<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "ImageView<{}> {{ {}×{}, parent_stride={} }}",
            std::any::type_name::<T>(),
            self.width,
            self.height,
            self.parent_stride,
        )?;
        for y in 0..self.height.min(8) {
            write!(f, "  row {y}: [")?;
            for x in 0..self.width.min(16) {
                if x > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", self.get(x, y))?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

impl<'a, T: Pixel> std::ops::Index<(usize, usize)> for ImageView<'a, T> {
    type Output = T;

    #[inline]
    fn index(&self, (x, y): (usize, usize)) -> &T {
        assert!(
            x < self.width && y < self.height,
            "ImageView pixel ({x},{y}) out of bounds for view {}×{}",
            self.width,
            self.height,
        );
        &self.data[y * self.parent_stride + self.x_offset + x]
    }
}

// ---------------------------------------------------------------------------
// Bilinear Interpolation (needed by KLT in Step 5, but useful generally)
// ---------------------------------------------------------------------------

/// Bilinear interpolation for sub-pixel access on an f32 image.
///
/// Given floating-point coordinates (x, y), computes a weighted average
/// of the four surrounding integer-coordinate pixels.
///
/// **Boundary handling:** Clamps coordinates to the image boundary
/// (replicates edge pixels). This means querying at x = width-1 or
/// y = height-1 is safe — the out-of-bounds neighbor is replaced by
/// the edge pixel. This matches vilib's border strategy and is benign
/// for LK tracking, where clamped boundary contributions in the error
/// and gradient images cancel each other out.
///
/// # Panics
/// Panics if the image is empty (width or height is 0).
pub fn interpolate_bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    assert!(img.width() > 0 && img.height() > 0, "cannot interpolate on an empty image");

    // Clamp to valid coordinate range.
    let max_x = (img.width() - 1) as f32;
    let max_y = (img.height() - 1) as f32;
    let x = x.clamp(0.0, max_x);
    let y = y.clamp(0.0, max_y);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    // Clamp x1/y1 so we don't exceed bounds at the right/bottom edge.
    let x1 = (x0 + 1).min(img.width() - 1);
    let y1 = (y0 + 1).min(img.height() - 1);

    // SAFETY: x0, x1 < width and y0, y1 < height after clamping.
    unsafe {
        let p00 = img.get_unchecked(x0, y0);
        let p10 = img.get_unchecked(x1, y0);
        let p01 = img.get_unchecked(x0, y1);
        let p11 = img.get_unchecked(x1, y1);
        (1.0 - fx) * (1.0 - fy) * p00
            + fx * (1.0 - fy) * p10
            + (1.0 - fx) * fy * p01
            + fx * fy * p11
    }
}

/// Bilinear interpolation without the assert or clamp overhead.
///
/// # Safety
/// Caller must guarantee:
///   - img is non-empty
///   - x is in [0.0, width-1] and y is in [0.0, height-1]
///
/// Used in the KLT inner loop where the window bounds are validated
/// once at the start and coordinates stay in range.
///
/// GPU EQUIVALENT: Hardware texture sampling with clamp-to-edge mode.
#[inline(always)]
pub unsafe fn interpolate_bilinear_unchecked(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let x0 = x as usize; // floor for non-negative
    let y0 = y as usize;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let w = img.width();
    let x1 = if x0 + 1 < w { x0 + 1 } else { x0 };
    let h = img.height();
    let y1 = if y0 + 1 < h { y0 + 1 } else { y0 };

    let p00 = img.get_unchecked(x0, y0);
    let p10 = img.get_unchecked(x1, y0);
    let p01 = img.get_unchecked(x0, y1);
    let p11 = img.get_unchecked(x1, y1);

    (1.0 - fx) * (1.0 - fy) * p00
        + fx * (1.0 - fy) * p10
        + (1.0 - fx) * fy * p01
        + fx * fy * p11
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_u8() {
        let img: Image<u8> = Image::new(10, 5);
        assert_eq!(img.width(), 10);
        assert_eq!(img.height(), 5);
        assert_eq!(img.stride(), 10);
        // All pixels should be zero-initialized.
        for (_, _, v) in img.pixels() {
            assert_eq!(v, 0u8);
        }
    }

    #[test]
    fn test_roundtrip() {
        let mut img: Image<u8> = Image::new(4, 3);
        img.set(0, 0, 10);
        img.set(3, 2, 255);
        img.set(1, 1, 42);
        assert_eq!(img.get(0, 0), 10);
        assert_eq!(img.get(3, 2), 255);
        assert_eq!(img.get(1, 1), 42);
        assert_eq!(img.get(2, 2), 0); // untouched pixel
    }

    #[test]
    fn test_from_vec() {
        let data: Vec<u8> = (0..12).collect();
        let img = Image::from_vec(4, 3, data);
        // Row 0: [0, 1, 2, 3], Row 1: [4, 5, 6, 7], Row 2: [8, 9, 10, 11]
        assert_eq!(img.get(0, 0), 0);
        assert_eq!(img.get(3, 0), 3);
        assert_eq!(img.get(0, 1), 4);
        assert_eq!(img.get(3, 2), 11);
    }

    #[test]
    fn test_stride_padding() {
        // stride=8, width=4 → 4 padding elements per row
        let img: Image<u8> = Image::new_with_stride(4, 3, 8);
        assert_eq!(img.stride(), 8);
        assert_eq!(img.buffer_len(), 3 * 8); // 24 elements total
    }

    #[test]
    fn test_row_slice() {
        let data: Vec<u8> = (0..12).collect();
        let img = Image::from_vec(4, 3, data);
        assert_eq!(img.row(0), &[0, 1, 2, 3]);
        assert_eq!(img.row(1), &[4, 5, 6, 7]);
        assert_eq!(img.row(2), &[8, 9, 10, 11]);
    }

    #[test]
    fn test_sub_image_basic() {
        // 4×4 image:
        //   0  1  2  3
        //   4  5  6  7
        //   8  9 10 11
        //  12 13 14 15
        let data: Vec<u8> = (0..16).collect();
        let img = Image::from_vec(4, 4, data);

        // 2×2 sub-image starting at (1, 1)
        let view = img.sub_image(1, 1, 2, 2);
        assert_eq!(view.width(), 2);
        assert_eq!(view.height(), 2);
        assert_eq!(view.get(0, 0), 5);  // img(1,1)
        assert_eq!(view.get(1, 0), 6);  // img(2,1)
        assert_eq!(view.get(0, 1), 9);  // img(1,2)
        assert_eq!(view.get(1, 1), 10); // img(2,2)
    }

    #[test]
    fn test_sub_image_to_owned() {
        let data: Vec<u8> = (0..16).collect();
        let img = Image::from_vec(4, 4, data);
        let view = img.sub_image(1, 1, 2, 2);
        let owned = view.to_owned_image();
        assert_eq!(owned.width(), 2);
        assert_eq!(owned.height(), 2);
        assert_eq!(owned.get(0, 0), 5);
        assert_eq!(owned.get(1, 1), 10);
    }

    #[test]
    fn test_pixels_iterator() {
        let data: Vec<u8> = (0..6).collect();
        let img = Image::from_vec(3, 2, data);
        let pixels: Vec<_> = img.pixels().collect();
        assert_eq!(pixels.len(), 6);
        assert_eq!(pixels[0], (0, 0, 0));
        assert_eq!(pixels[1], (1, 0, 1));
        assert_eq!(pixels[2], (2, 0, 2));
        assert_eq!(pixels[3], (0, 1, 3));
    }

    #[test]
    fn test_f32_image() {
        let mut img: Image<f32> = Image::new(3, 3);
        img.set(1, 1, 0.5);
        assert_eq!(img.get(1, 1), 0.5f32);
        assert_eq!(img.get(0, 0), 0.0f32);
    }

    #[test]
    fn test_bilinear_at_integer() {
        // At integer coordinates, bilinear should return the exact pixel value.
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let img = Image::from_vec(3, 3, data);
        assert!((interpolate_bilinear(&img, 0.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((interpolate_bilinear(&img, 1.0, 1.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_bilinear_midpoint() {
        // At the midpoint of four pixels, should be the average.
        let data: Vec<f32> = vec![0.0, 10.0, 20.0, 30.0];
        let img = Image::from_vec(2, 2, data);
        let v = interpolate_bilinear(&img, 0.5, 0.5);
        // (0 + 10 + 20 + 30) / 4 = 15
        assert!((v - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_bilinear_boundary_clamp() {
        // At the right/bottom edge, should clamp and return the edge pixel.
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let img = Image::from_vec(2, 2, data);

        // Exactly at (1.0, 1.0) — bottom-right pixel. x1/y1 clamp to 1.
        let v = interpolate_bilinear(&img, 1.0, 1.0);
        assert!((v - 4.0).abs() < 1e-6);

        // Beyond bounds — clamped back to edge.
        let v = interpolate_bilinear(&img, 5.0, 5.0);
        assert!((v - 4.0).abs() < 1e-6);

        // Negative — clamped to (0, 0).
        let v = interpolate_bilinear(&img, -1.0, -1.0);
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bilinear_edge_interpolation() {
        // At x = width-1, fy blending should still work along the edge.
        let data: Vec<f32> = vec![0.0, 10.0, 0.0, 20.0];
        let img = Image::from_vec(2, 2, data);
        // x=1.0 (right edge), y=0.5 (midway vertically)
        // Top-right = 10.0, bottom-right = 20.0 → blends to 15.0
        let v = interpolate_bilinear(&img, 1.0, 0.5);
        assert!((v - 15.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_get_out_of_bounds() {
        let img: Image<u8> = Image::new(4, 4);
        img.get(4, 0); // x == width → out of bounds
    }

    #[test]
    #[should_panic(expected = "stride")]
    fn test_stride_less_than_width() {
        let _img: Image<u8> = Image::new_with_stride(10, 5, 8); // stride < width
    }

    #[test]
    fn test_index_read() {
        let data: Vec<u8> = (0..12).collect();
        let img = Image::from_vec(4, 3, data);
        assert_eq!(img[(0, 0)], 0);
        assert_eq!(img[(3, 0)], 3);
        assert_eq!(img[(0, 1)], 4);
        assert_eq!(img[(3, 2)], 11);
    }

    #[test]
    fn test_index_mut_write() {
        let mut img: Image<u8> = Image::new(4, 3);
        img[(1, 2)] = 42;
        assert_eq!(img[(1, 2)], 42);
        assert_eq!(img.get(1, 2), 42); // consistent with get()
    }

    #[test]
    fn test_index_arithmetic() {
        // Verify auto-deref works seamlessly in expressions.
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let img = Image::from_vec(2, 2, data);
        let sum = img[(0, 0)] + img[(1, 0)] + img[(0, 1)] + img[(1, 1)];
        assert!((sum - 10.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_index_out_of_bounds() {
        let img: Image<u8> = Image::new(4, 4);
        let _ = img[(4, 0)];
    }

    #[test]
    fn test_imageview_index() {
        let data: Vec<u8> = (0..16).collect();
        let img = Image::from_vec(4, 4, data);
        let view = img.sub_image(1, 1, 2, 2);
        assert_eq!(view[(0, 0)], 5);
        assert_eq!(view[(1, 1)], 10);
    }
}
