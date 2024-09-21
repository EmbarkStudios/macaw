use core::f32::consts::LOG2_E;
use core::ops::Add;
use core::ops::Mul;
use core::ops::RangeInclusive;

/// Linear interpolation between a range
pub fn lerp<T>(range: RangeInclusive<T>, t: f32) -> T
where
    f32: Mul<T, Output = T>,
    T: Add<T, Output = T> + Copy,
{
    (1.0 - t) * *range.start() + t * *range.end()
}

/// Remap a value from one range to another, e.g. do a linear transform.
///
/// # Example
///
/// ```
/// # use macaw::remap;
/// let ocean_height = remap(0.2, -1.0..=1.0, 2.0..=3.1);
/// ```
///
/// # From range requirement
///
/// The range has to be from a low value to a high value, such as 0..=1.0, NOT 1.0..=0.0.
/// If it is outside of that range the results will be undefined
pub fn remap(x: f32, from: RangeInclusive<f32>, to: RangeInclusive<f32>) -> f32 {
    let t = (x - from.start()) / (from.end() - from.start());
    lerp(to, t)
}

/// Remap a value from one range to another, clamps the input value to be in the from range first.
///
/// # Example
///
/// ```
/// # use macaw::remap_clamp;
/// let ocean_height = remap_clamp(0.2, -1.0..=1.0, 2.0..=3.1);
/// ```
///
/// # From range requirement
///
/// The range has to be from a low value to a high value, such as 0..=1.0, NOT 1.0..=0.0.
/// If it is outside of that range the results will be undefined
#[inline]
pub fn remap_clamp(x: f32, from: RangeInclusive<f32>, to: RangeInclusive<f32>) -> f32 {
    if x <= *from.start() {
        *to.start()
    } else if *from.end() <= x {
        *to.end()
    } else {
        let t = (x - from.start()) / (from.end() - from.start());
        // Ensure no numerical inaccurcies sneak in:
        if 1.0 <= t {
            *to.end()
        } else {
            lerp(to, t)
        }
    }
}

/// Aka "lerp smoothing". Meant to be called per-frame, to interpolate between
/// `curr` and `target`. `decay_rate` controls the speed of interpolation, with a usable
/// range of around 1.0 and 25.0, from slow to fast. `dt` is the current frame time.
#[inline(always)]
pub fn exp_decay(curr: f32, target: f32, decay_rate: f32, dt: f32) -> f32 {
    target + (curr - target) * exp_fast(-decay_rate * dt)
}

/// Fast approximation of base 2 logarithm.
#[inline(always)]
pub fn log2_fast(x: f32) -> f32 {
    let vx = f32::to_bits(x);
    let mx = f32::from_bits((vx & 0x007FFFFF_u32) | 0x3f000000);
    let mut y = vx as f32;
    y *= 1.192_092_9e-7_f32;
    y - 124.225_52_f32 - 1.498_030_3_f32 * mx - 1.725_88_f32 / (0.352_088_72_f32 + mx)
}

/// Fast approximation of natural logarithm.
#[inline(always)]
pub fn ln_fast(x: f32) -> f32 {
    core::f32::consts::LN_2 * log2_fast(x)
}

/// Fast approximation of e^x
#[inline(always)]
pub fn exp_fast(p: f32) -> f32 {
    exp2_fast(p * LOG2_E)
}

/// Fast approximation of exponentiating 2 to a floating point power.
#[inline(always)]
pub fn exp2_fast(p: f32) -> f32 {
    let offset = if p < 0.0 { 1.0_f32 } else { 0.0_f32 };
    let clipp = if p < -126.0 { -126.0_f32 } else { p };
    let w = clipp as i32;
    let z = clipp - (w as f32) + offset;
    let v = ((1 << 23) as f32
        * (clipp + 121.274055_f32 + 27.728024_f32 / (4.8425255_f32 - z) - 1.4901291_f32 * z))
        as u32;
    f32::from_bits(v)
}

/// Raises a number to a floating point power.
#[inline(always)]
pub fn powf_fast(x: f32, p: f32) -> f32 {
    exp2_fast(p * log2_fast(x))
}

#[allow(clippy::float_cmp)]
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn remapping() {
        // verify simple case
        assert_eq!(remap(0.2, -1.0..=1.0, -2.0..=2.0), 0.4000001);
        assert_eq!(remap_clamp(0.2, -1.0..=1.0, -2.0..=2.0), 0.4000001);
        // out of range
        assert_eq!(remap(3.0, -1.0..=1.0, -2.0..=2.0), 6.0);
        assert_eq!(remap_clamp(3.0, -1.0..=1.0, -2.0..=2.0), 2.0);
        // invalid remapping
        assert!(remap(0.0, 0.0..=0.0, -2.0..=2.0).is_nan());
        assert_eq!(remap_clamp(0.0, 0.0..=0.0, -2.0..=2.0), -2.0);
    }
}
