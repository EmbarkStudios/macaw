//! # Linear color space
//! The preferred color space is linear 0-1 RGB, encoded with f32.
//! Use Vec3 for RGB and Vec4 for RGBA. Linear space allows you to add and multiply color.
//! Any color encoded as a float should be in this linear 0-1 space.
//!
//! # What is sRGB
//! If you want to compress a color to take up less space, there
//! is a standard called sRGB which compresses the 0-1 float range into a single byte.
//! It makes NO SENSE to multiply or add bytes in the compressed sRGB format.
//! You ALWAYS want to convert to linear space before working on colors.
//!
//! # Alpha
//! Many colors come with a fourth "alpha" component, which controls how opaque it is.
//! The alpha component is a linear multiplier and never undergoes sRGB compression,
//! but instead the 0-255 byte range is linearly mapped to the 0-1 range.
//!
//! # Pre-multiplied alpha
//! Alpha-blended images should use pre-multiplied alpha,
//! meaning the alpha has already been multiplied with the RGB triplet.
//! This means for instance that [0.5, 0.5, 0.5, 0.5] is transparent WHITE, NOT transparent GRAY.
//! In premulitplied alpha the RGB linear triplet values are smaller or equal to the alpha.
//! If the RGB triplet is larger you will get addative blending (which is cool).
//! For instance `[1.0, 0.0, 0.0, 0.0]` is red without any opaquness.
//! So it won't cover whatever is behind it, but will ADD to it.

#[cfg(feature = "with_bytemuck")]
use bytemuck::Pod;
#[cfg(feature = "with_bytemuck")]
use bytemuck::Zeroable;

use crate::Vec4;

/// A compressed sRGBA color, 8-bit per component, 32-bit total.
///
/// This should only be used when space is an issue, i.e. when compressing data.
/// Otherwise prefer a [`Vec4`] linear space in the [0-1] range,
/// as that allows you to add and multiply colors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "with_serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with_speedy", derive(speedy::Writable, speedy::Readable))]
#[repr(transparent)]
pub struct ColorRgba8(pub [u8; 4]);

impl From<ColorRgba8> for u32 {
    fn from(c: ColorRgba8) -> Self {
        Self::from(c.0[0]) << 24
            | Self::from(c.0[1]) << 16
            | Self::from(c.0[2]) << 8
            | Self::from(c.0[3])
    }
}

impl From<u32> for ColorRgba8 {
    fn from(c: u32) -> Self {
        Self([(c >> 24) as u8, (c >> 16) as u8, (c >> 8) as u8, c as u8])
    }
}

impl From<[u8; 4]> for ColorRgba8 {
    fn from(srgba: [u8; 4]) -> Self {
        Self(srgba)
    }
}

impl From<ColorRgba8> for [u8; 4] {
    fn from(srgba: ColorRgba8) -> Self {
        srgba.0
    }
}

#[cfg(feature = "with_bytemuck")]
// SAFETY: A `[u8; N]` is always Pod, and this is a transparent wrapper.
unsafe impl Pod for ColorRgba8 {}

#[cfg(feature = "with_bytemuck")]
// SAFETY: A `[u8; N]` is always Zeroable, and this is a transparent wrapper.
unsafe impl Zeroable for ColorRgba8 {}

/// sRGBA from linear RGBA in [0-1] range
impl From<Vec4> for ColorRgba8 {
    fn from(v: Vec4) -> Self {
        let a = v.w; // Note: alpha is always linear
        Self([
            srgb_byte_from_linear(v.x),
            srgb_byte_from_linear(v.y),
            srgb_byte_from_linear(v.z),
            if a > 1.0 {
                255
            } else if a <= 0.0 {
                0
            } else {
                (a * 255.0).round() as u8
            },
        ])
    }
}

/// sRGBA from linear RGBA in [0-1] range
impl From<[f32; 4]> for ColorRgba8 {
    fn from(c: [f32; 4]) -> Self {
        let a = c[3]; // Note: alpha is always linear
        Self([
            srgb_byte_from_linear(c[0]),
            srgb_byte_from_linear(c[1]),
            srgb_byte_from_linear(c[2]),
            if a > 1.0 {
                255
            } else if a <= 0.0 {
                0
            } else {
                (a * 255.0).round() as u8
            },
        ])
    }
}

/// Linear RGBA in 0-1 from sRGBA
impl From<ColorRgba8> for Vec4 {
    fn from(c: ColorRgba8) -> Self {
        // Note: alpha is always linear
        Self::new(
            linear_from_srgb_byte(c.0[0]),
            linear_from_srgb_byte(c.0[1]),
            linear_from_srgb_byte(c.0[2]),
            f32::from(c.0[3]) / 255.0,
        )
    }
}

/// Decodes 0-255 sRGB space to 0-1 linear space
#[inline]
fn linear_from_srgb_byte(s: u8) -> f32 {
    if s <= 10 {
        f32::from(s) / 3_294.6
    } else {
        ((f32::from(s) + 14.025) / 269.025).powf(2.4)
    }
}

/// Encodes 0-1 linear space as 0-255 sRGB space
#[inline]
fn srgb_byte_from_linear(l: f32) -> u8 {
    if l <= 0.0 {
        0
    } else if l <= 0.003_130_8 {
        (3_294.6 * l).round() as u8
    } else if l <= 1.0 {
        (269.025 * l.powf(1.0 / 2.4) - 14.025).round() as u8
    } else {
        255
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_convert() {
        let colors = [
            ColorRgba8([255, 0, 127, 1]),
            ColorRgba8([1, 2, 3, 4]),
            ColorRgba8([0, 0, 0, 0]),
            ColorRgba8([255, 254, 253, 252]),
        ];

        for c in &colors {
            let b = u32::from(*c);
            assert_eq!(*c, ColorRgba8::from(b));

            let v = Vec4::from(*c);
            assert!(v.min_element() >= 0.0 && v.max_element() <= 1.0); // must be normalized [0-1], as source can only be 0-255
            assert_eq!(*c, ColorRgba8::from(v));
        }

        // test some HDR color ranges that need to be clamped properly
        assert_eq!(
            ColorRgba8::from([-5.0, 0.0, 1.0, 100.0]),
            ColorRgba8([0, 0, 255, 255])
        );
        assert_eq!(
            ColorRgba8::from([100.0, 0.0, 0.1, 1.0]),
            ColorRgba8([255, 0, srgb_byte_from_linear(0.1), 255])
        );
        assert_eq!(
            ColorRgba8::from([100.0, 0.0, -0.1, -0.1]),
            ColorRgba8([255, 0, 0, 0])
        );
    }

    #[test]
    fn test_srgba() {
        #![allow(clippy::float_cmp)]

        assert_eq!(linear_from_srgb_byte(0), 0.0);
        assert!(linear_from_srgb_byte(1) > 0.0);
        assert!(linear_from_srgb_byte(254) < 1.0);
        assert_eq!(linear_from_srgb_byte(255), 1.0);

        assert_eq!(srgb_byte_from_linear(-1.0), 0);
        assert_eq!(srgb_byte_from_linear(0.0), 0);
        assert_eq!(srgb_byte_from_linear(1.0), 255);
        assert_eq!(srgb_byte_from_linear(1.1), 255);
        assert_eq!(srgb_byte_from_linear(2.0), 255);

        for b in 0..=255_u8 {
            let l = linear_from_srgb_byte(b);
            assert!((0.0..=1.0).contains(&l));
            assert_eq!(srgb_byte_from_linear(l), b);
        }
    }
}
