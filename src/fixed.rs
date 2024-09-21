#![allow(clippy::manual_range_contains)]
#![allow(clippy::cast_lossless)]

use core::ops::*;

const U16_MAX: f32 = u16::MAX as f32;
const U8_MAX: f32 = u8::MAX as f32;

#[derive(Debug, Clone)]
pub enum UNormError {
    UnnormalizedFloat,
}

impl core::fmt::Display for UNormError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl std::error::Error for UNormError {}

/// A 16-bit fixed point value in the range `[0.0..=1.0]` (inclusive).
///
/// Basic arithmetic operations are implemented. Multiplication of two
/// `UNorm16` is always defined. However, addition, subtraction, and
/// division can very easily overflow or underflow, so be careful!
#[derive(Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "with_serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with_serde", serde(transparent))]
#[cfg_attr(feature = "with_speedy", derive(speedy::Readable, speedy::Writable))]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(transparent)]
pub struct UNorm16(pub u16);

impl UNorm16 {
    #[inline]
    pub fn new(x: f32) -> Result<Self, UNormError> {
        if !x.is_nan() && x >= 0.0 && x <= 1.0 {
            Ok(Self::new_unchecked(x))
        } else {
            Err(UNormError::UnnormalizedFloat)
        }
    }

    #[inline]
    pub fn new_clamped(x: f32) -> Self {
        Self::new_unchecked(x.clamp(0.0, 1.0))
    }

    #[inline(always)]
    pub fn new_unchecked(x: f32) -> Self {
        Self((x * U16_MAX) as u16)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / U16_MAX
    }
}

impl Add<Self> for UNorm16 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for UNorm16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub<Self> for UNorm16 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for UNorm16 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Mul<Self> for UNorm16 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        // numerator = numerator * numerator = self * rhs
        // denominator = denominator * denominator = 2^16 * 2^16
        // we want numerator / 2^16, so need to divide numerator by 2^16 (aka rsh 16)
        // this can't overflow :)
        let numerator: u32 = self.0 as u32 * rhs.0 as u32;
        let output: u32 = numerator >> 16;
        Self(output as u16)
    }
}

impl MulAssign for UNorm16 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div<Self> for UNorm16 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        // numerator = numerator / numerator = self / rhs
        // denominator = denominator / denominator = 2^16 / 2^16 = 1
        // we want numerator / 2^16 so need to multiply numerator by 2^16 (aka lsh 16)
        // we do this before the division to preserve as much precision as we can.
        // the result can overflow u16 very easily...
        let output: u32 = ((self.0 as u32) << 16) / (rhs.0 as u32);
        Self(output as u16)
    }
}

impl DivAssign for UNorm16 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

#[cfg(not(target_arch = "spirv"))]
impl core::fmt::Debug for UNorm16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f32())?;
        if f.alternate() {
            write!(f, " ({}/{})", self.0, u16::MAX)?;
        }
        Ok(())
    }
}

/// An 8-bit fixed point value in the range `[0.0..=1.0]` (inclusive).
///
/// Basic arithmetic operations are implemented. Multiplication of two
/// `UNorm8` is always defined. However, addition, subtraction, and
/// division can very easily overflow or underflow, so be careful!
#[derive(Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "with_serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with_serde", serde(transparent))]
#[cfg_attr(feature = "with_speedy", derive(speedy::Readable, speedy::Writable))]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(transparent)]
pub struct UNorm8(pub u8);

impl UNorm8 {
    #[inline]
    pub fn new(x: f32) -> Result<Self, UNormError> {
        if !x.is_nan() && x >= 0.0 && x <= 1.0 {
            Ok(Self::new_unchecked(x))
        } else {
            Err(UNormError::UnnormalizedFloat)
        }
    }

    #[inline]
    pub fn new_clamped(x: f32) -> Self {
        Self::new_unchecked(x.clamp(0.0, 1.0))
    }

    #[inline(always)]
    pub fn new_unchecked(x: f32) -> Self {
        Self((x * U8_MAX) as u8)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / U8_MAX
    }
}

impl Add<Self> for UNorm8 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for UNorm8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub<Self> for UNorm8 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for UNorm8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Mul<Self> for UNorm8 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        // numerator = numerator * numerator = self * rhs
        // denominator = denominator * denominator = 2^8 * 2^8
        // we want denominator = 2^8, so need to divide both by 2^8 (aka rsh 8)
        // this can't overflow :)
        let numerator: u32 = self.0 as u32 * rhs.0 as u32;
        let output: u32 = numerator >> 8;
        Self(output as u8)
    }
}

impl MulAssign for UNorm8 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div<Self> for UNorm8 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        // numerator = numerator / numerator = self / rhs
        // denominator = denominator / denominator = 2^8 / 2^8 = 1
        // we want denominator = 2^8 so need to multiply both by 2^8 (aka lsh 8)
        // we do this before the division to preserve as much precision as we can.
        // the result can overflow u8 very easily...
        let output: u32 = ((self.0 as u32) << 8) / (rhs.0 as u32);
        Self(output as u8)
    }
}

impl DivAssign for UNorm8 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

#[cfg(not(target_arch = "spirv"))]
impl core::fmt::Debug for UNorm8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f32())?;
        if f.alternate() {
            write!(f, " ({}/{})", self.0, u8::MAX)?;
        }
        Ok(())
    }
}
