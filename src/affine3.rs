use glam::Affine3A;
use glam::Mat3;
use glam::Quat;
use glam::Vec3;

use core::ops::*;

use crate::IsoTransform;
use crate::Mat3Ext;

/// A type which has the same representation on the CPU and GPU for the underlying
/// storage of an [`Affine3`].
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "bytemuck",
    derive(bytemuck::NoUninit, bytemuck::AnyBitPattern)
)]
#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[repr(C, align(16))]
pub struct Affine3Storage(pub [f32; 12]);

impl Affine3Storage {
    // must compile same on both spirv and cpu
    const fn _assert_repr() {
        let _: [(); core::mem::size_of::<Self>()] = [(); 48];
        let _: [(); core::mem::align_of::<Self>()] = [(); 16];
    }

    #[inline]
    pub const fn unpack(self) -> Affine3 {
        Affine3::from_storage(self)
    }
}

/// The same as [`Affine3A`] except using the non-aligned versions of the underlying types.
/// Useful when doing interop with the GPU. Note that this unfortunately has a different
/// repr calculated on the CPU and GPU, so you still need to use [`Affine3Storage`] as a
/// intermediary.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
#[cfg_attr(
    feature = "bytemuck",
    derive(bytemuck::NoUninit, bytemuck::AnyBitPattern)
)]
#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[repr(C, align(4))]
pub struct Affine3 {
    pub mat3: Mat3,
    pub translation: Vec3,
}

impl Affine3 {
    pub const IDENTITY: Self = Self {
        mat3: Mat3::IDENTITY,
        translation: Vec3::ZERO,
    };

    #[inline]
    pub fn from_affine3a(a: Affine3A) -> Self {
        Self {
            mat3: Mat3::from(a.matrix3),
            translation: a.translation.into(),
        }
    }

    #[inline]
    pub fn from_scale_rotation_translation(scale: Vec3, rotation: Quat, translation: Vec3) -> Self {
        let mat3 = Mat3::from_quat(rotation).mul_diagonal_scale(scale);
        Self { mat3, translation }
    }

    #[inline]
    pub fn from_iso_transform(iso: IsoTransform) -> Self {
        Self {
            mat3: Mat3::from_quat(iso.rotation),
            translation: iso.translation(),
        }
    }

    #[inline]
    pub fn left_mul_diagonal_scale(&self, scale: Vec3) -> Self {
        Self {
            mat3: self.mat3.mul_diagonal_scale(scale),
            translation: self.translation * scale,
        }
    }

    #[inline]
    pub const fn from_storage(storage: Affine3Storage) -> Self {
        let a = storage.0;
        Self {
            mat3: Mat3::from_cols(
                Vec3::new(a[0], a[1], a[2]),
                Vec3::new(a[3], a[4], a[5]),
                Vec3::new(a[6], a[7], a[8]),
            ),
            translation: Vec3::new(a[9], a[10], a[11]),
        }
    }

    #[inline]
    pub const fn const_to_storage(self) -> Affine3Storage {
        Affine3Storage([
            self.mat3.x_axis.x,
            self.mat3.x_axis.y,
            self.mat3.x_axis.z,
            self.mat3.y_axis.x,
            self.mat3.y_axis.y,
            self.mat3.y_axis.z,
            self.mat3.z_axis.x,
            self.mat3.z_axis.y,
            self.mat3.z_axis.z,
            self.translation.x,
            self.translation.y,
            self.translation.z,
        ])
    }

    #[cfg(all(not(target_arch = "spirv"), feature = "bytemuck"))]
    #[inline]
    pub fn cast_to_storage(self) -> Affine3Storage {
        bytemuck::cast(self)
    }
}

impl Mul<f32> for Affine3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            mat3: self.mat3 * rhs,
            translation: self.translation * rhs,
        }
    }
}

impl Mul<Affine3> for f32 {
    type Output = Affine3;
    #[inline]
    fn mul(self, rhs: Affine3) -> Self::Output {
        rhs.mul(self)
    }
}

impl Mul<Vec3> for Affine3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, rhs: Vec3) -> Self::Output {
        (self.mat3.x_axis * rhs.x)
            + (self.mat3.y_axis * rhs.y)
            + (self.mat3.z_axis * rhs.z)
            + self.translation
    }
}

impl Add for Affine3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            mat3: self.mat3 + rhs.mat3,
            translation: self.translation + rhs.translation,
        }
    }
}

impl AddAssign for Affine3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.mat3 += rhs.mat3;
        self.translation += rhs.translation;
    }
}
