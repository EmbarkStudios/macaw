//! Dual scalar and dual quaternion implementation.
//!
//! References:
//! - <https://users.cs.utah.edu/~ladislav/kavan07skinning/kavan07skinning.pdf>
//! - <https://faculty.sites.iastate.edu/jia/files/inline-files/dual-quaternion.pdf>
//! - <https://stackoverflow.com/questions/23174899/properly-normalizing-a-dual-quaternion>
//! - <http://wscg.zcu.cz/wscg2012/short/a29-full.pdf>
//! - <https://borodust.github.io/public/shared/paper_dual-quats.pdf>
//! - <https://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf>

use crate::IsoTransform;
use crate::Quat;
use crate::Vec3;
use crate::Vec4;
use crate::Vec4Swizzles;

#[cfg(target_arch = "spirv")]
use num_traits::Float;

/// A dual scalar, with a real and dual part
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "with_serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with_speedy", derive(speedy::Writable, speedy::Readable))]
#[cfg_attr(
    feature = "bytemuck",
    derive(bytemuck::NoUninit, bytemuck::AnyBitPattern)
)]
#[repr(C)]
pub struct DualScalar {
    /// The coefficient of the real part
    pub real: f32,
    /// The coefficient of the dual part
    pub dual: f32,
}

impl DualScalar {
    /// Defined only for dual numbers with a positive real part.
    #[inline]
    pub fn sqrt(self) -> Self {
        let real_sqrt = self.real.sqrt();

        let dual = self.dual / 2.0 * real_sqrt;

        Self {
            real: real_sqrt,
            dual,
        }
    }

    /// More efficient than `.sqrt().inverse()`. Defined only for dual numbers with positive,
    /// non-zero real part.
    #[inline]
    pub fn inverse_sqrt(self) -> Self {
        let real_sqrt = self.real.sqrt();

        Self {
            real: 1.0 / real_sqrt,
            dual: -self.dual / (2.0 * self.real * real_sqrt),
        }
    }

    /// Gives the inverse of `self` such that the inverse multiplied with `self` will equal
    /// `(1.0, 0.0)`.
    #[inline]
    pub fn inverse(self) -> Self {
        let real_inv = 1.0 / self.real;
        Self {
            real: real_inv,
            dual: -self.dual * real_inv * real_inv,
        }
    }
}

impl core::ops::Mul for DualScalar {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real * rhs.real,
            dual: self.real * rhs.dual + self.dual * rhs.real,
        }
    }
}

impl core::ops::Add for DualScalar {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real + rhs.real,
            dual: self.dual + rhs.dual,
        }
    }
}

impl core::ops::Sub for DualScalar {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real - rhs.real,
            dual: self.dual - rhs.dual,
        }
    }
}

impl core::ops::Mul<f32> for DualScalar {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            real: self.real * rhs,
            dual: self.dual * rhs,
        }
    }
}

/// Represents a rigid body transformation which can be thought of as a "screw" motion
/// which is the combination of a translation along a vector and a rotation around that vector.
///
/// Represents the same kind of transformation as an `IsoTransform` but interpolates and transforms
/// in a way that preserves volume.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "with_serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with_speedy", derive(speedy::Writable, speedy::Readable))]
#[cfg_attr(
    feature = "bytemuck",
    derive(bytemuck::NoUninit, bytemuck::AnyBitPattern)
)]
#[repr(C, align(16))]
pub struct DualQuat {
    /// The real quaternion part, i.e. the rotator
    pub real: Quat,
    /// The dual quaternion part, i.e. the translator
    pub dual: Quat,
}

impl DualQuat {
    const fn _assert_repr() {
        let _: [(); core::mem::size_of::<Self>()] = [(); 32];
        let _: [(); core::mem::align_of::<Self>()] = [(); 16];
    }

    /// The identity transform: doesn't transform at all. Like multiplying with `1`.
    pub const IDENTITY: Self = Self {
        real: Quat::IDENTITY,
        dual: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
    };

    /// Not a valid quaternion in and of itself. All values filled with 0s.
    /// Can be useful when blending a set of transforms.
    pub const ZERO: Self = Self {
        real: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
        dual: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
    };

    /// Create a dual quaternion from an [`IsoTransform`]
    #[inline]
    pub fn from_iso_transform(iso_transform: IsoTransform) -> Self {
        Self::from_rotation_translation(iso_transform.rotation, iso_transform.translation())
    }

    /// Create a dual quaternion that rotates and then translates by the specified amount.
    /// `rotation` is assumed to be normalized.
    #[inline]
    pub fn from_rotation_translation(rotation: Quat, translation: Vec3) -> Self {
        Self {
            real: rotation,
            dual: Quat::from_vec4((translation * 0.5).extend(0.0)) * rotation,
        }
    }

    /// A pure translation without any rotation.
    #[inline]
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            real: Quat::IDENTITY,
            dual: Quat::from_vec4((translation * 0.5).extend(0.0)),
        }
    }

    /// A pure rotation without any translation.
    #[inline]
    pub fn from_quat(rotation: Quat) -> Self {
        Self {
            real: rotation,
            dual: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
        }
    }

    /// Returns this transform decomposed into `(rotation, translation)`. *Assumes `self` is
    /// already normalized.* If not, consider
    /// using [`normalize_to_rotation_translation`][Self::normalize_to_rotation_translation]
    ///
    /// You can then apply this to a point by doing `rotation.mul_vec3(point) + translation`.
    #[inline]
    pub fn to_rotation_translation(self) -> (Quat, Vec3) {
        let translation = 2.0 * Vec4::from(self.dual * self.real.conjugate()).xyz();
        (self.real, translation)
    }

    /// Quaternion conjugate of this dual quaternion, which is given `q = (real + dual)`,
    /// `q* = (real* + dual*)`. This form of conjugate is also the inverse as long as this
    /// dual quaternion is unit.
    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            real: self.real.conjugate(),
            dual: self.dual.conjugate(),
        }
    }

    /// Gives the inverse of this dual quaternion such that `self * self.inverse() = identity`.
    ///
    /// This function should only be used if self is unit, i.e if `self.is_normalized()` is true.
    /// Will panic in debug builds if it is not normalized.
    #[inline]
    pub fn inverse(self) -> Self {
        debug_assert!(self.is_normalized());
        self.conjugate()
    }

    /// Gives the norm squared of the dual quaternion.
    ///
    /// `self` is normalized if `real = 1.0` and `dual = 0.0`.
    #[inline]
    pub fn norm_squared(self) -> DualScalar {
        // https://math.stackexchange.com/questions/4609827/dual-quaternion-norm-expression-from-skinning-with-dual-quaternions-paper
        // https://users.cs.utah.edu/~ladislav/kavan07skinning/kavan07skinning.pdf
        let real_vec = Vec4::from(self.real);
        let dual_vec = Vec4::from(self.dual);
        let dot = real_vec.dot(dual_vec);

        DualScalar {
            real: self.real.length_squared(),
            dual: 2.0 * dot,
        }
    }

    /// Gives the norm of the dual quaternion.
    ///
    /// `self` is normalized if `real = 1.0` and `dual = 0.0`.
    #[inline]
    pub fn norm(self) -> DualScalar {
        // https://math.stackexchange.com/questions/4609827/dual-quaternion-norm-expression-from-skinning-with-dual-quaternions-paper
        // https://users.cs.utah.edu/~ladislav/kavan07skinning/kavan07skinning.pdf
        let real_mag = self.real.length();
        let real_vec = Vec4::from(self.real);
        let dual_vec = Vec4::from(self.dual);
        let dot = real_vec.dot(dual_vec);

        DualScalar {
            real: real_mag,
            dual: dot / real_mag,
        }
    }

    /// Whether `self` is normalized, i.e. a unit dual quaternion.
    #[inline]
    pub fn is_normalized(self) -> bool {
        let norm_sq = self.norm_squared();

        let eps = 1e-4; // same as glam chose...
        (norm_sq.real - 1.0).abs() < eps && norm_sq.dual.abs() < eps
    }

    /// Normalizes `self` to make it a unit dual quaternion.
    ///
    /// If you will immediately apply the normalized dual quat to transform a point/vector, consider
    /// using [`normalize_to_rotation_translation`][Self::normalize_to_rotation_translation]
    /// instead.
    #[inline]
    pub fn normalize_full(self) -> Self {
        // combine norm + inverse of norm into single calculation to optimize slightly.
        // see https://users.cs.utah.edu/~ladislav/kavan07skinning/kavan07skinning.pdf section 3.4
        //
        // norm = (norm(real), (real dot dual) / norm(real))
        //
        // inv = (1.0 / real, -dual / real^2)
        //
        // inverse norm = (1.0 / norm(real), -(real dot dual) / (norm(real) * normsq(real)))

        let real_normsq = self.real.length_squared();
        let real_norm = real_normsq.sqrt();
        let real_vec = Vec4::from(self.real);
        let dual_vec = Vec4::from(self.dual);
        let dot = real_vec.dot(dual_vec);

        let normalizer = DualScalar {
            real: 1.0 / real_norm,
            dual: -dot / (real_norm * real_normsq),
        };

        normalizer * self
    }

    /// Normalize `self` and then extract its rotation and translation components which can be
    /// applied to a point by doing `rotation.mul_vec3(point) + translation`
    ///
    /// This will be faster than `self.normalize().to_rotation_translation()` as we can use
    /// the full expansion of the operation to cancel out some calculations that would
    /// otherwise need to be performed. For justification of this, see
    /// <https://users.cs.utah.edu/~ladislav/kavan07skinning/kavan07skinning.pdf> particularly
    /// equation (4) and section 3.4.
    ///
    /// You can then apply this to a point by doing `(rotation * point + translation)`
    #[inline]
    pub fn normalize_to_rotation_translation(mut self) -> (Quat, Vec3) {
        let real_norm_inv = self.real.length_recip();

        self.real = self.real * real_norm_inv;
        self.dual = self.dual * real_norm_inv;
        self.to_rotation_translation()
    }

    /// Whether `self` is approximately equal to `other` with `max_abs_diff` error
    #[inline]
    pub fn abs_diff_eq(self, other: Self, max_abs_diff: f32) -> bool {
        self.real.abs_diff_eq(other.real, max_abs_diff)
            && self.dual.abs_diff_eq(other.dual, max_abs_diff)
    }

    /// An optimized form of `self * DualQuat::from_translation(trans)`
    /// (note the order!)
    #[inline]
    pub fn right_mul_translation(mut self, trans: Vec3) -> Self {
        self.dual.w -=
            0.5 * (self.real.x * trans.x + self.real.y * trans.y + self.real.z * trans.z);
        self.dual.x +=
            0.5 * (self.real.w * trans.x + self.real.y * trans.z - self.real.z * trans.y);
        self.dual.y +=
            0.5 * (self.real.w * trans.y + self.real.z * trans.x - self.real.x * trans.z);
        self.dual.z +=
            0.5 * (self.real.w * trans.z + self.real.x * trans.y - self.real.y * trans.x);
        self
    }
}

impl core::ops::Mul for DualQuat {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let Self { real: q0, dual: q1 } = self;
        let Self { real: q2, dual: q3 } = rhs;
        let real = q0 * q2;
        let dual = q0 * q3 + q1 * q2;
        Self { real, dual }
    }
}

impl core::ops::Add for DualQuat {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            real: self.real + rhs.real,
            dual: self.dual + rhs.dual,
        }
    }
}

impl core::ops::AddAssign for DualQuat {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.real = self.real + rhs.real;
        self.dual = self.dual + rhs.dual;
    }
}

impl core::ops::Sub for DualQuat {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            real: self.real - rhs.real,
            dual: self.dual - rhs.dual,
        }
    }
}

impl core::ops::SubAssign for DualQuat {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.real = self.real - rhs.real;
        self.dual = self.dual - rhs.dual;
    }
}

impl core::ops::Mul<f32> for DualQuat {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            real: self.real * rhs,
            dual: self.dual * rhs,
        }
    }
}

impl core::ops::Mul<DualQuat> for f32 {
    type Output = DualQuat;

    #[inline]
    fn mul(self, rhs: DualQuat) -> Self::Output {
        rhs * self
    }
}

impl core::ops::Mul<DualQuat> for DualScalar {
    type Output = DualQuat;

    #[inline]
    fn mul(self, rhs: DualQuat) -> Self::Output {
        DualQuat {
            real: rhs.real * self.real,
            dual: (rhs.dual * self.real) + (rhs.real * self.dual),
        }
    }
}

impl core::ops::Mul<DualScalar> for DualQuat {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: DualScalar) -> Self::Output {
        rhs * self
    }
}

#[cfg(feature = "std")]
impl core::fmt::Debug for DualQuat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (rot, translation) = self.to_rotation_translation();
        let (axis, angle) = rot.to_axis_angle();
        f.debug_struct("DualQuat")
            .field(
                "translation",
                &format!("[{} {} {}]", translation.x, translation.y, translation.z),
            )
            .field(
                "rotation",
                &format!(
                    "{:.1}° around [{} {} {}]",
                    angle.to_degrees(),
                    axis[0],
                    axis[1],
                    axis[2],
                ),
            )
            .field("real(raw)", &self.real)
            .field("dual(raw)", &self.dual)
            .finish()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn approx_eq_dualquat(a: DualQuat, b: DualQuat) -> bool {
        let max_abs_diff = 1e-6;
        a.abs_diff_eq(b, max_abs_diff)
    }

    macro_rules! assert_approx_eq_dualquat {
        ($a: expr, $b: expr) => {
            assert!(approx_eq_dualquat($a, $b), "{:#?} != {:#?}", $a, $b,);
        };
    }

    #[test]
    fn test_inverse() {
        let transform = DualQuat::from_rotation_translation(
            Quat::from_rotation_y(std::f32::consts::PI),
            Vec3::ONE,
        );
        let identity = transform * transform.inverse();
        assert_approx_eq_dualquat!(identity, DualQuat::IDENTITY);

        let transform = DualQuat::from_rotation_translation(
            Quat::from_axis_angle(Vec3::ONE.normalize(), 1.234),
            Vec3::new(1.0, 2.0, 3.0),
        );
        let identity = transform * transform.inverse();
        assert_approx_eq_dualquat!(identity, DualQuat::IDENTITY);
    }
}
