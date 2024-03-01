use super::Mat3A;
use super::Vec3;

/// A 3-dimensional axis-aligned bounding box
#[derive(Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "with_serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with_speedy", derive(speedy::Writable, speedy::Readable))]
#[cfg_attr(feature = "with_speedy", allow(clippy::manual_slice_size_calculation))]
pub struct BoundingBox {
    /// Bounding box minimum (inclusive).
    pub min: Vec3,
    /// Bounding box maximum (inclusive).
    pub max: Vec3,
}

#[cfg(not(target_arch = "spirv"))]
impl core::fmt::Debug for BoundingBox {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?} - {:?}", self.min, self.max)
    }
}

#[allow(unused)]
impl BoundingBox {
    /// A [`BoundingBox`] that only contains [`Vec3::ZERO`].
    pub const ZERO: Self = Self {
        min: Vec3::ZERO,
        max: Vec3::ZERO,
    };

    /// A [`BoundingBox`] that contains no points.
    ///
    /// This is useful as the seed for bounding bounding boxes.
    #[inline]
    pub fn nothing() -> Self {
        Self {
            min: Vec3::splat(core::f32::INFINITY),
            max: Vec3::splat(core::f32::NEG_INFINITY),
        }
    }

    /// A [`BoundingBox`] that contains every point.
    #[inline]
    pub fn everything() -> Self {
        Self {
            min: Vec3::splat(core::f32::NEG_INFINITY),
            max: Vec3::splat(core::f32::INFINITY),
        }
    }

    /// Create a bounding box from a minimum and maximum position.
    #[inline]
    pub fn from_min_max(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn from_min_size(min: Vec3, size: Vec3) -> Self {
        Self {
            min,
            max: min + size,
        }
    }

    /// Create a bounding box from a center position and a size.
    pub fn from_center_size(center: Vec3, size: Vec3) -> Self {
        Self::from_min_max(center - 0.5 * size, center + 0.5 * size)
    }

    /// Create a bounding box from an iterator of points that the bounding box will cover.
    pub fn from_points(points: impl Iterator<Item = Vec3>) -> Self {
        let mut bb = Self::nothing();
        for p in points {
            bb.extend(p);
        }
        bb
    }

    /// Returns the center point of the bounding box.
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Returns the 3D axis size of the bounding box.
    #[inline]
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// Returns half the size (similar to a radius).
    #[inline]
    pub fn half_size(&self) -> Vec3 {
        0.5 * (self.max - self.min)
    }

    /// Only correct for positively sized boxes.
    pub fn volume(&self) -> f32 {
        let s = self.size();
        s.x * s.y * s.z
    }

    /// True if and only if there is at least one point for which `bb.contains(point)` is true.
    ///
    /// Will return `true` if [`Self::min`] == [`Self::max`].
    /// The opposite of `is_nothing()`.
    pub fn is_something(&self) -> bool {
        self.min.x <= self.max.x && self.min.y <= self.max.y && self.min.z <= self.max.z
    }

    /// True if and only if there is no point for which `bb.contains(point)` is true.
    ///
    /// The opposite of `is_something()`.
    pub fn is_nothing(&self) -> bool {
        self.max.x < self.min.x || self.max.y < self.min.y || self.max.z < self.min.z
    }

    /// True if this box contains exactly one point.
    ///
    /// `true` if [`Self::min`] == [`Self::max`].
    #[inline]
    pub fn is_point(&self) -> bool {
        self.min == self.max
    }

    /// Returns `true` if, and only if, all elements are finite.
    ///
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.min.is_finite() && self.max.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.min.is_nan() || self.max.is_nan()
    }

    #[cfg(not(target_arch = "spirv"))]
    /// The eight corners of this bounding box.
    pub fn corners(&self) -> [Vec3; 8] {
        [
            self.min,
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            self.max,
        ]
    }

    /// The minimum radius of a sphere, centered at the origin, fully containing the box.
    ///
    /// Requires a well-formed box for the result to be valid.
    #[cfg(not(target_arch = "spirv"))]
    pub fn bounding_sphere_radius(&self) -> f32 {
        let mut max_dist_square = 0.0f32;
        for corner in self.corners() {
            max_dist_square = max_dist_square.max(corner.length_squared());
        }
        max_dist_square.sqrt()
    }

    /// The minimum radius of a sphere, centered at the bounding box, fully containing the box.
    ///
    /// Requires a well-formed box for the result to be valid.
    #[cfg(not(target_arch = "spirv"))]
    pub fn centered_bounding_sphere_radius(&self) -> f32 {
        let mut max_dist_square = 0.0f32;
        let center = self.center();
        for corner in self.corners() {
            max_dist_square = max_dist_square.max((corner - center).length_squared());
        }
        max_dist_square.sqrt()
    }

    /// The twelve edges of this bounding box.
    pub fn edges(&self) -> [[Vec3; 2]; 12] {
        use glam::vec3;
        let a = self.min;
        let b = self.max;
        [
            // along X:
            [vec3(a.x, a.y, a.z), vec3(b.x, a.y, a.z)],
            [vec3(a.x, b.y, a.z), vec3(b.x, b.y, a.z)],
            [vec3(a.x, a.y, b.z), vec3(b.x, a.y, b.z)],
            [vec3(a.x, b.y, b.z), vec3(b.x, b.y, b.z)],
            // along Y:
            [vec3(a.x, a.y, a.z), vec3(a.x, b.y, a.z)],
            [vec3(b.x, a.y, a.z), vec3(b.x, b.y, a.z)],
            [vec3(a.x, a.y, b.z), vec3(a.x, b.y, b.z)],
            [vec3(b.x, a.y, b.z), vec3(b.x, b.y, b.z)],
            // along Z:
            [vec3(a.x, a.y, a.z), vec3(a.x, a.y, b.z)],
            [vec3(b.x, a.y, a.z), vec3(b.x, a.y, b.z)],
            [vec3(a.x, b.y, a.z), vec3(a.x, b.y, b.z)],
            [vec3(b.x, b.y, a.z), vec3(b.x, b.y, b.z)],
        ]
    }

    /// Enlarge the box to include this point.
    #[inline]
    pub fn extend(&mut self, pos: Vec3) {
        self.min = self.min.min(pos);
        self.max = self.max.max(pos);
    }

    #[must_use]
    pub fn union(mut self, other: Self) -> Self {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self
    }

    /// Returns the smallest volume that is covered by both `self` and `other`,
    /// or [`Self::nothing`] if the boxes are disjoint.
    #[must_use]
    pub fn intersection(mut self, other: Self) -> Self {
        let intersection = Self {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        };
        if intersection.is_nothing() {
            Self::nothing()
        } else {
            intersection
        }
    }

    /// Returns `true` if the point is within (or on the edge of) the box.
    #[must_use]
    pub fn contains(&self, point: Vec3) -> bool {
        (self.min.x <= point.x && point.x <= self.max.x)
            && (self.min.y <= point.y && point.y <= self.max.y)
            && (self.min.z <= point.z && point.z <= self.max.z)
    }

    /// Expand with this much padding on each side.
    #[must_use]
    pub fn expanded(&self, padding: Vec3) -> Self {
        Self {
            min: self.min - padding,
            max: self.max + padding,
        }
    }

    /// Translate (move) the box by this much.
    #[must_use]
    pub fn translated(&self, translation: Vec3) -> Self {
        Self {
            min: self.min + translation,
            max: self.max + translation,
        }
    }

    /// Return a bounding box that contains this box after it has been rotated around [`Vec3::ZERO`].
    ///
    /// Note that the rotated bounding box is very likely larger than the original,
    /// since it must be large enough to contain the now rotated box.
    #[must_use]
    #[cfg(not(target_arch = "spirv"))]
    pub fn rotated_around_origin(&self, q: &crate::Quat) -> Self {
        if self.is_nothing() {
            Self::nothing()
        } else {
            rotate_bounding_box(self.half_size(), self.center(), *q)
        }
    }

    /// Return a bounding box that contains this box after it has been transformed.
    ///
    /// Note that the rotated bounding box is very likely larger than the original,
    /// since it must be large enough to contain the now rotated box.
    #[must_use]
    #[cfg(not(target_arch = "spirv"))]
    pub fn transform_iso(&self, m: &crate::IsoTransform) -> Self {
        if self.is_nothing() {
            Self::nothing()
        } else {
            transform_bounding_box(self.half_size(), self.center(), m)
        }
    }

    /// Return a bounding box that contains this box after it has been transformed.
    ///
    /// Note that the rotated bounding box is very likely larger than the original,
    /// since it must be large enough to contain the now rotated box.
    #[must_use]
    #[cfg(not(target_arch = "spirv"))]
    pub fn transform_affine3(&self, m: &crate::Affine3A) -> Self {
        if self.is_nothing() {
            Self::nothing()
        } else {
            transform_bounding_box(self.half_size(), self.center(), m)
        }
    }

    /// Return a bounding box that contains this box after it has been transformed.
    ///
    /// Note that the rotated bounding box is very likely larger than the original,
    /// since it must be large enough to contain the now rotated box.
    #[must_use]
    #[cfg(not(target_arch = "spirv"))]
    pub fn transform_conformal3(&self, m: &crate::Conformal3) -> Self {
        if self.is_nothing() {
            Self::nothing()
        } else {
            transform_bounding_box(self.half_size(), self.center(), m)
        }
    }
}

trait TransformPoint3 {
    fn transform_point3(&self, p: Vec3) -> Vec3;
}

impl TransformPoint3 for crate::IsoTransform {
    #[inline(always)]
    fn transform_point3(&self, p: Vec3) -> Vec3 {
        self.transform_point3(p)
    }
}

impl TransformPoint3 for crate::Affine3A {
    #[inline(always)]
    fn transform_point3(&self, p: Vec3) -> Vec3 {
        self.transform_point3(p)
    }
}

impl TransformPoint3 for crate::Conformal3 {
    #[inline(always)]
    fn transform_point3(&self, p: Vec3) -> Vec3 {
        self.transform_point3(p)
    }
}

trait ToScaledMat3A {
    fn to_scaled_mat3a(&self) -> Mat3A;
}

impl ToScaledMat3A for crate::IsoTransform {
    #[inline(always)]
    fn to_scaled_mat3a(&self) -> Mat3A {
        Mat3A::from_quat(self.rotation())
    }
}

impl ToScaledMat3A for crate::Affine3A {
    #[inline(always)]
    fn to_scaled_mat3a(&self) -> Mat3A {
        self.matrix3
    }
}

impl ToScaledMat3A for crate::Conformal3 {
    #[inline(always)]
    fn to_scaled_mat3a(&self) -> Mat3A {
        Mat3A::from_quat(self.rotation()).mul_scalar(self.scale())
    }
}

#[cfg(not(target_arch = "spirv"))]
fn transform_bounding_box<T: TransformPoint3 + ToScaledMat3A>(
    half_size: Vec3,
    center: Vec3,
    m: &T,
) -> BoundingBox {
    // Inspired by:
    // https://zeux.io/2010/10/17/aabb-from-obb-with-component-wise-abs

    let center_transformed = m.transform_point3(center);

    let half_size_transformed = {
        let matrix3 = m.to_scaled_mat3a();
        let abs_matrix3 = Mat3A::from_cols(
            matrix3.x_axis.abs(),
            matrix3.y_axis.abs(),
            matrix3.z_axis.abs(),
        );
        abs_matrix3.mul_vec3(half_size)
    };

    BoundingBox {
        min: center_transformed - half_size_transformed,
        max: center_transformed + half_size_transformed,
    }
}

#[cfg(not(target_arch = "spirv"))]
fn rotate_bounding_box(half_size: Vec3, center: Vec3, q: crate::Quat) -> BoundingBox {
    // Inspired by:
    // https://zeux.io/2010/10/17/aabb-from-obb-with-component-wise-abs

    let half_size_rotated = {
        let matrix3 = Mat3A::from_quat(q);
        let abs_matrix3 = Mat3A::from_cols(
            matrix3.x_axis.abs(),
            matrix3.y_axis.abs(),
            matrix3.z_axis.abs(),
        );
        abs_matrix3.mul_vec3(half_size)
    };

    BoundingBox {
        min: center - half_size_rotated,
        max: center + half_size_rotated,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Affine3A;
    use crate::Conformal3;
    use crate::IsoTransform;
    use crate::Quat;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn test_bounding_box() {
        let bb = BoundingBox::from_min_max(Vec3::ZERO, Vec3::ZERO);
        assert!(bb.contains(Vec3::ZERO));
        assert!(bb.is_something());
        assert!(!bb.is_nothing());
        let bb_rotated =
            bb.transform_affine3(&Affine3A::from_quat(Quat::from_axis_angle(Vec3::X, 0.5)));
        assert_eq!(bb, bb_rotated);
        let bb_translated =
            bb.transform_affine3(&Affine3A::from_translation(Vec3::new(2.0, 3.0, 5.0)));
        assert!(bb_translated.is_something());
        assert!(bb_translated.contains(Vec3::new(2.0, 3.0, 5.0)));
    }

    #[test]
    fn test_intersection() {
        assert_eq!(
            BoundingBox::from_min_max(Vec3::splat(0.0), Vec3::splat(2.0)).intersection(
                BoundingBox::from_min_max(Vec3::splat(1.0), Vec3::splat(3.0))
            ),
            BoundingBox::from_min_max(Vec3::splat(1.0), Vec3::splat(2.0))
        );
        assert_eq!(
            BoundingBox::from_min_max(Vec3::splat(0.0), Vec3::splat(1.0)).intersection(
                BoundingBox::from_min_max(Vec3::splat(2.0), Vec3::splat(3.0))
            ),
            BoundingBox::nothing()
        );
    }

    #[test]
    fn test_rotated_around_origin() {
        const EPSILON: f32 = 1e-6;
        const SIZE: f32 = 1.0;
        const ANGLE: f32 = FRAC_PI_2 * 0.5; // 45 deg
        let rotation = Quat::from_rotation_z(ANGLE);
        let bb = BoundingBox::from_center_size(Vec3::ZERO, Vec3::splat(SIZE));
        let bb_transformed = bb.rotated_around_origin(&rotation);

        let expected_size_xy: f32 = (2.0 * SIZE * SIZE).sqrt();
        assert!(Vec3::distance(bb_transformed.center(), Vec3::ZERO) < EPSILON);
        assert!(
            Vec3::distance(
                bb_transformed.size(),
                Vec3::new(expected_size_xy, expected_size_xy, SIZE)
            ) < EPSILON
        );
    }

    #[test]
    fn test_transform_iso() {
        const EPSILON: f32 = 1e-6;
        const SIZE: f32 = 1.0;
        const ANGLE: f32 = FRAC_PI_2 * 0.5; // 45 deg
        let rotation = Quat::from_rotation_z(ANGLE);
        let translation = Vec3::splat(1.0);
        let transform = IsoTransform::from_rotation_translation(rotation, translation);
        let bb = BoundingBox::from_center_size(Vec3::ZERO, Vec3::splat(SIZE));
        let bb_transformed = bb.transform_iso(&transform);

        let expected_size_xy: f32 = (2.0 * SIZE * SIZE).sqrt();
        assert!(Vec3::distance(bb_transformed.center(), translation) < EPSILON);
        assert!(
            Vec3::distance(
                bb_transformed.size(),
                Vec3::new(expected_size_xy, expected_size_xy, SIZE)
            ) < EPSILON
        );
    }

    #[test]
    fn test_transform_affine3() {
        const EPSILON: f32 = 1e-6;
        const SIZE: f32 = 1.0;
        const SCALE: f32 = 2.0;
        const ANGLE: f32 = FRAC_PI_2 * 0.5; // 45 deg
        let rotation = Quat::from_rotation_z(ANGLE);
        let translation = Vec3::splat(1.0);
        let transform =
            Affine3A::from_scale_rotation_translation(Vec3::splat(SCALE), rotation, translation);
        let bb = BoundingBox::from_center_size(Vec3::ZERO, Vec3::splat(SIZE));
        let bb_transformed = bb.transform_affine3(&transform);

        let expected_size_xy: f32 = (2.0 * SIZE * SIZE * SCALE * SCALE).sqrt();
        assert!(Vec3::distance(bb_transformed.center(), translation) < EPSILON);
        assert!(
            Vec3::distance(
                bb_transformed.size(),
                Vec3::new(expected_size_xy, expected_size_xy, SIZE * SCALE)
            ) < EPSILON
        );
    }

    #[test]
    fn test_transform_conformal3() {
        const EPSILON: f32 = 1e-6;
        const SIZE: f32 = 1.0;
        const SCALE: f32 = 2.0;
        const ANGLE: f32 = FRAC_PI_2 * 0.5; // 45 deg
        let rotation = Quat::from_rotation_z(ANGLE);
        let translation = Vec3::splat(1.0);
        let transform = Conformal3::from_scale_rotation_translation(SCALE, rotation, translation);
        let bb = BoundingBox::from_center_size(Vec3::ZERO, Vec3::splat(SIZE));
        let bb_transformed = bb.transform_conformal3(&transform);

        let expected_size_xy: f32 = (2.0 * SIZE * SIZE * SCALE * SCALE).sqrt();
        assert!(Vec3::distance(bb_transformed.center(), translation) < EPSILON);
        assert!(
            Vec3::distance(
                bb_transformed.size(),
                Vec3::new(expected_size_xy, expected_size_xy, SIZE * SCALE)
            ) < EPSILON
        );
    }
}
