use glam::Mat3;
use glam::Vec3;

pub trait Mat3Ext {
    /// Multiply `self` by a scaling vector `scale` faster than creating a whole diagonal scaling
    /// matrix and then multiplying that. This operation is commutative.
    fn mul_diagonal_scale(self, scale: Vec3) -> Self;
}

impl Mat3Ext for Mat3 {
    #[inline]
    fn mul_diagonal_scale(mut self, scale: Vec3) -> Self {
        self.x_axis *= scale.x;
        self.y_axis *= scale.y;
        self.z_axis *= scale.z;
        self
    }
}
