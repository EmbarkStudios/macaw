//! Copied from <https://github.com/bitshifter/glam-rs>

#![allow(dead_code)]
use core::f32;
use macaw::Affine3A;
use macaw::BoundingBox;
use macaw::Conformal3;
use macaw::IsoTransform;
use macaw::Quat;
use macaw::Vec2;
use macaw::Vec3;
use macaw::Vec3A;
use macaw::Vec4;

#[allow(clippy::upper_case_acronyms)]
pub struct PCG32 {
    state: u64,
    inc: u64,
}

impl PCG32 {
    pub fn seed(initstate: u64, initseq: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (initseq << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(initstate);
        rng.next_u32();
        rng
    }

    pub fn next_u32(&mut self) -> u32 {
        let oldstate = self.state;
        self.state = oldstate
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc | 1);
        let xorshifted = ((oldstate >> 18) ^ oldstate) >> 27;
        let rot = oldstate >> 59;
        ((xorshifted >> rot) | (xorshifted << (rot.wrapping_neg() & 31))) as u32
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() & 0xffffff) as f32 / 16777216.0
    }
}

impl Default for PCG32 {
    fn default() -> Self {
        Self::seed(0x853c49e6748fea9b, 0xda3e39cb94b95bdb)
    }
}

pub fn random_vec2(rng: &mut PCG32) -> Vec2 {
    Vec2::new(rng.next_f32(), rng.next_f32())
}

pub fn random_vec3(rng: &mut PCG32) -> Vec3 {
    Vec3::new(rng.next_f32(), rng.next_f32(), rng.next_f32())
}

pub fn random_vec3a(rng: &mut PCG32) -> Vec3A {
    Vec3A::new(rng.next_f32(), rng.next_f32(), rng.next_f32())
}

pub fn random_vec4(rng: &mut PCG32) -> Vec4 {
    Vec4::new(
        rng.next_f32(),
        rng.next_f32(),
        rng.next_f32(),
        rng.next_f32(),
    )
}

pub fn random_nonzero_vec2(rng: &mut PCG32) -> Vec2 {
    loop {
        let v = random_vec2(rng);
        if v.length_squared() > 0.01 {
            return v;
        }
    }
}

pub fn random_nonzero_vec3(rng: &mut PCG32) -> Vec3 {
    loop {
        let v = random_vec3(rng);
        if v.length_squared() > 0.01 {
            return v;
        }
    }
}

pub fn random_f32(rng: &mut PCG32) -> f32 {
    rng.next_f32()
}

pub fn random_radians(rng: &mut PCG32) -> f32 {
    -f32::consts::PI + rng.next_f32() * 2.0 * f32::consts::PI
}

pub fn random_quat(rng: &mut PCG32) -> Quat {
    let yaw = random_radians(rng);
    let pitch = random_radians(rng);
    let roll = random_radians(rng);
    Quat::from_euler(glam::EulerRot::YXZ, yaw, pitch, roll)
}

pub fn random_iso_transform(rng: &mut PCG32) -> IsoTransform {
    IsoTransform::from_rotation_translation(random_quat(rng), random_vec3(rng))
}

pub fn random_affine3(rng: &mut PCG32) -> Affine3A {
    Affine3A::from_scale_rotation_translation(random_vec3(rng), random_quat(rng), random_vec3(rng))
}

pub fn random_conformal3(rng: &mut PCG32) -> Conformal3 {
    Conformal3::from_scale_rotation_translation(random_f32(rng), random_quat(rng), random_vec3(rng))
}

pub fn random_bounding_box(rng: &mut PCG32) -> BoundingBox {
    BoundingBox::from_center_size(random_vec3(rng), random_vec3(rng))
}
