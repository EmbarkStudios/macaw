#[path = "support/macros.rs"]
#[macro_use]
mod macros;
mod support;

use std::ops::Mul;
use support::*;

fn main() {
    iso_inverse();
    iso_mul_iso();
    iso_transform_point3();
    iso_transform_vector3();
}

bench_unop!(iso_inverse, "iso inverse", op => inverse, from => random_iso_transform);

bench_binop!(iso_mul_iso, "iso mul iso", op => mul, from => random_iso_transform);

bench_binop!(
    iso_transform_point3,
    "iso transform point3",
    op => transform_point3,
    from1 => random_iso_transform,
    from2 => random_vec3
);

bench_binop!(
    iso_transform_vector3,
    "iso transform vector3",
    op => transform_vector3,
    from1 => random_iso_transform,
    from2 => random_vec3
);
