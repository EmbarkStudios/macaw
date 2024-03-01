#[path = "support/macros.rs"]
#[macro_use]
mod macros;
mod support;

use std::ops::Mul;
use support::*;

fn main() {
    cnf3_inverse();
    cnf3_mul_iso();
    cnf3_transform_point3();
    cnf3_transform_vector3();
}

bench_unop!(cnf3_inverse, "conformal3 inverse", op => inverse, from => random_conformal3);

bench_binop!(cnf3_mul_iso, "conformal3 mul conformal3", op => mul, from => random_conformal3);

bench_binop!(
    cnf3_transform_point3,
    "cnf3 transform point3",
    op => transform_point3,
    from1 => random_conformal3,
    from2 => random_vec3
);

bench_binop!(
    cnf3_transform_vector3,
    "cnf3 transform vector3",
    op => transform_vector3,
    from1 => random_conformal3,
    from2 => random_vec3
);
