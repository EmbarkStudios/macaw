#[path = "support/macros.rs"]
#[macro_use]
mod macros;
mod support;

use support::*;

fn main() {
    rotated_around_origin();
    transform_iso();
    transform_affine3();
    transform_conformal3();
}

bench_binop_ref!(
    rotated_around_origin,
    "bbox rotated around origin",
    op => rotated_around_origin,
    from1 => random_bounding_box,
    from2 => random_quat
);

bench_binop_ref!(
    transform_iso,
    "bbox transformed with iso",
    op => transform_iso,
    from1 => random_bounding_box,
    from2 => random_iso_transform
);

bench_binop_ref!(
    transform_affine3,
    "bbox transformed with affine3",
    op => transform_affine3,
    from1 => random_bounding_box,
    from2 => random_affine3
);

bench_binop_ref!(
    transform_conformal3,
    "bbox transformed with conformal3",
    op => transform_conformal3,
    from1 => random_bounding_box,
    from2 => random_conformal3
);
