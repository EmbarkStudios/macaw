# `macaw` changelog

<!-- markdownlint-disable MD024 -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [0.19.1] - 2023-02-26

- Upgrade `glam` from 0.22 to 0.25, see the [glam changelog](https://github.com/bitshifter/glam-rs/blob/main/CHANGELOG.md?plain=1#L8-L119).

## [0.19.0] - 2023-02-21

- Require Rust 1.76.0
- `Conformal3::to_affine3a` and `Conformal3::to_scale_rotation_translation` now for consistency takes `self` by value instead of by reference

## [0.18.6] - 2023-08-18

- Enable publishing `macaw` to crates.io by not relying on wildcard deps

## [0.18.5] - 2023-08-18

### Changed 🔧

- Require [Rust 1.71.1+](https://blog.rust-lang.org/2023/08/03/Rust-1.71.1.htlm)
- Fix `speedy` clippy lint
- Update `tiny-bench` to `v0.3`.
- Exclude `transform_bounding_box()`and `rotate_bounding_box()` functions for `spirv` architecture.

## [0.18.4] - 2023-05-31

### Changed 🔧

- Optimize transform and rotations of bounding boxes

## [0.18.3] - 2023-05-17

- Fixed bug in `bounding_box::intersection()`

## [0.18.2] - 2023-04-28

### Added 🔧

- Export glam's `f64` types.
- Export `glam::avec3`, `glam::Mat3A` and `glam::mat3a`.
- Add `Conformal3::transform_point3a` and `Conformal3::transform_vector3a`.

## [0.18.1] - 2023-02-13

### Added 🔧

- Added `assert` and `debug_assert` features which enables the glam features `glam-assert` and `debug-glam-assert` respectively. In the future these features might also enable additional checks in `macaw` itself.

## [0.18.0] - 2022-12-16

- Upgraded `glam` from 0.20 to 0.22, see [changelog](https://github.com/bitshifter/glam-rs/blob/main/CHANGELOG.md#0220---2022-10-24).
- Upgraded `speedy` from 0.8.2 to 0.8.4 to use new `glam`
- Removed deprecated `const_*` macro. These have been replaced by const fn methods.

## [0.17.2] - 2022-09-13

### Added 🔧

- Added `Conformal3` transform combining translation, rotation and uniform scale.
- Added `Conformal3::inv_scale` to retrieve inverse scale.
- Added `Conformal3::normalize` to create a Conformal3 with a normalized rotation

## [0.17.1] - 2022-09-08

### Changed 🔧

- Upgraded `speedy` 0.8.0 -> 0.8.2 which includes our required [`glam` support PR](https://github.com/koute/speedy/pull/13).
- Added `centered_bounding_sphere_radius` to get the bounding sphere radius relative to the center of the mesh.
- Added `mean` and `has_equal_components` extension methods to Vec2, Vec3 and Vec4.

### Fixed 🐛

- Normalize ray direction when multiplying Ray3 with Affine3A and Mat4

## [0.17.0] - 2022-04-04

### Added 🔧

- Added the methods `from_min_size` and 'bounding_sphere_radius' to 'BoundingBox'

### Changed 🔧

- Replace requirement of forked `bytemuck` crate with latest released v1.9 crate and macros

### Fixed 🐛

- Clippy lints

## [0.16.0] - 2021-11-05

- Add `#[must_use]` to types returning `Self` to catch issues where return value is not used, such as with the user expecting mutation instead of return by value
- Update `glam` to `0.20`

## [0.15.0] - 2021-11-05

- Reverse the argument order of the `step` extension functions to match HLSL/GLSL.

## [0.14.1] - 2021-10-18

- Add `impl From<IsoTransform> for Affine3A`

## [0.14.0] - 2021-10-18

- Remove deprecated `QuatExt::from_look_rotation`, use `QuatExt::rotate_positive_z_towards` instead
- Add `BoundingBox::ZERO`
- Add `BoundingBox::edges`

## [0.13.0] - 2021-10-12

- Add `reflect` to `Vec3Ext`.
- Remove `Mat3Ext` and `Mat4Ext`. Use `Mat3::from` and `Mat4::from` instead for conversion between `Mat3` and `Mat4`.

## [0.12.0] - 2021-10-04

- Update `glam` to `0.18`
- Add `Affine3A * Ray3` and `Mat3 * Ray3`.
- Add `trunc` to `Vec2Ext`, `Vec3Ext` and `Vec4Ext`.
- Add `step` and `fract` to `Vec4Ext`.
- Add `with_speedy` feature which allows serialization via the [`speedy`](https://crates.io/crates/speedy) crate.

## [0.11.2] - 2021-09-14

- Remove dependency on `spirv-std`

## [0.11.1] - 2021-08-18

- Add `with_bytemuck` feature that implements `bytemuck::Pod` and `bytemuck::Zeroable` to `ColorRgba8` for easy conversion to/from raw bytes

## [0.11.0] - 2021-08-03

- Update `glam` to `0.17`
- Update `spirv-std` to `0.4.0-alpha.11`

## [0.10.6] - 2021-08-02

- First public published version

## [0.10.5] - 2021-07-08

- Export `Affine3A`

## [0.10.4] - 2021-06-23

- Updated glam to 0.16

## [0.10.3] - 2021-05-18

- Updated glam to 0.15.1
- Expose some integer vector types (e.g. UVec2)

## [0.10.2] - 2021-04-15

### Added ⭐

- Add more methods to `Ray3` and `Plane3`.

## [0.10.1] - 2021-04-14

### Changed 🔧

- Removed `debug_assert` from `IsoTransform` constructors.

## [0.10.0] - 2021-04-13

### Changed 🔧

- Breaking change: `IsoTransform` now has private members.
- Updated `glam` to `0.14`.

### Added ⭐

- `IsoTransform::look_at_rh`.

## [0.9.0] - 2021-03-31

### Changed 🔧

- Renamed `IsoTransform::identity()` to `IsoTransform::IDENTITY`
- Update `glam` to `0.13`

## ???

### Changed 🔧

- Renamed `AffineTransform` to `IsoTransform`.

## [0.4.0] - 2020-11-24

### Added ⭐

- Add `is_finite` member to `AffineTransform`, `Plane3` and `Ray3`.
- Derive `PartialEq` for all types
- Add `BoundingBox` functions `contains`, `is_nothing` and `is_something`

### Fixed 🐛

- Fixed bug where `BoundingBox` transform functions would always return an empty bbox when transforming point-like bboxes (where `bb.is_point() == true`).

### Removed 🔥

- Removed `BoundingBox::is_empty()`. Consider using `is_nothing()` instead.

## [0.3.1] - 2020-11-18

### Changed 🔧

- Upgraded to [`glam`](https://github.com/bitshifter/glam-rs) 0.10.2 which and switched to use the new public field accessors: `.x` instead of `.x()`. The old accessor functions still work but have been deprecated. [Full changelog](https://github.com/bitshifter/glam-rs/blob/master/CHANGELOG.md#0100---2020-10-31)

## [0.3.0] - 2020-11-08

### Added ⭐

- Added `Ray3::offset_along_ray` to easily create a new ray given an offset along an existing ray

### Changed 🔧

- Upgraded to [`glam`](https://github.com/bitshifter/glam-rs) 0.10.0 which contains breaking change: "Changed the return type of `Vec4::truncate` from `Vec3A` to `Vec3`". [Full changelog](https://github.com/bitshifter/glam-rs/blob/master/CHANGELOG.md#0100---2020-10-31)

## [0.2.0] - 2020-10-10

### Added ⭐

- First published private release

[0.2.0]: https://github.com/EmbarkStudios/wim-app/compare/35b042afe7f6696ee730ef0040b50ee427b474db...macaw-0.2.0
