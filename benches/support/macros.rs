//! Copied from <https://github.com/bitshifter/glam-rs>

#[macro_export]
macro_rules! bench_unop {
    ($name: ident, $desc: expr, op => $unop: ident, from => $from: expr) => {
        #[allow(clippy::undocumented_unsafe_blocks)] // just benchmarking code
        pub(crate) fn $name() {
            const SIZE: usize = 1 << 13;
            let mut rng = support::PCG32::default();
            let inputs =
                tiny_bench::black_box((0..SIZE).map(|_| $from(&mut rng)).collect::<Vec<_>>());
            // pre-fill output vector with some random value
            let mut outputs = vec![$from(&mut rng).$unop(); SIZE];
            let mut i = 0;
            tiny_bench::bench_labeled($desc, || {
                i = (i + 1) & (SIZE - 1);
                unsafe {
                    *outputs.get_unchecked_mut(i) = inputs.get_unchecked(i).$unop();
                }
            });
            tiny_bench::black_box(outputs);
        }
    };
}

#[macro_export]
macro_rules! bench_binop {
    ($name: ident, $desc: expr, op => $binop: ident, from1 => $from1:expr, from2 => $from2:expr) => {
        #[allow(clippy::undocumented_unsafe_blocks)] // just benchmarking code
        pub(crate) fn $name() {
            const SIZE: usize = 1 << 13;
            let mut rng = support::PCG32::default();
            let inputs1 =
                tiny_bench::black_box((0..SIZE).map(|_| $from1(&mut rng)).collect::<Vec<_>>());
            let inputs2 =
                tiny_bench::black_box((0..SIZE).map(|_| $from2(&mut rng)).collect::<Vec<_>>());
            // pre-fill output vector with some random value
            let mut outputs = vec![$from1(&mut rng).$binop($from2(&mut rng)); SIZE];
            let mut i = 0;
            tiny_bench::bench_labeled($desc, || {
                    i = (i + 1) & (SIZE - 1);
                    unsafe {
                        *outputs.get_unchecked_mut(i) = inputs1.get_unchecked(i).$binop(*inputs2.get_unchecked(i));
                    }
            });
            tiny_bench::black_box(outputs);
        }
    };
    ($name: ident, $desc: expr, op => $binop: ident, from => $from: expr) => {
        bench_binop!($name, $desc, op => $binop, from1 => $from, from2 => $from);
    };
}

// Plain copy paste of `bench_binop` above, but passes it's argument to the operation by ref
// instead of by value
#[macro_export]
macro_rules! bench_binop_ref {
    ($name: ident, $desc: expr, op => $binop: ident, from1 => $from1:expr, from2 => $from2:expr) => {
        #[allow(clippy::undocumented_unsafe_blocks)] // just benchmarking code
        pub(crate) fn $name() {
            const SIZE: usize = 1 << 13;
            let mut rng = support::PCG32::default();
            let inputs1 =
                tiny_bench::black_box((0..SIZE).map(|_| $from1(&mut rng)).collect::<Vec<_>>());
            let inputs2 =
                tiny_bench::black_box((0..SIZE).map(|_| $from2(&mut rng)).collect::<Vec<_>>());
            // pre-fill output vector with some random value
            let mut outputs = vec![$from1(&mut rng).$binop(&$from2(&mut rng)); SIZE];
            let mut i = 0;
            tiny_bench::bench_labeled($desc, || {
                    i = (i + 1) & (SIZE - 1);
                    unsafe {
                        *outputs.get_unchecked_mut(i) = inputs1.get_unchecked(i).$binop(&*inputs2.get_unchecked(i));
                    }
            });
            tiny_bench::black_box(outputs);
        }
    };
    ($name: ident, $desc: expr, op => $binop: ident, from => $from: expr) => {
        bench_binop!($name, $desc, op => $binop, from1 => $from, from2 => $from);
    };
}
