// benches/benchmarks.rs — placeholder for future benchmarks
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_image_create(c: &mut Criterion) {
    c.bench_function("image_new_640x480_u8", |b| {
        b.iter(|| {
            let _img: rudolf_v::image::Image<u8> =
                rudolf_v::image::Image::new(640, 480);
        })
    });
}

criterion_group!(benches, bench_image_create);
criterion_main!(benches);
