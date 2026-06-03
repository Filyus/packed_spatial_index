# packed_spatial_index WASM Demo

Browser demo for `SimdIndex2D` point search. The Rust wrapper builds with
`wasm-bindgen` and `wasm-pack`; the UI is a Vite + TypeScript canvas app.

The demo is repository-only and excluded from the published crates.io package.

## Run

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

The `wasm:build` script compiles the wrapper with:

```bash
RUSTFLAGS=-Ctarget-feature=+simd128 wasm-pack build crate --target web --out-dir ../web/pkg --release -- --no-default-features --features simd
```

This first demo targets modern browsers with WebAssembly SIMD support. It does
not build a separate scalar fallback bundle.
