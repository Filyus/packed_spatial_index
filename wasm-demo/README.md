# packed_spatial_index WASM Demo

Browser demo for `SimdIndex2D` and `SimdIndex3D` search over boxes and points.
The Rust wrapper builds with `wasm-bindgen` and `wasm-pack`; the UI is a Vite +
TypeScript + WebGL2 canvas app.

The demo is repository-only and excluded from the published crates.io package.

For the object-storage serving story, see:

- [`worker`](worker): low-level core `PSINDEX` range-read Worker demo.
- [`geo-worker`](geo-worker): GeoParquet -> `gp2psindex` -> R2 -> HTTP
  feature/search API demo.

The wrapper exposes the core 2D API used by the demo:

- 2D and 3D range search with `search`, `any`, and `first`;
- nearest-neighbor search with optional max distance;
- `extent`, `len`, `node_size`, and `is_empty`;
- binary persistence round-trip with `to_bytes` and `from_bytes`.

The 3D mode renders an XY projection of the data. Range queries use the XY
rectangle plus the current depth slice, and nearest-neighbor queries use the
same depth value as the query point's Z coordinate.

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
