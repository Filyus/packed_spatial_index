import { WasmIndex2D, WasmIndex3D } from '../pkg/packed_spatial_index_wasm_demo';

type SearchProfile = {
  hits: Uint32Array<ArrayBufferLike>;
};

type IndexExtent = {
  empty: boolean;
  minX?: number;
  minY?: number;
  minZ?: number;
  maxX?: number;
  maxY?: number;
  maxZ?: number;
};

export function validateWasmWrapper(): void {
  validateWasm2DWrapper();
  validateWasm3DWrapper();
}

function validateWasm2DWrapper(): void {
  const sample = new Float64Array([0, 0, 5, 5, 10, 10, 6, 4]);
  const sampleIndex = WasmIndex2D.from_points(sample, 16);
  const actual = Array.from(sampleIndex.search(4, 4, 7, 7)).sort((a, b) => a - b);
  const expected = bruteForceSearch(sample, 4, 4, 7, 7);
  if (actual.length !== expected.length || actual.some((value, i) => value !== expected[i])) {
    throw new Error(`WASM validation failed: got [${actual}], expected [${expected}]`);
  }
  const profile = sampleIndex.search_profile(4, 4, 7, 7) as SearchProfile;
  const profiled = Array.from(profile.hits).sort((a, b) => a - b);
  if (profiled.length !== expected.length || profiled.some((value, i) => value !== expected[i])) {
    throw new Error(`WASM profile validation failed: got [${profiled}], expected [${expected}]`);
  }
  const nearest = Array.from(sampleIndex.neighbors(5.5, 4.5, 2)).sort((a, b) => a - b);
  if (nearest.length !== 2 || nearest[0] !== 1 || nearest[1] !== 3) {
    throw new Error(`WASM neighbors validation failed: got [${nearest}], expected [1,3]`);
  }
  const nearestWithin = Array.from(sampleIndex.neighbors_within(5.5, 4.5, 4, 1.0)).sort((a, b) => a - b);
  if (nearestWithin.length !== 2 || nearestWithin[0] !== 1 || nearestWithin[1] !== 3) {
    throw new Error(`WASM neighbors_within validation failed: got [${nearestWithin}], expected [1,3]`);
  }
  const extent = sampleIndex.extent() as IndexExtent;
  if (extent.empty || extent.minX !== 0 || extent.minY !== 0 || extent.maxX !== 10 || extent.maxY !== 10) {
    throw new Error(`WASM extent validation failed: got ${JSON.stringify(extent)}`);
  }
  if (!sampleIndex.any(4, 4, 7, 7) || sampleIndex.any(20, 20, 30, 30)) {
    throw new Error('WASM any validation failed');
  }
  const first = sampleIndex.first(4, 4, 7, 7);
  if (first !== 1 && first !== 3) {
    throw new Error(`WASM first validation failed: got ${first}, expected 1 or 3`);
  }
  const roundtrip = WasmIndex2D.from_bytes(sampleIndex.to_bytes());
  const roundtripHits = Array.from(roundtrip.search(4, 4, 7, 7)).sort((a, b) => a - b);
  if (roundtripHits.length !== expected.length || roundtripHits.some((value, i) => value !== expected[i])) {
    throw new Error(`WASM persistence validation failed: got [${roundtripHits}], expected [${expected}]`);
  }

  let rejectedOddLength = false;
  try {
    WasmIndex2D.from_points(new Float64Array([1, 2, 3]), 16);
  } catch {
    rejectedOddLength = true;
  }
  if (!rejectedOddLength) {
    throw new Error('odd-length input was accepted');
  }

  const boxes = new Float64Array([0, 0, 2, 2, 5, 5, 6, 6, 8, 1, 9, 3]);
  const boxIndex = new WasmIndex2D(boxes, 16);
  const boxHits = Array.from(boxIndex.search(1, 1, 5.5, 5.5)).sort((a, b) => a - b);
  if (boxHits.length !== 2 || boxHits[0] !== 0 || boxHits[1] !== 1) {
    throw new Error(`WASM box validation failed: got [${boxHits}], expected [0,1]`);
  }

  const empty = new WasmIndex2D(new Float64Array(), 16);
  if (empty.search(0, 0, 1, 1).length !== 0) {
    throw new Error('empty index returned hits');
  }
}

function validateWasm3DWrapper(): void {
  const sample = new Float64Array([0, 0, 0, 5, 5, 5, 10, 10, 10, 6, 4, 5]);
  const sampleIndex = WasmIndex3D.from_points(sample, 16);
  const actual = Array.from(sampleIndex.search(4, 4, 4, 7, 7, 7)).sort((a, b) => a - b);
  const expected = bruteForceSearch3d(sample, 4, 4, 4, 7, 7, 7);
  if (actual.length !== expected.length || actual.some((value, i) => value !== expected[i])) {
    throw new Error(`WASM 3D validation failed: got [${actual}], expected [${expected}]`);
  }
  const profile = sampleIndex.search_profile(4, 4, 4, 7, 7, 7) as SearchProfile;
  const profiled = Array.from(profile.hits).sort((a, b) => a - b);
  if (profiled.length !== expected.length || profiled.some((value, i) => value !== expected[i])) {
    throw new Error(`WASM 3D profile validation failed: got [${profiled}], expected [${expected}]`);
  }
  const nearest = Array.from(sampleIndex.neighbors(5.5, 4.5, 5.0, 2)).sort((a, b) => a - b);
  if (nearest.length !== 2 || nearest[0] !== 1 || nearest[1] !== 3) {
    throw new Error(`WASM 3D neighbors validation failed: got [${nearest}], expected [1,3]`);
  }
  const nearestWithin = Array.from(sampleIndex.neighbors_within(5.5, 4.5, 5.0, 4, 1.0)).sort((a, b) => a - b);
  if (nearestWithin.length !== 2 || nearestWithin[0] !== 1 || nearestWithin[1] !== 3) {
    throw new Error(`WASM 3D neighbors_within validation failed: got [${nearestWithin}], expected [1,3]`);
  }
  const extent = sampleIndex.extent() as IndexExtent;
  if (
    extent.empty ||
    extent.minX !== 0 ||
    extent.minY !== 0 ||
    extent.minZ !== 0 ||
    extent.maxX !== 10 ||
    extent.maxY !== 10 ||
    extent.maxZ !== 10
  ) {
    throw new Error(`WASM 3D extent validation failed: got ${JSON.stringify(extent)}`);
  }
  if (!sampleIndex.any(4, 4, 4, 7, 7, 7) || sampleIndex.any(20, 20, 20, 30, 30, 30)) {
    throw new Error('WASM 3D any validation failed');
  }
  const first = sampleIndex.first(4, 4, 4, 7, 7, 7);
  if (first !== 1 && first !== 3) {
    throw new Error(`WASM 3D first validation failed: got ${first}, expected 1 or 3`);
  }
  const roundtrip = WasmIndex3D.from_bytes(sampleIndex.to_bytes());
  const roundtripHits = Array.from(roundtrip.search(4, 4, 4, 7, 7, 7)).sort((a, b) => a - b);
  if (roundtripHits.length !== expected.length || roundtripHits.some((value, i) => value !== expected[i])) {
    throw new Error(`WASM 3D persistence validation failed: got [${roundtripHits}], expected [${expected}]`);
  }

  let rejectedOddLength = false;
  try {
    WasmIndex3D.from_points(new Float64Array([1, 2, 3, 4]), 16);
  } catch {
    rejectedOddLength = true;
  }
  if (!rejectedOddLength) {
    throw new Error('odd-length 3D input was accepted');
  }

  const boxes = new Float64Array([0, 0, 0, 2, 2, 2, 5, 5, 5, 6, 6, 6, 8, 1, 2, 9, 3, 4]);
  const boxIndex = new WasmIndex3D(boxes, 16);
  const boxHits = Array.from(boxIndex.search(1, 1, 1, 5.5, 5.5, 5.5)).sort((a, b) => a - b);
  if (boxHits.length !== 2 || boxHits[0] !== 0 || boxHits[1] !== 1) {
    throw new Error(`WASM 3D box validation failed: got [${boxHits}], expected [0,1]`);
  }

  const empty = new WasmIndex3D(new Float64Array(), 16);
  if (empty.search(0, 0, 0, 1, 1, 1).length !== 0) {
    throw new Error('empty 3D index returned hits');
  }
}

function bruteForceSearch(
  data: Float64Array,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
): number[] {
  const out: number[] = [];
  for (let i = 0; i < data.length; i += 2) {
    const x = data[i];
    const y = data[i + 1];
    if (x >= minX && x <= maxX && y >= minY && y <= maxY) {
      out.push(i / 2);
    }
  }
  return out;
}

function bruteForceSearch3d(
  data: Float64Array,
  minX: number,
  minY: number,
  minZ: number,
  maxX: number,
  maxY: number,
  maxZ: number,
): number[] {
  const out: number[] = [];
  for (let i = 0; i < data.length; i += 3) {
    const x = data[i];
    const y = data[i + 1];
    const z = data[i + 2];
    if (x >= minX && x <= maxX && y >= minY && y <= maxY && z >= minZ && z <= maxZ) {
      out.push(i / 3);
    }
  }
  return out;
}
