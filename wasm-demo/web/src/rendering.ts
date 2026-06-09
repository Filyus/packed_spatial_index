export type Geometry = 'boxes' | 'points';
export type Dimension = '2d' | '3d';

export type WorldView = {
  width: number;
  height: number;
};

export type Rgba = readonly [number, number, number, number];

export type SceneColors = {
  background: Rgba;
  point: Rgba;
  hit: Rgba;
  box: Rgba;
  hitBox: Rgba;
};

/**
 * Everything a renderer needs to draw one frame. Both backends consume the same
 * scene; per-frame state (geometry, dimension, colors, sizes, the view) lives
 * here, while uploaded item/hit buffers are retained inside each renderer.
 */
export type Scene = {
  geometry: Geometry;
  dimension: Dimension;
  itemCount: number;
  itemStride: number;
  worldView: WorldView;
  pointSize: number;
  hitSize: number;
  colors: SceneColors;
  zValueForPoint(data: Float64Array<ArrayBufferLike>, offset: number): number;
  zValueForBox(data: Float64Array<ArrayBufferLike>, offset: number): number;
};

/**
 * Unified backend contract shared by the WebGL and WebGPU renderers.
 *
 * Lifecycle: `uploadItems` whenever the dataset changes, `uploadHits` whenever
 * the query result changes, and `render` every frame. Each renderer owns its
 * canvas and retains the item/hit buffers between calls.
 */
export interface Renderer {
  readonly canvas: HTMLCanvasElement;
  uploadItems(data: Float64Array<ArrayBufferLike>, scene: Scene): void;
  uploadHits(hitIndices: Uint32Array<ArrayBufferLike>): void;
  render(scene: Scene): void;
}

export const SCENE_COLORS: SceneColors = {
  background: [5 / 255, 7 / 255, 10 / 255, 1],
  point: [68 / 255, 174 / 255, 210 / 255, 0.74],
  hit: [1, 1, 1, 0.98],
  box: [68 / 255, 174 / 255, 210 / 255, 0.24],
  hitBox: [1, 1, 1, 0.72],
};

/** Number of `f64` fields per item in the packed item array. */
export function itemStrideFor(dimension: Dimension, geometry: Geometry): number {
  if (geometry === 'boxes') {
    return dimension === '3d' ? 6 : 4;
  }
  return dimension === '3d' ? 3 : 2;
}

/** Point sprite size in CSS pixels, scaled down as the dataset grows. */
export function pointSizeForCount(count: number): number {
  if (count <= 5_000) {
    return 2.1;
  }
  if (count <= 50_000) {
    return 1.55;
  }
  return 1.0;
}

/** Highlighted-hit sprite size in CSS pixels, scaled down as hits grow. */
export function hitSizeForCount(count: number): number {
  if (count <= 5_000) {
    return 3.0;
  }
  if (count <= 50_000) {
    return 2.35;
  }
  if (count <= 300_000) {
    return 1.8;
  }
  return 1.45;
}

/** Map a world-space depth to the 0..1 range used by the depth gradient. */
export function normalizeDepthColor(z: number, worldZSize: number): number {
  return Math.min(1, Math.max(0, z / worldZSize));
}

export function zValueForPoint(
  data: Float64Array<ArrayBufferLike>,
  offset: number,
  dimension: Dimension,
  worldZSize: number,
): number {
  if (dimension !== '3d') {
    return 0.5;
  }
  return normalizeDepthColor(data[offset + 2], worldZSize);
}

export function zValueForBox(
  data: Float64Array<ArrayBufferLike>,
  offset: number,
  dimension: Dimension,
  worldZSize: number,
): number {
  if (dimension !== '3d') {
    return 0.5;
  }
  return normalizeDepthColor((data[offset + 2] + data[offset + 5]) * 0.5, worldZSize);
}
