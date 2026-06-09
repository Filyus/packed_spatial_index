import { itemStrideFor, type Dimension, type Geometry, type WorldView } from './rendering';

export type Distribution = 'uniform' | 'clustered';

export type DatasetOptions = {
  count: number;
  geometry: Geometry;
  distribution: Distribution;
  dimension: Dimension;
  worldView: WorldView;
  worldZSize: number;
};

type Cluster = {
  x: number;
  y: number;
  z: number;
  sigmaX: number;
  sigmaY: number;
  sigmaZ: number;
  rotation: number;
};

/** Generate a packed item array (points or boxes) for the given world. */
export function generateDataset(options: DatasetOptions): Float64Array {
  return options.geometry === 'boxes' ? generateBoxes(options) : generatePoints(options);
}

/** Clamp a coordinate into `[min, max]`, mapping non-finite values to `min`. */
export function clampCoordinate(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.min(max, Math.max(min, value));
}

function generatePoints(options: DatasetOptions): Float64Array {
  const { count, distribution, dimension, worldView, worldZSize } = options;
  const stride = itemStrideFor(dimension, 'points');
  const out = new Float64Array(count * stride);
  if (distribution === 'clustered') {
    const clusters = generateClusters(options);
    for (let i = 0; i < count; i++) {
      const point = randomGaussianClusterPoint(clusters[i % clusters.length]);
      writePoint(out, i, point.x, point.y, point.z, options);
    }
    return out;
  }

  for (let i = 0; i < count; i++) {
    writePoint(out, i, Math.random() * worldView.width, Math.random() * worldView.height, Math.random() * worldZSize, options);
  }
  return out;
}

function generateBoxes(options: DatasetOptions): Float64Array {
  const { count, distribution, dimension, worldView, worldZSize } = options;
  const stride = itemStrideFor(dimension, 'boxes');
  const out = new Float64Array(count * stride);
  const minWorldSide = Math.min(worldView.width, worldView.height);
  const minSize = minWorldSide * 0.005;
  const maxSize = minWorldSide * 0.022;
  const minDepth = worldZSize * 0.005;
  const maxDepth = worldZSize * 0.022;

  if (distribution === 'clustered') {
    const clusters = generateClusters(options);
    for (let i = 0; i < count; i++) {
      const center = randomGaussianClusterPoint(clusters[i % clusters.length]);
      writeRandomBox(out, i, center.x, center.y, center.z, minSize, maxSize, minDepth, maxDepth, options);
    }
    return out;
  }

  for (let i = 0; i < count; i++) {
    writeRandomBox(
      out,
      i,
      Math.random() * worldView.width,
      Math.random() * worldView.height,
      Math.random() * worldZSize,
      minSize,
      maxSize,
      minDepth,
      maxDepth,
      options,
    );
  }
  return out;
}

function generateClusters(options: DatasetOptions): Cluster[] {
  const { worldView, worldZSize } = options;
  const minWorldSide = Math.min(worldView.width, worldView.height);
  const centerMargin = minWorldSide * 0.22;
  const zMargin = worldZSize * 0.22;
  const sigma = minWorldSide * 0.032;
  const sigmaZ = worldZSize * 0.032;
  return Array.from({ length: 12 }, () => ({
    x: centerMargin + Math.random() * (worldView.width - centerMargin * 2),
    y: centerMargin + Math.random() * (worldView.height - centerMargin * 2),
    z: zMargin + Math.random() * (worldZSize - zMargin * 2),
    sigmaX: sigma * (0.85 + Math.random() * 0.35),
    sigmaY: sigma * (0.85 + Math.random() * 0.35),
    sigmaZ: sigmaZ * (0.85 + Math.random() * 0.35),
    rotation: Math.random() * Math.PI,
  }));
}

function writePoint(out: Float64Array, index: number, x: number, y: number, z: number, options: DatasetOptions): void {
  const { dimension, worldView, worldZSize } = options;
  const offset = index * itemStrideFor(dimension, 'points');
  out[offset] = clampCoordinate(x, 0, worldView.width);
  out[offset + 1] = clampCoordinate(y, 0, worldView.height);
  if (dimension === '3d') {
    out[offset + 2] = clampCoordinate(z, 0, worldZSize);
  }
}

function writeRandomBox(
  out: Float64Array,
  index: number,
  centerX: number,
  centerY: number,
  centerZ: number,
  minSize: number,
  maxSize: number,
  minDepth: number,
  maxDepth: number,
  options: DatasetOptions,
): void {
  const { dimension, worldView, worldZSize } = options;
  const width = minSize + Math.random() * Math.random() * (maxSize - minSize);
  const height = minSize + Math.random() * Math.random() * (maxSize - minSize);
  const depth = minDepth + Math.random() * Math.random() * (maxDepth - minDepth);
  const minX = clampCoordinate(centerX - width * 0.5, 0, worldView.width);
  const minY = clampCoordinate(centerY - height * 0.5, 0, worldView.height);
  const maxX = clampCoordinate(centerX + width * 0.5, minX, worldView.width);
  const maxY = clampCoordinate(centerY + height * 0.5, minY, worldView.height);
  const offset = index * itemStrideFor(dimension, 'boxes');
  out[offset] = minX;
  out[offset + 1] = minY;
  if (dimension === '3d') {
    const minZ = clampCoordinate(centerZ - depth * 0.5, 0, worldZSize);
    const maxZ = clampCoordinate(centerZ + depth * 0.5, minZ, worldZSize);
    out[offset + 2] = minZ;
    out[offset + 3] = maxX;
    out[offset + 4] = maxY;
    out[offset + 5] = maxZ;
  } else {
    out[offset + 2] = maxX;
    out[offset + 3] = maxY;
  }
}

function randomGaussianClusterPoint(cluster: Cluster): { x: number; y: number; z: number } {
  const gx = randomNormal();
  const gy = randomNormal();
  const gz = randomNormal();
  const localX = gx * cluster.sigmaX;
  const localY = gy * cluster.sigmaY;
  const cos = Math.cos(cluster.rotation);
  const sin = Math.sin(cluster.rotation);
  return {
    x: cluster.x + localX * cos - localY * sin,
    y: cluster.y + localX * sin + localY * cos,
    z: cluster.z + gz * cluster.sigmaZ,
  };
}

function randomNormal(): number {
  const u = Math.max(Math.random(), Number.EPSILON);
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
