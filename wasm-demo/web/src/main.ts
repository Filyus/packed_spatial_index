import init, { WasmIndex2D, WasmIndex3D } from '../pkg/packed_spatial_index_wasm_demo';

type Distribution = 'uniform' | 'clustered';
type QueryMode = 'range' | 'nearest';
type ResultMode = 'all' | 'any' | 'first';
type Geometry = 'boxes' | 'points';
type Dimension = '2d' | '3d';
type WasmIndex = WasmIndex2D | WasmIndex3D;

type QueryRect = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

type QueryPoint = {
  x: number;
  y: number;
};

type DepthSlice = {
  min: number;
  max: number;
  center: number;
  thickness: number;
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

type SearchProfile = {
  hits: Uint32Array<ArrayBufferLike>;
  traverseMs: number;
  convertMs: number;
  copyMs: number;
  totalMs: number;
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

type WorldView = {
  width: number;
  height: number;
};

type WebGlRenderer = {
  gl: WebGL2RenderingContext;
  pointProgram: WebGLProgram;
  boxProgram: WebGLProgram;
  positionLocation: number;
  colorLocation: WebGLUniformLocation;
  pointSizeLocation: WebGLUniformLocation;
  worldSizeLocation: WebGLUniformLocation;
  boxCornerLocation: number;
  boxLocation: number;
  boxColorLocation: WebGLUniformLocation;
  boxWorldSizeLocation: WebGLUniformLocation;
  unitQuadBuffer: WebGLBuffer;
  pointBuffer: WebGLBuffer;
  hitBuffer: WebGLBuffer;
  boxBuffer: WebGLBuffer;
  hitBoxBuffer: WebGLBuffer;
};

const WORLD_SIZE = 10_000;
const WORLD_Z_SIZE = 10_000;
const BACKGROUND = [247 / 255, 247 / 255, 242 / 255, 1] as const;
const POINT_COLOR = [37 / 255, 51 / 255, 63 / 255, 0.46] as const;
const HIT_COLOR = [32 / 255, 131 / 255, 74 / 255, 0.95] as const;
const BOX_COLOR = [37 / 255, 51 / 255, 63 / 255, 0.12] as const;
const HIT_BOX_COLOR = [32 / 255, 131 / 255, 74 / 255, 0.62] as const;

const stage = mustQuery<HTMLElement>('.stage');
const glCanvas = mustQuery<HTMLCanvasElement>('#glCanvas');
const queryRectEl = mustQuery<HTMLDivElement>('#queryRect');
const queryPointEl = mustQuery<HTMLDivElement>('#queryPoint');
const firstHitEl = mustQuery<HTMLDivElement>('#firstHit');
const pointCountInput = mustQuery<HTMLInputElement>('#pointCount');
const dimensionSelect = mustQuery<HTMLSelectElement>('#dimension');
const geometrySelect = mustQuery<HTMLSelectElement>('#geometry');
const nodeSizeSelect = mustQuery<HTMLSelectElement>('#nodeSize');
const modeSelect = mustQuery<HTMLSelectElement>('#mode');
const resultModeSelect = mustQuery<HTMLSelectElement>('#resultMode');
const neighborCountInput = mustQuery<HTMLInputElement>('#neighborCount');
const maxDistanceInput = mustQuery<HTMLInputElement>('#maxDistance');
const depthInput = mustQuery<HTMLInputElement>('#depth');
const thicknessInput = mustQuery<HTMLInputElement>('#thickness');
const depthLabel = mustParentLabel(depthInput);
const thicknessLabel = mustParentLabel(thicknessInput);
const distributionSelect = mustQuery<HTMLSelectElement>('#distribution');
const roundtripButton = mustQuery<HTMLButtonElement>('#roundtrip');
const regenerateButton = mustQuery<HTMLButtonElement>('#regenerate');
const statusEl = mustQuery<HTMLSpanElement>('#status');
const buildTimeEl = mustQuery<HTMLElement>('#buildTime');
const queryTimeEl = mustQuery<HTMLElement>('#queryTime');
const coreTimeEl = mustQuery<HTMLElement>('#coreTime');
const convertTimeEl = mustQuery<HTMLElement>('#convertTime');
const copyTimeEl = mustQuery<HTMLElement>('#copyTime');
const resultLabelEl = mustQuery<HTMLElement>('#resultLabel');
const hitCountEl = mustQuery<HTMLElement>('#hitCount');
const pointTotalEl = mustQuery<HTMLElement>('#pointTotal');

const renderer = createWebGlRenderer(glCanvas);

let items: Float64Array<ArrayBufferLike> = new Float64Array();
let itemCount = 0;
let index: WasmIndex | null = null;
let hits: Uint32Array<ArrayBufferLike> = new Uint32Array();
let hitClipCoords = new Float32Array();
let hitBoxCoords = new Float32Array();
let query: QueryRect | null = null;
let queryPoint: QueryPoint | null = null;
let depthSlice: DepthSlice = defaultDepthSlice();
let dragStart: { x: number; y: number } | null = null;
let buildMs = 0;
let queryMs = 0;
let coreMs = 0;
let convertMs = 0;
let copyMs = 0;
let resultSummary: string | null = null;
let anyResult: boolean | null = null;
let renderPending = false;
let worldView: WorldView = { width: WORLD_SIZE, height: WORLD_SIZE };
let hasBuiltInitialDataset = false;
let observedStageWidth = 0;
let observedStageHeight = 0;

await init();
validateWasmWrapper();
statusEl.textContent = 'WASM SIMD build loaded';
rebuild();

regenerateButton.addEventListener('click', rebuild);
pointCountInput.addEventListener('change', rebuild);
dimensionSelect.addEventListener('change', () => {
  syncModeControls();
  rebuild();
});
geometrySelect.addEventListener('change', rebuild);
nodeSizeSelect.addEventListener('change', rebuildIndex);
modeSelect.addEventListener('change', () => {
  syncModeControls();
  search();
});
resultModeSelect.addEventListener('change', search);
neighborCountInput.addEventListener('change', search);
maxDistanceInput.addEventListener('change', search);
depthInput.addEventListener('change', () => {
  syncZInputs();
  search();
});
thicknessInput.addEventListener('change', () => {
  syncZInputs();
  search();
});
distributionSelect.addEventListener('change', rebuild);
roundtripButton.addEventListener('click', roundtripIndex);

stage.addEventListener('pointerdown', (event) => {
  stage.setPointerCapture(event.pointerId);
  const point = canvasToWorld(event);
  if (currentMode() === 'nearest') {
    queryPoint = point;
  } else {
    dragStart = point;
    query = { minX: dragStart.x, minY: dragStart.y, maxX: dragStart.x, maxY: dragStart.y };
  }
  search();
});

stage.addEventListener('pointermove', (event) => {
  const current = canvasToWorld(event);
  if (currentMode() === 'nearest') {
    if (event.buttons !== 1 && queryPoint) {
      return;
    }
    queryPoint = current;
  } else {
    if (!dragStart) {
      return;
    }
    query = normalizeRect(dragStart.x, dragStart.y, current.x, current.y);
  }
  search();
});

stage.addEventListener('pointerup', (event) => {
  stage.releasePointerCapture(event.pointerId);
  dragStart = null;
});

stage.addEventListener('pointercancel', () => {
  dragStart = null;
});

new ResizeObserver(() => {
  const resized = rememberStageSize();
  updateWorldView();
  if (hasBuiltInitialDataset && resized) {
    scheduleRender();
    statusEl.textContent = 'Resized; regenerate to refill view';
  }
}).observe(stage);
syncModeControls();
updateWorldView();

function rebuild(): void {
  updateWorldView();
  rememberStageSize();
  syncZInputs();
  const count = normalizePointCount(Number(pointCountInput.value));
  pointCountInput.value = String(count);
  const geometry = currentGeometry();
  const distribution = distributionSelect.value as Distribution;

  items = geometry === 'boxes' ? generateBoxes(count, distribution) : generatePoints(count, distribution);
  itemCount = count;
  uploadItems(renderer, items);
  hasBuiltInitialDataset = true;
  rebuildIndex();
}

function rebuildIndex(): void {
  const nodeSize = Number(nodeSizeSelect.value);

  const started = performance.now();
  index = buildCurrentIndex(items, nodeSize);
  buildMs = performance.now() - started;
  statusEl.textContent = statusText('WASM SIMD build loaded');
  query ??= defaultQuery();
  queryPoint ??= defaultQueryPoint();
  search();
}

function search(): void {
  if (!index) {
    return;
  }
  if (currentMode() === 'nearest') {
    if (!queryPoint) {
      return;
    }
    const started = performance.now();
    const maxDistance = normalizeMaxDistance(Number(maxDistanceInput.value));
    const maxResults = normalizeNeighborCount(Number(neighborCountInput.value));
    hits =
      currentDimension() === '3d'
        ? searchNearest3d(index as WasmIndex3D, queryPoint, depthSlice.center, maxResults, maxDistance)
        : searchNearest2d(index as WasmIndex2D, queryPoint, maxResults, maxDistance);
    queryMs = performance.now() - started;
    coreMs = queryMs;
    convertMs = 0;
    copyMs = 0;
    resultSummary = null;
    anyResult = null;
  } else {
    if (!query) {
      return;
    }
    if (currentResultMode() === 'any') {
      const started = performance.now();
      const found =
        currentDimension() === '3d'
          ? (index as WasmIndex3D).any(
              query.minX,
              query.minY,
              depthSlice.min,
              query.maxX,
              query.maxY,
              depthSlice.max,
            )
          : (index as WasmIndex2D).any(query.minX, query.minY, query.maxX, query.maxY);
      queryMs = performance.now() - started;
      coreMs = queryMs;
      convertMs = 0;
      copyMs = 0;
      hits = new Uint32Array();
      resultSummary = found ? 'true' : 'false';
      anyResult = found;
    } else if (currentResultMode() === 'first') {
      const started = performance.now();
      const first =
        currentDimension() === '3d'
          ? (index as WasmIndex3D).first(
              query.minX,
              query.minY,
              depthSlice.min,
              query.maxX,
              query.maxY,
              depthSlice.max,
            )
          : (index as WasmIndex2D).first(query.minX, query.minY, query.maxX, query.maxY);
      queryMs = performance.now() - started;
      coreMs = queryMs;
      convertMs = 0;
      copyMs = 0;
      hits = first >= 0 ? new Uint32Array([first]) : new Uint32Array();
      resultSummary = first >= 0 ? first.toLocaleString() : 'none';
      anyResult = null;
    } else {
      const profile =
        currentDimension() === '3d'
          ? ((index as WasmIndex3D).search_profile(
              query.minX,
              query.minY,
              depthSlice.min,
              query.maxX,
              query.maxY,
              depthSlice.max,
            ) as SearchProfile)
          : ((index as WasmIndex2D).search_profile(query.minX, query.minY, query.maxX, query.maxY) as SearchProfile);
      hits = profile.hits;
      queryMs = profile.totalMs;
      coreMs = profile.traverseMs;
      convertMs = profile.convertMs;
      copyMs = profile.copyMs;
      resultSummary = null;
      anyResult = null;
    }
  }

  uploadHits(renderer, items, hits);
  scheduleRender();
}

function roundtripIndex(): void {
  if (!index) {
    return;
  }

  const serializeStarted = performance.now();
  const bytes = index.to_bytes();
  const serializeMs = performance.now() - serializeStarted;
  const loadStarted = performance.now();
  index = currentDimension() === '3d' ? WasmIndex3D.from_bytes(bytes) : WasmIndex2D.from_bytes(bytes);
  const loadMs = performance.now() - loadStarted;
  statusEl.textContent = statusText(
    `Roundtrip ${bytes.byteLength.toLocaleString()} bytes, save ${formatDuration(serializeMs)}, load ${formatDuration(loadMs)}`,
  );
  search();
}

function scheduleRender(): void {
  if (renderPending) {
    return;
  }
  renderPending = true;
  requestAnimationFrame(() => {
    renderPending = false;
    render();
  });
}

function render(): void {
  resizeCanvasToDisplaySize(glCanvas);

  const { gl } = renderer;
  gl.viewport(0, 0, glCanvas.width, glCanvas.height);
  gl.clearColor(BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], BACKGROUND[3]);
  gl.clear(gl.COLOR_BUFFER_BIT);

  if (currentGeometry() === 'boxes') {
    drawBoxes(renderer, renderer.boxBuffer, itemCount, BOX_COLOR);
    drawBoxes(renderer, renderer.hitBoxBuffer, hits.length, HIT_BOX_COLOR);
  } else {
    drawPoints(renderer, renderer.pointBuffer, itemCount, POINT_COLOR, pointSizeForCount(itemCount));
    drawPoints(renderer, renderer.hitBuffer, hits.length, HIT_COLOR, hitSizeForCount(itemCount));
  }
  renderQueryOverlay();
  renderFirstHitOverlay();

  buildTimeEl.textContent = formatDuration(buildMs);
  queryTimeEl.textContent = formatQueryDuration(queryMs);
  coreTimeEl.textContent = formatQueryDuration(coreMs);
  convertTimeEl.textContent = formatQueryDuration(convertMs);
  copyTimeEl.textContent = formatQueryDuration(copyMs);
  resultLabelEl.textContent = resultLabel();
  hitCountEl.textContent = resultSummary ?? hits.length.toLocaleString();
  pointTotalEl.textContent = itemCount.toLocaleString();
}

function renderQueryOverlay(): void {
  const mode = currentMode();
  if (mode !== 'range' || !query) {
    queryRectEl.style.display = 'none';
  } else {
    const x0 = worldToCanvasX(query.minX);
    const y0 = worldToCanvasY(query.minY);
    const x1 = worldToCanvasX(query.maxX);
    const y1 = worldToCanvasY(query.maxY);
    queryRectEl.style.display = 'block';
    queryRectEl.classList.toggle('is-match', anyResult === true);
    queryRectEl.classList.toggle('is-miss', anyResult === false);
    queryRectEl.style.transform = `translate(${x0}px, ${y0}px)`;
    queryRectEl.style.width = `${x1 - x0}px`;
    queryRectEl.style.height = `${y1 - y0}px`;
  }

  if (mode !== 'nearest' || !queryPoint) {
    queryPointEl.style.display = 'none';
  } else {
    queryPointEl.style.display = 'block';
    queryPointEl.style.transform = `translate(${worldToCanvasX(queryPoint.x)}px, ${worldToCanvasY(queryPoint.y)}px)`;
  }
}

function renderFirstHitOverlay(): void {
  if (currentMode() !== 'range' || currentResultMode() !== 'first' || hits.length === 0) {
    firstHitEl.style.display = 'none';
    return;
  }

  const index = hits[0];
  const stride = itemStride();
  let x0: number;
  let y0: number;
  let x1: number;
  let y1: number;
  if (currentGeometry() === 'boxes') {
    const offset = index * stride;
    x0 = worldToCanvasX(items[offset]);
    y0 = worldToCanvasY(items[offset + 1]);
    const maxOffset = currentDimension() === '3d' ? offset + 3 : offset + 2;
    x1 = worldToCanvasX(items[maxOffset]);
    y1 = worldToCanvasY(items[maxOffset + 1]);
    firstHitEl.classList.remove('is-point');
    firstHitEl.classList.add('is-box');
  } else {
    const offset = index * stride;
    const x = worldToCanvasX(items[offset]);
    const y = worldToCanvasY(items[offset + 1]);
    const radius = 8;
    x0 = x - radius;
    y0 = y - radius;
    x1 = x + radius;
    y1 = y + radius;
    firstHitEl.classList.remove('is-box');
    firstHitEl.classList.add('is-point');
  }

  const centerX = (x0 + x1) * 0.5;
  const centerY = (y0 + y1) * 0.5;
  const minSize = currentGeometry() === 'boxes' ? 10 : 16;
  const padding = currentGeometry() === 'boxes' ? 4 : 0;
  const width = Math.max(minSize, Math.abs(x1 - x0) + padding * 2);
  const height = Math.max(minSize, Math.abs(y1 - y0) + padding * 2);
  const left = centerX - width * 0.5;
  const top = centerY - height * 0.5;
  firstHitEl.style.display = 'block';
  firstHitEl.style.transform = `translate(${left}px, ${top}px)`;
  firstHitEl.style.width = `${width}px`;
  firstHitEl.style.height = `${height}px`;
}

function drawPoints(
  renderer: WebGlRenderer,
  buffer: WebGLBuffer,
  count: number,
  color: readonly [number, number, number, number],
  pointSize: number,
): void {
  if (count === 0) {
    return;
  }

  const { gl } = renderer;
  gl.useProgram(renderer.pointProgram);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(renderer.positionLocation);
  gl.vertexAttribPointer(renderer.positionLocation, 2, gl.FLOAT, false, 0, 0);
  gl.uniform4f(renderer.colorLocation, color[0], color[1], color[2], color[3]);
  gl.uniform1f(renderer.pointSizeLocation, pointSize * (window.devicePixelRatio || 1));
  gl.uniform2f(renderer.worldSizeLocation, worldView.width, worldView.height);
  gl.drawArrays(gl.POINTS, 0, count);
}

function drawBoxes(
  renderer: WebGlRenderer,
  buffer: WebGLBuffer,
  count: number,
  color: readonly [number, number, number, number],
): void {
  if (count === 0) {
    return;
  }

  const { gl } = renderer;
  gl.useProgram(renderer.boxProgram);

  gl.bindBuffer(gl.ARRAY_BUFFER, renderer.unitQuadBuffer);
  gl.enableVertexAttribArray(renderer.boxCornerLocation);
  gl.vertexAttribPointer(renderer.boxCornerLocation, 2, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(renderer.boxCornerLocation, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(renderer.boxLocation);
  gl.vertexAttribPointer(renderer.boxLocation, 4, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(renderer.boxLocation, 1);

  gl.uniform4f(renderer.boxColorLocation, color[0], color[1], color[2], color[3]);
  gl.uniform2f(renderer.boxWorldSizeLocation, worldView.width, worldView.height);
  gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, count);
  gl.vertexAttribDivisor(renderer.boxLocation, 0);
}

function uploadItems(renderer: WebGlRenderer, data: Float64Array<ArrayBufferLike>): void {
  if (currentGeometry() === 'boxes') {
    uploadBoxes(renderer, data);
  } else {
    uploadPoints(renderer, data);
  }
}

function uploadPoints(renderer: WebGlRenderer, data: Float64Array<ArrayBufferLike>): void {
  const { gl } = renderer;
  gl.bindBuffer(gl.ARRAY_BUFFER, renderer.pointBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, toFloat32Points(data), gl.STATIC_DRAW);
}

function uploadBoxes(renderer: WebGlRenderer, data: Float64Array<ArrayBufferLike>): void {
  const { gl } = renderer;
  gl.bindBuffer(gl.ARRAY_BUFFER, renderer.boxBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, toFloat32Boxes(data), gl.STATIC_DRAW);
}

function uploadHits(
  renderer: WebGlRenderer,
  data: Float64Array<ArrayBufferLike>,
  hitIndices: Uint32Array<ArrayBufferLike>,
): void {
  if (currentGeometry() === 'boxes') {
    uploadHitBoxes(renderer, data, hitIndices);
    return;
  }

  if (hitClipCoords.length !== hitIndices.length * 2) {
    hitClipCoords = new Float32Array(hitIndices.length * 2);
  }

  for (let i = 0; i < hitIndices.length; i++) {
    const offset = hitIndices[i] * itemStride();
    const out = i * 2;
    hitClipCoords[out] = data[offset];
    hitClipCoords[out + 1] = data[offset + 1];
  }

  const { gl } = renderer;
  gl.bindBuffer(gl.ARRAY_BUFFER, renderer.hitBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, hitClipCoords, gl.DYNAMIC_DRAW);
}

function uploadHitBoxes(
  renderer: WebGlRenderer,
  data: Float64Array<ArrayBufferLike>,
  hitIndices: Uint32Array<ArrayBufferLike>,
): void {
  if (hitBoxCoords.length !== hitIndices.length * 4) {
    hitBoxCoords = new Float32Array(hitIndices.length * 4);
  }

  for (let i = 0; i < hitIndices.length; i++) {
    const offset = hitIndices[i] * itemStride();
    const maxOffset = currentDimension() === '3d' ? offset + 3 : offset + 2;
    const out = i * 4;
    hitBoxCoords[out] = data[offset];
    hitBoxCoords[out + 1] = data[offset + 1];
    hitBoxCoords[out + 2] = data[maxOffset];
    hitBoxCoords[out + 3] = data[maxOffset + 1];
  }

  const { gl } = renderer;
  gl.bindBuffer(gl.ARRAY_BUFFER, renderer.hitBoxBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, hitBoxCoords, gl.DYNAMIC_DRAW);
}

function toFloat32Points(data: Float64Array<ArrayBufferLike>): Float32Array {
  const stride = itemStride();
  const out = new Float32Array(itemCount * 2);
  for (let i = 0; i < itemCount; i++) {
    const offset = i * stride;
    const outOffset = i * 2;
    out[outOffset] = data[offset];
    out[outOffset + 1] = data[offset + 1];
  }
  return out;
}

function toFloat32Boxes(data: Float64Array<ArrayBufferLike>): Float32Array {
  const stride = itemStride();
  const out = new Float32Array(itemCount * 4);
  for (let i = 0; i < itemCount; i++) {
    const offset = i * stride;
    const maxOffset = currentDimension() === '3d' ? offset + 3 : offset + 2;
    const outOffset = i * 4;
    out[outOffset] = data[offset];
    out[outOffset + 1] = data[offset + 1];
    out[outOffset + 2] = data[maxOffset];
    out[outOffset + 3] = data[maxOffset + 1];
  }
  return out;
}

function toClipX(x: number): number {
  return (x / worldView.width) * 2 - 1;
}

function toClipY(y: number): number {
  return 1 - (y / worldView.height) * 2;
}

function pointSizeForCount(count: number): number {
  if (count <= 5_000) {
    return 2.1;
  }
  if (count <= 50_000) {
    return 1.55;
  }
  return 1.0;
}

function hitSizeForCount(count: number): number {
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

function formatDuration(ms: number): string {
  return `${roundToChromeTimerStep(ms).toFixed(1)} ms`;
}

function formatQueryDuration(ms: number): string {
  return `${roundToChromeTimerStep(ms).toFixed(1)} ms`;
}

function roundToChromeTimerStep(ms: number): number {
  return Math.round(ms * 10) / 10;
}

function generatePoints(count: number, distribution: Distribution): Float64Array {
  const stride = itemStrideFor(currentDimension(), 'points');
  const out = new Float64Array(count * stride);
  if (distribution === 'clustered') {
    const clusters = generateClusters();
    for (let i = 0; i < count; i++) {
      const cluster = clusters[i % clusters.length];
      const point = randomGaussianClusterPoint(cluster);
      writePoint(out, i, point.x, point.y, point.z);
    }
    return out;
  }

  for (let i = 0; i < count; i++) {
    writePoint(out, i, Math.random() * worldView.width, Math.random() * worldView.height, Math.random() * WORLD_Z_SIZE);
  }
  return out;
}

function generateBoxes(count: number, distribution: Distribution): Float64Array {
  const stride = itemStrideFor(currentDimension(), 'boxes');
  const out = new Float64Array(count * stride);
  const minWorldSide = Math.min(worldView.width, worldView.height);
  const minSize = minWorldSide * 0.003;
  const maxSize = minWorldSide * 0.014;
  const minDepth = WORLD_Z_SIZE * 0.003;
  const maxDepth = WORLD_Z_SIZE * 0.014;

  if (distribution === 'clustered') {
    const clusters = generateClusters();
    for (let i = 0; i < count; i++) {
      const center = randomGaussianClusterPoint(clusters[i % clusters.length]);
      writeRandomBox(out, i, center.x, center.y, center.z, minSize, maxSize, minDepth, maxDepth);
    }
    return out;
  }

  for (let i = 0; i < count; i++) {
    writeRandomBox(
      out,
      i,
      Math.random() * worldView.width,
      Math.random() * worldView.height,
      Math.random() * WORLD_Z_SIZE,
      minSize,
      maxSize,
      minDepth,
      maxDepth,
    );
  }
  return out;
}

function generateClusters(): Cluster[] {
  const minWorldSide = Math.min(worldView.width, worldView.height);
  const centerMargin = minWorldSide * 0.22;
  const zMargin = WORLD_Z_SIZE * 0.22;
  const sigma = minWorldSide * 0.032;
  const sigmaZ = WORLD_Z_SIZE * 0.032;
  return Array.from({ length: 12 }, () => ({
    x: centerMargin + Math.random() * (worldView.width - centerMargin * 2),
    y: centerMargin + Math.random() * (worldView.height - centerMargin * 2),
    z: zMargin + Math.random() * (WORLD_Z_SIZE - zMargin * 2),
    sigmaX: sigma * (0.85 + Math.random() * 0.35),
    sigmaY: sigma * (0.85 + Math.random() * 0.35),
    sigmaZ: sigmaZ * (0.85 + Math.random() * 0.35),
    rotation: Math.random() * Math.PI,
  }));
}

function writePoint(out: Float64Array, index: number, x: number, y: number, z: number): void {
  const offset = index * itemStrideFor(currentDimension(), 'points');
  out[offset] = clampCoordinate(x, 0, worldView.width);
  out[offset + 1] = clampCoordinate(y, 0, worldView.height);
  if (currentDimension() === '3d') {
    out[offset + 2] = clampCoordinate(z, 0, WORLD_Z_SIZE);
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
): void {
  const width = minSize + Math.random() * Math.random() * (maxSize - minSize);
  const height = minSize + Math.random() * Math.random() * (maxSize - minSize);
  const depth = minDepth + Math.random() * Math.random() * (maxDepth - minDepth);
  const minX = clampCoordinate(centerX - width * 0.5, 0, worldView.width);
  const minY = clampCoordinate(centerY - height * 0.5, 0, worldView.height);
  const maxX = clampCoordinate(centerX + width * 0.5, minX, worldView.width);
  const maxY = clampCoordinate(centerY + height * 0.5, minY, worldView.height);
  const offset = index * itemStrideFor(currentDimension(), 'boxes');
  out[offset] = minX;
  out[offset + 1] = minY;
  if (currentDimension() === '3d') {
    const minZ = clampCoordinate(centerZ - depth * 0.5, 0, WORLD_Z_SIZE);
    const maxZ = clampCoordinate(centerZ + depth * 0.5, minZ, WORLD_Z_SIZE);
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

function clampCoordinate(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.min(max, Math.max(min, value));
}

function defaultQuery(): QueryRect {
  return {
    minX: worldView.width * 0.32,
    minY: worldView.height * 0.32,
    maxX: worldView.width * 0.68,
    maxY: worldView.height * 0.68,
  };
}

function defaultQueryPoint(): QueryPoint {
  return {
    x: worldView.width * 0.5,
    y: worldView.height * 0.5,
  };
}

function currentMode(): QueryMode {
  return modeSelect.value as QueryMode;
}

function currentDimension(): Dimension {
  return dimensionSelect.value as Dimension;
}

function currentResultMode(): ResultMode {
  return resultModeSelect.value as ResultMode;
}

function currentGeometry(): Geometry {
  return geometrySelect.value as Geometry;
}

function syncModeControls(): void {
  const nearest = currentMode() === 'nearest';
  const is3d = currentDimension() === '3d';
  neighborCountInput.disabled = !nearest;
  maxDistanceInput.disabled = !nearest;
  resultModeSelect.disabled = nearest;
  depthLabel.hidden = !is3d;
  thicknessLabel.hidden = !is3d || nearest;
}

function syncZInputs(): void {
  const center = normalizeZ(Number(depthInput.value), WORLD_Z_SIZE * 0.5);
  const thickness = normalizeThickness(Number(thicknessInput.value));
  const half = thickness * 0.5;
  depthSlice = {
    min: center - half,
    max: center + half,
    center,
    thickness,
  };
  depthInput.value = String(Math.round(depthSlice.center));
  thicknessInput.value = String(Math.round(depthSlice.thickness));
}

function normalizeNeighborCount(value: number): number {
  if (!Number.isFinite(value)) {
    return 500;
  }
  const count = Math.max(1, Math.round(value));
  neighborCountInput.value = String(count);
  return count;
}

function normalizeMaxDistance(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return Number.POSITIVE_INFINITY;
  }
  return value;
}

function normalizeZ(value: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return value;
}

function normalizeThickness(value: number): number {
  if (!Number.isFinite(value)) {
    return WORLD_Z_SIZE * 0.36;
  }
  return Math.max(0, value);
}

function resultLabel(): string {
  if (currentMode() === 'nearest') {
    return 'Hits';
  }
  if (currentResultMode() === 'any') {
    return 'Any';
  }
  if (currentResultMode() === 'first') {
    return 'First';
  }
  return 'Hits';
}

function defaultDepthSlice(): DepthSlice {
  const center = WORLD_Z_SIZE * 0.5;
  const thickness = WORLD_Z_SIZE * 0.36;
  return {
    min: center - thickness * 0.5,
    max: center + thickness * 0.5,
    center,
    thickness,
  };
}

function itemStride(): number {
  return itemStrideFor(currentDimension(), currentGeometry());
}

function itemStrideFor(dimension: Dimension, geometry: Geometry): number {
  if (geometry === 'boxes') {
    return dimension === '3d' ? 6 : 4;
  }
  return dimension === '3d' ? 3 : 2;
}

function buildCurrentIndex(data: Float64Array<ArrayBufferLike>, nodeSize: number): WasmIndex {
  if (currentDimension() === '3d') {
    return currentGeometry() === 'boxes'
      ? new WasmIndex3D(data, nodeSize)
      : WasmIndex3D.from_points(data, nodeSize);
  }
  return currentGeometry() === 'boxes' ? new WasmIndex2D(data, nodeSize) : WasmIndex2D.from_points(data, nodeSize);
}

function searchNearest2d(
  index: WasmIndex2D,
  point: QueryPoint,
  maxResults: number,
  maxDistance: number,
): Uint32Array<ArrayBufferLike> {
  return Number.isFinite(maxDistance)
    ? index.neighbors_within(point.x, point.y, maxResults, maxDistance)
    : index.neighbors(point.x, point.y, maxResults);
}

function searchNearest3d(
  index: WasmIndex3D,
  point: QueryPoint,
  z: number,
  maxResults: number,
  maxDistance: number,
): Uint32Array<ArrayBufferLike> {
  return Number.isFinite(maxDistance)
    ? index.neighbors_within(point.x, point.y, z, maxResults, maxDistance)
    : index.neighbors(point.x, point.y, z, maxResults);
}

function statusText(message: string): string {
  if (currentDimension() === '3d') {
    return `${message}; 3D XY projection, depth ${Math.round(depthSlice.center).toLocaleString()} +/- ${Math.round(
      depthSlice.thickness * 0.5,
    ).toLocaleString()}`;
  }
  return message;
}

function normalizeRect(x0: number, y0: number, x1: number, y1: number): QueryRect {
  return {
    minX: Math.min(x0, x1),
    minY: Math.min(y0, y1),
    maxX: Math.max(x0, x1),
    maxY: Math.max(y0, y1),
  };
}

function canvasToWorld(event: PointerEvent): { x: number; y: number } {
  const rect = stage.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * worldView.width;
  const y = ((event.clientY - rect.top) / rect.height) * worldView.height;
  return {
    x: clampCoordinate(x, 0, worldView.width),
    y: clampCoordinate(y, 0, worldView.height),
  };
}

function worldToCanvasX(x: number): number {
  return (x / worldView.width) * stage.clientWidth;
}

function worldToCanvasY(y: number): number {
  return (y / worldView.height) * stage.clientHeight;
}

function clampNumber(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.min(max, Math.max(min, Math.round(value)));
}

function normalizePointCount(value: number): number {
  if (!Number.isFinite(value)) {
    return 50_000;
  }
  return Math.max(1_000, Math.round(value));
}

function resizeCanvasToDisplaySize(canvas: HTMLCanvasElement): void {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function updateWorldView(): void {
  const width = Math.max(1, stage.clientWidth);
  const height = Math.max(1, stage.clientHeight);
  if (width >= height) {
    worldView = {
      width: WORLD_SIZE * (width / height),
      height: WORLD_SIZE,
    };
  } else {
    worldView = {
      width: WORLD_SIZE,
      height: WORLD_SIZE * (height / width),
    };
  }
}

function rememberStageSize(): boolean {
  const width = Math.round(stage.clientWidth);
  const height = Math.round(stage.clientHeight);
  const changed = width !== observedStageWidth || height !== observedStageHeight;
  observedStageWidth = width;
  observedStageHeight = height;
  return changed;
}

function createWebGlRenderer(canvas: HTMLCanvasElement): WebGlRenderer {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    powerPreference: 'high-performance',
  });
  if (!gl) {
    throw new Error('WebGL2 is not available');
  }

  const pointProgram = createProgram(
    gl,
    `#version 300 es
    in vec2 a_position;
    uniform float u_pointSize;
    uniform vec2 u_worldSize;

    void main() {
      vec2 clip = vec2(
        (a_position.x / u_worldSize.x) * 2.0 - 1.0,
        1.0 - (a_position.y / u_worldSize.y) * 2.0
      );
      gl_Position = vec4(clip, 0.0, 1.0);
      gl_PointSize = u_pointSize;
    }`,
    `#version 300 es
    precision highp float;

    uniform vec4 u_color;
    out vec4 outColor;

    void main() {
      vec2 delta = gl_PointCoord - vec2(0.5);
      if (dot(delta, delta) > 0.25) {
        discard;
      }
      outColor = u_color;
    }`,
  );

  const boxProgram = createProgram(
    gl,
    `#version 300 es
    in vec2 a_corner;
    in vec4 a_box;
    uniform vec2 u_worldSize;

    void main() {
      vec2 world = mix(a_box.xy, a_box.zw, a_corner);
      vec2 clip = vec2(
        (world.x / u_worldSize.x) * 2.0 - 1.0,
        1.0 - (world.y / u_worldSize.y) * 2.0
      );
      gl_Position = vec4(clip, 0.0, 1.0);
    }`,
    `#version 300 es
    precision highp float;

    uniform vec4 u_color;
    out vec4 outColor;

    void main() {
      outColor = u_color;
    }`,
  );

  const unitQuadBuffer = gl.createBuffer();
  const pointBuffer = gl.createBuffer();
  const hitBuffer = gl.createBuffer();
  const boxBuffer = gl.createBuffer();
  const hitBoxBuffer = gl.createBuffer();
  if (!unitQuadBuffer || !pointBuffer || !hitBuffer || !boxBuffer || !hitBoxBuffer) {
    throw new Error('failed to create WebGL buffers');
  }

  gl.bindBuffer(gl.ARRAY_BUFFER, unitQuadBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1]), gl.STATIC_DRAW);

  const colorLocation = gl.getUniformLocation(pointProgram, 'u_color');
  const pointSizeLocation = gl.getUniformLocation(pointProgram, 'u_pointSize');
  const worldSizeLocation = gl.getUniformLocation(pointProgram, 'u_worldSize');
  if (!colorLocation || !pointSizeLocation || !worldSizeLocation) {
    throw new Error('failed to resolve WebGL uniforms');
  }
  const boxColorLocation = gl.getUniformLocation(boxProgram, 'u_color');
  const boxWorldSizeLocation = gl.getUniformLocation(boxProgram, 'u_worldSize');
  if (!boxColorLocation || !boxWorldSizeLocation) {
    throw new Error('failed to resolve WebGL box uniforms');
  }

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  return {
    gl,
    pointProgram,
    boxProgram,
    positionLocation: gl.getAttribLocation(pointProgram, 'a_position'),
    colorLocation,
    pointSizeLocation,
    worldSizeLocation,
    boxCornerLocation: gl.getAttribLocation(boxProgram, 'a_corner'),
    boxLocation: gl.getAttribLocation(boxProgram, 'a_box'),
    boxColorLocation,
    boxWorldSizeLocation,
    unitQuadBuffer,
    pointBuffer,
    hitBuffer,
    boxBuffer,
    hitBoxBuffer,
  };
}

function createProgram(gl: WebGL2RenderingContext, vertexSource: string, fragmentSource: string): WebGLProgram {
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  if (!program) {
    throw new Error('failed to create WebGL program');
  }

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) ?? 'unknown WebGL program error';
    gl.deleteProgram(program);
    throw new Error(log);
  }

  return program;
}

function createShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) {
    throw new Error('failed to create WebGL shader');
  }

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? 'unknown WebGL shader error';
    gl.deleteShader(shader);
    throw new Error(log);
  }

  return shader;
}

function mustQuery<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`missing element: ${selector}`);
  }
  return element;
}

function mustParentLabel(element: HTMLElement): HTMLLabelElement {
  const label = element.closest('label');
  if (!(label instanceof HTMLLabelElement)) {
    throw new Error(`missing parent label for #${element.id}`);
  }
  return label;
}

function validateWasmWrapper(): void {
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
