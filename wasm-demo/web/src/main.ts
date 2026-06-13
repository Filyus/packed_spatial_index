import init, { WasmIndex2D, WasmIndex3D } from '../pkg/packed_spatial_index_wasm_demo';
import {
  SCENE_COLORS,
  hitSizeForCount,
  itemStrideFor,
  pointSizeForCount,
  zValueForBox,
  zValueForPoint,
  type Dimension,
  type Geometry,
  type Renderer,
  type Scene,
  type WorldView,
} from './rendering';
import { clampCoordinate, generateDataset, type Distribution } from './dataset';
import { mustQuery } from './dom';
import { createOverlays, type QueryPoint, type QueryRect } from './overlays';
import { parsePsiHeader, reconstructItems } from './psi-format';
import { validateWasmWrapper } from './wasm-validate';
import { createWebGlRenderer } from './webgl-renderer';
import { createWebGpuRenderer } from './webgpu-renderer';

type QueryMode = 'range' | 'nearest';
type ResultMode = 'all' | 'any' | 'first';
type RendererMode = 'webgl' | 'webgpu';
type WasmIndex = WasmIndex2D | WasmIndex3D;

type DepthSlice = {
  min: number;
  max: number;
  center: number;
  thickness: number;
};

type SearchProfile = {
  hits: Uint32Array<ArrayBufferLike>;
  traverseMs: number;
  convertMs: number;
  copyMs: number;
  totalMs: number;
};

const WORLD_SIZE = 10_000;
const WORLD_Z_SIZE = 10_000;

const stage = mustQuery<HTMLElement>('.stage');
const glCanvas = mustQuery<HTMLCanvasElement>('#glCanvas');
const gpuCanvas = mustQuery<HTMLCanvasElement>('#gpuCanvas');
const errorOverlayEl = mustQuery<HTMLElement>('#errorOverlay');
const loaderOverlayEl = mustQuery<HTMLElement>('#loaderOverlay');
const errorMessageEl = mustQuery<HTMLPreElement>('#errorMessage');
const reloadDemoButton = mustQuery<HTMLButtonElement>('#reloadDemo');
const depthLegendEl = mustQuery<HTMLDivElement>('#depthLegend');
const pointCountInput = mustQuery<HTMLInputElement>('#pointCount');
const dimensionSelect = mustQuery<HTMLSelectElement>('#dimension');
const geometrySelect = mustQuery<HTMLSelectElement>('#geometry');
const rendererSelect = mustQuery<HTMLSelectElement>('#renderer');
const nodeSizeSelect = mustQuery<HTMLSelectElement>('#nodeSize');
const modeSelect = mustQuery<HTMLSelectElement>('#mode');
const resultModeSelect = mustQuery<HTMLSelectElement>('#resultMode');
const neighborCountInput = mustQuery<HTMLInputElement>('#neighborCount');
const maxDistanceInput = mustQuery<HTMLInputElement>('#maxDistance');
const depthInput = mustQuery<HTMLInputElement>('#depth');
const depthValueInput = mustQuery<HTMLInputElement>('#depthValue');
const thicknessInput = mustQuery<HTMLInputElement>('#thickness');
const thicknessValueInput = mustQuery<HTMLInputElement>('#thicknessValue');
const distributionSelect = mustQuery<HTMLSelectElement>('#distribution');
const saveIndexButton = mustQuery<HTMLButtonElement>('#saveIndex');
const loadIndexButton = mustQuery<HTMLButtonElement>('#loadIndex');
const loadIndexFileInput = mustQuery<HTMLInputElement>('#loadIndexFile');
const regenerateButton = mustQuery<HTMLButtonElement>('#regenerate');
const statusEl = mustQuery<HTMLSpanElement>('#status');
const buildLabelEl = mustQuery<HTMLElement>('#buildLabel');
const buildTimeEl = mustQuery<HTMLElement>('#buildTime');
const queryTimeEl = mustQuery<HTMLElement>('#queryTime');
const coreTimeEl = mustQuery<HTMLElement>('#coreTime');
const convertTimeEl = mustQuery<HTMLElement>('#convertTime');
const copyTimeEl = mustQuery<HTMLElement>('#copyTime');
const resultLabelEl = mustQuery<HTMLElement>('#resultLabel');
const hitCountEl = mustQuery<HTMLElement>('#hitCount');
const pointTotalEl = mustQuery<HTMLElement>('#pointTotal');

const webglRenderer: Renderer = createWebGlRenderer(glCanvas);
let webgpuRenderer: Renderer | null = null;
const overlays = createOverlays(stage);

let items: Float64Array<ArrayBufferLike> = new Float64Array();
let itemCount = 0;
let index: WasmIndex | null = null;
let hits: Uint32Array<ArrayBufferLike> = new Uint32Array();
let query: QueryRect | null = null;
let queryPoint: QueryPoint | null = null;
let depthSlice: DepthSlice = defaultDepthSlice();
let dragStart: { x: number; y: number } | null = null;
let buildMs = 0;
let buildLabel: 'Build' | 'Load' = 'Build';
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
let errorVisible = false;

window.addEventListener('error', (event) => {
  showBuildError(event.error ?? event.message);
});

window.addEventListener('unhandledrejection', (event) => {
  showBuildError(event.reason);
});

await init();
validateWasmWrapper();
setStatus('WASM module loaded');
webgpuRenderer = await createWebGpuRenderer(gpuCanvas, showBuildError);
if (webgpuRenderer) {
  // Prefer WebGPU when the browser supports it; WebGL stays the HTML default so
  // the first paint before this async probe resolves always has a backend.
  rendererSelect.value = 'webgpu';
} else {
  // Without WebGPU the picker only offers WebGL, so hide it entirely rather than
  // showing a one-option dropdown.
  hideRendererControl();
}
rebuild();

regenerateButton.addEventListener('click', rebuild);
pointCountInput.addEventListener('change', rebuild);
dimensionSelect.addEventListener('change', () => {
  syncModeControls();
  rebuild();
});
geometrySelect.addEventListener('change', rebuild);
rendererSelect.addEventListener('change', () => {
  try {
    if (!index) {
      showBuildError(new Error('No index is available; the previous build failed or has not completed'));
      scheduleRender();
      return;
    }
    activeRenderer().uploadItems(items, scene());
    search();
  } catch (error) {
    showBuildError(error);
  }
});
nodeSizeSelect.addEventListener('change', rebuildIndex);
modeSelect.addEventListener('change', () => {
  syncModeControls();
  search();
});
resultModeSelect.addEventListener('change', search);
neighborCountInput.addEventListener('change', search);
maxDistanceInput.addEventListener('change', search);
depthInput.addEventListener('input', () => {
  syncDepthInputs('slider');
  search();
});
depthValueInput.addEventListener('change', () => {
  syncDepthInputs('number');
  search();
});
thicknessInput.addEventListener('input', () => {
  syncThicknessInputs('slider');
  search();
});
thicknessValueInput.addEventListener('change', () => {
  syncThicknessInputs('number');
  search();
});
distributionSelect.addEventListener('change', rebuild);
saveIndexButton.addEventListener('click', saveIndex);
loadIndexButton.addEventListener('click', () => loadIndexFileInput.click());
loadIndexFileInput.addEventListener('change', () => {
  const file = loadIndexFileInput.files?.[0];
  if (file) {
    loadIndex(file);
  }
  loadIndexFileInput.value = '';
});
reloadDemoButton.addEventListener('click', () => {
  window.location.reload();
});
for (const eventName of ['pointerdown', 'pointermove', 'pointerup', 'pointercancel', 'click']) {
  errorOverlayEl.addEventListener(eventName, (event) => {
    event.stopPropagation();
  });
}

depthLegendEl.addEventListener('pointerdown', (event) => {
  event.stopPropagation();
});
depthLegendEl.addEventListener('pointermove', (event) => {
  event.stopPropagation();
});
depthLegendEl.addEventListener('pointerup', (event) => {
  event.stopPropagation();
});

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
    if (!errorVisible) {
      setStatus('View resized; data unchanged');
    }
  }
}).observe(stage);
syncModeControls();
updateWorldView();

async function rebuild(): Promise<void> {
  const previous = snapshotScene();
  showLoader();
  setStatus('Generating…');
  try {
    // Yield a frame so the loader and status paint before the synchronous
    // generate+build blocks the main thread.
    await nextPaint();
    updateWorldView();
    rememberStageSize();
    syncZInputs();
    const count = normalizePointCount(Number(pointCountInput.value));
    pointCountInput.value = String(count);
    const distribution = distributionSelect.value as Distribution;

    const nextItems = generateDataset({
      count,
      geometry: currentGeometry(),
      distribution,
      dimension: currentDimension(),
      worldView,
      worldZSize: WORLD_Z_SIZE,
    });
    const nodeSize = Number(nodeSizeSelect.value);
    const started = performance.now();
    const nextIndex = buildCurrentIndex(nextItems, nodeSize);
    const nextBuildMs = performance.now() - started;

    items = nextItems;
    itemCount = count;
    index = nextIndex;
    buildMs = nextBuildMs;
    buildLabel = 'Build';
    query ??= defaultQuery();
    queryPoint ??= defaultQueryPoint();
    activeRenderer().uploadItems(items, scene());
    hasBuiltInitialDataset = true;
    setStatus('Dataset generated');
    search();
  } catch (error) {
    restoreScene(previous);
    restoreRenderedScene(previous);
    showBuildError(error);
  } finally {
    hideLoader();
  }
}

function showLoader(): void {
  loaderOverlayEl.classList.add('is-visible');
  loaderOverlayEl.setAttribute('aria-hidden', 'false');
}

function hideLoader(): void {
  loaderOverlayEl.classList.remove('is-visible');
  loaderOverlayEl.setAttribute('aria-hidden', 'true');
}

function nextPaint(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  });
}

function rebuildIndex(): void {
  try {
    const nodeSize = Number(nodeSizeSelect.value);

    const started = performance.now();
    const nextIndex = buildCurrentIndex(items, nodeSize);
    buildMs = performance.now() - started;
    buildLabel = 'Build';
    index = nextIndex;
    setStatus('Index rebuilt');
    query ??= defaultQuery();
    queryPoint ??= defaultQueryPoint();
    search();
  } catch (error) {
    showBuildError(error);
  }
}

type SceneSnapshot = {
  items: Float64Array<ArrayBufferLike>;
  itemCount: number;
  index: WasmIndex | null;
  hits: Uint32Array<ArrayBufferLike>;
  buildMs: number;
  buildLabel: 'Build' | 'Load';
  queryMs: number;
  coreMs: number;
  convertMs: number;
  copyMs: number;
  resultSummary: string | null;
  anyResult: boolean | null;
  hasBuiltInitialDataset: boolean;
};

function snapshotScene(): SceneSnapshot {
  return {
    items,
    itemCount,
    index,
    hits,
    buildMs,
    buildLabel,
    queryMs,
    coreMs,
    convertMs,
    copyMs,
    resultSummary,
    anyResult,
    hasBuiltInitialDataset,
  };
}

function restoreScene(snapshot: SceneSnapshot): void {
  items = snapshot.items;
  itemCount = snapshot.itemCount;
  index = snapshot.index;
  hits = snapshot.hits;
  buildMs = snapshot.buildMs;
  buildLabel = snapshot.buildLabel;
  queryMs = snapshot.queryMs;
  coreMs = snapshot.coreMs;
  convertMs = snapshot.convertMs;
  copyMs = snapshot.copyMs;
  resultSummary = snapshot.resultSummary;
  anyResult = snapshot.anyResult;
  hasBuiltInitialDataset = snapshot.hasBuiltInitialDataset;
}

function restoreRenderedScene(snapshot: SceneSnapshot): void {
  if (!snapshot.hasBuiltInitialDataset) {
    return;
  }
  try {
    const current = scene();
    activeRenderer().uploadItems(items, current);
    activeRenderer().uploadHits(hits);
  } catch (error) {
    console.warn('failed to restore previous rendered scene', error);
  }
}

function clearEmptyScene(): void {
  items = new Float64Array();
  itemCount = 0;
  index = null;
  hits = new Uint32Array();
  buildMs = 0;
  queryMs = 0;
  coreMs = 0;
  convertMs = 0;
  copyMs = 0;
  resultSummary = null;
  anyResult = null;
  hasBuiltInitialDataset = false;
}

function showBuildError(error: unknown): void {
  if (!hasBuiltInitialDataset) {
    clearEmptyScene();
  }
  setError(errorMessage(error));
  scheduleRender();
}

function setStatus(message: string): void {
  errorVisible = false;
  errorOverlayEl.classList.remove('is-visible');
  errorMessageEl.textContent = '';
  statusEl.textContent = statusText(message);
}

function setError(message: string): void {
  errorVisible = true;
  errorMessageEl.textContent = formatErrorForOverlay(message);
  errorOverlayEl.classList.add('is-visible');
  statusEl.textContent = statusText(`Build failed: ${message.split('\n')[0] ?? message}`);
}

function formatErrorForOverlay(message: string): string {
  if (message.includes('rust_oom') || message.includes('RuntimeError: unreachable')) {
    return `WASM ran out of memory while building the index. Reload the demo to reset the WASM instance.\n\n${message}`;
  }
  return message;
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.stack ?? error.message;
  }
  return String(error);
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

  activeRenderer().uploadHits(hits);
  scheduleRender();
}

function saveIndex(): void {
  if (!index) {
    return;
  }
  const bytes = index.to_bytes();
  const copy = new Uint8Array(bytes);
  const blob = new Blob([copy], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const distributionPrefix = distributionSelect.value === 'uniform' ? '' : `${distributionSelect.value}-`;
  a.download = `${distributionPrefix}${currentGeometry()}-${itemCount}-${currentDimension()}.psi`;
  a.click();
  URL.revokeObjectURL(url);
  setStatus('Index saved');
}

function loadIndex(file: File): void {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const buffer = reader.result as ArrayBuffer;
      const header = parsePsiHeader(new DataView(buffer));
      if (header.isF32) {
        throw new Error('This demo builds f64 indexes; f32 .psi files are not supported here');
      }

      const dimension: Dimension = header.is3d ? '3d' : '2d';
      const bytes = new Uint8Array(buffer);
      const started = performance.now();
      const nextIndex = header.is3d ? WasmIndex3D.from_bytes(bytes) : WasmIndex2D.from_bytes(bytes);
      const elapsed = performance.now() - started;
      const { items: nextItems, geometry } = reconstructItems(buffer, header);

      // Sync the UI to the loaded index so rendering and stats match it.
      dimensionSelect.value = dimension;
      geometrySelect.value = geometry;
      if (Array.from(nodeSizeSelect.options).some((option) => Number(option.value) === header.nodeSize)) {
        nodeSizeSelect.value = String(header.nodeSize);
      }
      pointCountInput.value = String(header.numItems);

      items = nextItems;
      itemCount = header.numItems;
      index = nextIndex;
      buildMs = elapsed;
      buildLabel = 'Load';
      hasBuiltInitialDataset = true;
      query ??= defaultQuery();
      queryPoint ??= defaultQueryPoint();
      syncModeControls();
      updateWorldView();
      activeRenderer().uploadItems(items, scene());
      setStatus('Index loaded');
      search();
    } catch (error) {
      showBuildError(error);
    }
  };
  reader.readAsArrayBuffer(file);
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
  stage.classList.toggle('is-3d', currentDimension() === '3d');

  const gpu = usingGpu();
  glCanvas.style.display = gpu ? 'none' : 'block';
  gpuCanvas.style.display = gpu ? 'block' : 'none';
  activeRenderer().render(scene());
  overlays.render({
    mode: currentMode(),
    resultMode: currentResultMode(),
    dimension: currentDimension(),
    geometry: currentGeometry(),
    query,
    queryPoint,
    anyResult,
    hits,
    items,
    worldView,
  });

  buildLabelEl.textContent = buildLabel;
  buildTimeEl.textContent = formatDuration(buildMs);
  queryTimeEl.textContent = formatQueryDuration(queryMs);
  coreTimeEl.textContent = formatQueryDuration(coreMs);
  convertTimeEl.textContent = formatQueryDuration(convertMs);
  copyTimeEl.textContent = formatQueryDuration(copyMs);
  resultLabelEl.textContent = resultLabel();
  hitCountEl.textContent = resultSummary ?? hits.length.toLocaleString();
  pointTotalEl.textContent = itemCount.toLocaleString();
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

function currentRendererMode(): RendererMode {
  return rendererSelect.value as RendererMode;
}

function usingGpu(): boolean {
  return currentRendererMode() === 'webgpu' && webgpuRenderer !== null;
}

function activeRenderer(): Renderer {
  return usingGpu() ? (webgpuRenderer as Renderer) : webglRenderer;
}

function scene(): Scene {
  const dimension = currentDimension();
  return {
    geometry: currentGeometry(),
    dimension,
    itemCount,
    itemStride: itemStride(),
    worldView,
    pointSize: pointSizeForCount(itemCount),
    hitSize: hitSizeForCount(itemCount),
    colors: SCENE_COLORS,
    zValueForPoint: (data, offset) => zValueForPoint(data, offset, dimension, WORLD_Z_SIZE),
    zValueForBox: (data, offset) => zValueForBox(data, offset, dimension, WORLD_Z_SIZE),
  };
}

function hideRendererControl(): void {
  // WebGL is the only remaining backend, so force it and hide the picker.
  rendererSelect.value = 'webgl';
  const control = rendererSelect.closest('label');
  if (control) {
    control.hidden = true;
  }
}

function syncModeControls(): void {
  const nearest = currentMode() === 'nearest';
  const is3d = currentDimension() === '3d';
  neighborCountInput.disabled = !nearest;
  maxDistanceInput.disabled = !nearest;
  resultModeSelect.disabled = nearest;
  thicknessInput.disabled = !is3d || nearest;
  thicknessValueInput.disabled = !is3d || nearest;
}

function syncZInputs(): void {
  const center = normalizeZ(Number(depthValueInput.value), WORLD_Z_SIZE * 0.5);
  const thickness = normalizeThickness(thicknessValueInput.value);
  const half = thickness * 0.5;
  depthSlice = {
    min: center - half,
    max: center + half,
    center,
    thickness,
  };
  const roundedCenter = String(Math.round(depthSlice.center));
  depthInput.value = String(Math.round(clampCoordinate(depthSlice.center, 0, WORLD_Z_SIZE)));
  depthValueInput.value = roundedCenter;
  const roundedThickness = String(Math.round(depthSlice.thickness));
  thicknessInput.value = String(Math.round(clampCoordinate(depthSlice.thickness, 0, WORLD_Z_SIZE)));
  thicknessValueInput.value = roundedThickness;
}

function syncDepthInputs(source: 'slider' | 'number'): void {
  const sourceValue = source === 'slider' ? depthInput.value : depthValueInput.value;
  depthValueInput.value = sourceValue;
  syncZInputs();
}

function syncThicknessInputs(source: 'slider' | 'number'): void {
  const sourceValue = source === 'slider' ? thicknessInput.value : thicknessValueInput.value;
  thicknessValueInput.value = sourceValue;
  syncZInputs();
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

function normalizeThickness(rawValue: string): number {
  const value = rawValue.trim() === '' ? Number.NaN : Number(rawValue);
  if (!Number.isFinite(value)) {
    return 1000;
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
  const thickness = 1000;
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

