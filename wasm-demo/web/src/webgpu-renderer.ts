import type { Renderer, Rgba, Scene } from './rendering';

type GpuState = {
  device: GPUDevice;
  context: GPUCanvasContext;
  canvas: HTMLCanvasElement;
  bindGroupLayout: GPUBindGroupLayout;
  boxPipeline: GPURenderPipeline;
  pointPipeline: GPURenderPipeline;
  uniformBase: GPUBuffer;
  uniformHit: GPUBuffer;
  geomBuffer: GPUBuffer;
  depthBuffer: GPUBuffer;
  hitBuffer: GPUBuffer;
  geomCapacity: number;
  depthCapacity: number;
  hitCapacity: number;
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  bindGroupBase: GPUBindGroup;
  bindGroupHit: GPUBindGroup;
  itemCount: number;
  hitCount: number;
};

const GPU_UNIFORM_BYTES = 48;
const GPU_SHADER_STAGE_VERTEX = 0x1;
const GPU_BUFFER_USAGE_COPY_DST = 0x8;
const GPU_BUFFER_USAGE_UNIFORM = 0x40;
const GPU_BUFFER_USAGE_STORAGE = 0x80;
const DEFAULT_MAX_BUFFER_SIZE = 256 * 1024 * 1024;
const DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE = 128 * 1024 * 1024;
const GPU_UPLOAD_CHUNK_ITEMS = 1_000_000;

export async function createWebGpuRenderer(
  canvas: HTMLCanvasElement,
  onError?: (error: unknown) => void,
): Promise<Renderer | null> {
  if (!navigator.gpu) {
    return null;
  }

  let device: GPUDevice;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return null;
    }
    const maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
    const maxBufferSize = adapter.limits.maxBufferSize;
    device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize,
        maxStorageBufferBindingSize,
      },
    });
  } catch (error) {
    console.warn('WebGPU initialization failed', error);
    return null;
  }

  const context = canvas.getContext('webgpu') as GPUCanvasContext | null;
  if (!context) {
    return null;
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const module = device.createShaderModule({ code: shaderSource() });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPU_SHADER_STAGE_VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPU_SHADER_STAGE_VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPU_SHADER_STAGE_VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPU_SHADER_STAGE_VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const target: GPUColorTargetState = {
    format,
    blend: {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    },
  };
  const boxPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module, entryPoint: 'vsBox' },
    fragment: { module, entryPoint: 'fsSolid', targets: [target] },
    primitive: { topology: 'triangle-list' },
  });
  const pointPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module, entryPoint: 'vsPoint' },
    fragment: { module, entryPoint: 'fsPoint', targets: [target] },
    primitive: { topology: 'triangle-list' },
  });

  const uniformUsage = GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST;
  const storageUsage = GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST;
  const state: GpuState = {
    device,
    context,
    canvas,
    bindGroupLayout,
    boxPipeline,
    pointPipeline,
    uniformBase: device.createBuffer({ size: GPU_UNIFORM_BYTES, usage: uniformUsage }),
    uniformHit: device.createBuffer({ size: GPU_UNIFORM_BYTES, usage: uniformUsage }),
    geomBuffer: device.createBuffer({ size: 16, usage: storageUsage }),
    depthBuffer: device.createBuffer({ size: 4, usage: storageUsage }),
    hitBuffer: device.createBuffer({ size: 4, usage: storageUsage }),
    geomCapacity: 1,
    depthCapacity: 1,
    hitCapacity: 1,
    maxBufferSize: device.limits.maxBufferSize ?? DEFAULT_MAX_BUFFER_SIZE,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize ?? DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE,
    bindGroupBase: null as unknown as GPUBindGroup,
    bindGroupHit: null as unknown as GPUBindGroup,
    itemCount: 0,
    hitCount: 0,
  };
  rebuildBindGroups(state);

  if (onError) {
    device.addEventListener('uncapturederror', (event) => {
      onError((event as GPUUncapturedErrorEvent).error);
    });
    device.lost.then((info) => {
      onError(new Error(`WebGPU device lost: ${info.reason} ${info.message}`.trim()));
    });
  }

  return {
    canvas,
    uploadItems: (data, scene) => uploadGpuItems(state, data, scene),
    uploadHits: (hitIndices) => uploadGpuHits(state, hitIndices),
    render: (scene) => renderGpuScene(state, scene),
  };
}

function uploadGpuItems(renderer: GpuState, data: Float64Array<ArrayBufferLike>, scene: Scene): void {
  const count = scene.itemCount;
  const slots = Math.max(1, count);

  const grownGeom = growBuffer(renderer, renderer.geomBuffer, renderer.geomCapacity, slots, 16, 'geometry');
  renderer.geomBuffer = grownGeom.buffer;
  renderer.geomCapacity = grownGeom.capacity;

  const grownDepth = growBuffer(renderer, renderer.depthBuffer, renderer.depthCapacity, slots, 4, 'depth');
  renderer.depthBuffer = grownDepth.buffer;
  renderer.depthCapacity = grownDepth.capacity;

  renderer.itemCount = count;
  rebuildBindGroups(renderer);
  writeItems(renderer, data, scene);
}

function uploadGpuHits(renderer: GpuState, hitIndices: Uint32Array<ArrayBufferLike>): void {
  const count = hitIndices.length;
  const grown = growBuffer(renderer, renderer.hitBuffer, renderer.hitCapacity, Math.max(1, count), 4, 'hit index');
  if (grown.changed) {
    renderer.hitBuffer = grown.buffer;
    renderer.hitCapacity = grown.capacity;
  }
  renderer.hitCount = count;
  rebuildBindGroups(renderer);
  if (count > 0) {
    renderer.device.queue.writeBuffer(renderer.hitBuffer, 0, hitIndices);
  }
}

function renderGpuScene(renderer: GpuState, scene: Scene): void {
  resizeCanvasToDisplaySize(renderer.canvas);

  const view = renderer.context.getCurrentTexture().createView();
  const encoder = renderer.device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view,
        clearValue: {
          r: scene.colors.background[0],
          g: scene.colors.background[1],
          b: scene.colors.background[2],
          a: scene.colors.background[3],
        },
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  });

  const is3d = scene.dimension === '3d';
  const dpr = window.devicePixelRatio || 1;
  if (scene.geometry === 'boxes') {
    writeUniform(renderer, renderer.uniformBase, scene, scene.colors.box, 0, is3d, false);
    if (renderer.itemCount > 0) {
      pass.setPipeline(renderer.boxPipeline);
      pass.setBindGroup(0, renderer.bindGroupBase);
      pass.draw(6, renderer.itemCount);
    }
    writeUniform(renderer, renderer.uniformHit, scene, scene.colors.hitBox, 0, false, true);
    if (renderer.hitCount > 0) {
      pass.setPipeline(renderer.boxPipeline);
      pass.setBindGroup(0, renderer.bindGroupHit);
      pass.draw(6, renderer.hitCount);
    }
  } else {
    writeUniform(renderer, renderer.uniformBase, scene, scene.colors.point, scene.pointSize * dpr, is3d, false);
    if (renderer.itemCount > 0) {
      pass.setPipeline(renderer.pointPipeline);
      pass.setBindGroup(0, renderer.bindGroupBase);
      pass.draw(6, renderer.itemCount);
    }
    writeUniform(renderer, renderer.uniformHit, scene, scene.colors.hit, scene.hitSize * dpr, false, true);
    if (renderer.hitCount > 0) {
      pass.setPipeline(renderer.pointPipeline);
      pass.setBindGroup(0, renderer.bindGroupHit);
      pass.draw(6, renderer.hitCount);
    }
  }

  pass.end();
  renderer.device.queue.submit([encoder.finish()]);
}

function rebuildBindGroups(renderer: GpuState): void {
  const geomBindingSize = storageBindingSize(renderer.itemCount, 16, renderer.maxStorageBufferBindingSize, 'geometry');
  const depthBindingSize = storageBindingSize(renderer.itemCount, 4, renderer.maxStorageBufferBindingSize, 'depth');
  const hitBindingSize = storageBindingSize(renderer.hitCount, 4, renderer.maxStorageBufferBindingSize, 'hit index');
  const storage = [
    { binding: 1, resource: { buffer: renderer.geomBuffer, size: geomBindingSize } },
    { binding: 2, resource: { buffer: renderer.depthBuffer, size: depthBindingSize } },
    { binding: 3, resource: { buffer: renderer.hitBuffer, size: hitBindingSize } },
  ];
  renderer.bindGroupBase = renderer.device.createBindGroup({
    layout: renderer.bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: renderer.uniformBase } }, ...storage],
  });
  renderer.bindGroupHit = renderer.device.createBindGroup({
    layout: renderer.bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: renderer.uniformHit } }, ...storage],
  });
}

function storageBindingSize(
  count: number,
  bytesPerElement: number,
  maxStorageBufferBindingSize: number,
  label: string,
): number {
  const size = Math.max(1, count) * bytesPerElement;
  if (size > maxStorageBufferBindingSize) {
    throw new Error(`${label} buffer needs ${size} bytes, but this WebGPU adapter allows ${maxStorageBufferBindingSize}`);
  }
  return size;
}

function growBuffer(
  renderer: GpuState,
  current: GPUBuffer,
  capacity: number,
  needed: number,
  bytesPerElement: number,
  label: string,
): { buffer: GPUBuffer; capacity: number; changed: boolean } {
  if (needed <= capacity) {
    return { buffer: current, capacity, changed: false };
  }
  current.destroy();
  const nextCapacity = Math.max(needed, capacity * 2);
  const size = nextCapacity * bytesPerElement;
  if (size > renderer.maxBufferSize) {
    throw new Error(`${label} buffer needs ${size} bytes, but this WebGPU adapter allows ${renderer.maxBufferSize}`);
  }
  const buffer = renderer.device.createBuffer({
    size,
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  return { buffer, capacity: nextCapacity, changed: true };
}

function writeItems(renderer: GpuState, data: Float64Array<ArrayBufferLike>, scene: Scene): void {
  const maxXOffset = scene.dimension === '3d' ? 3 : 2;
  const maxYOffset = scene.dimension === '3d' ? 4 : 3;
  for (let start = 0; start < scene.itemCount; start += GPU_UPLOAD_CHUNK_ITEMS) {
    const count = Math.min(GPU_UPLOAD_CHUNK_ITEMS, scene.itemCount - start);
    const geom = new Float32Array(count * 4);
    const depth = new Float32Array(count);
    if (scene.geometry === 'boxes') {
      for (let i = 0; i < count; i++) {
        const src = (start + i) * scene.itemStride;
        const dst = i * 4;
        geom[dst] = data[src];
        geom[dst + 1] = data[src + 1];
        geom[dst + 2] = data[src + maxXOffset];
        geom[dst + 3] = data[src + maxYOffset];
        depth[i] = scene.zValueForBox(data, src);
      }
    } else {
      for (let i = 0; i < count; i++) {
        const src = (start + i) * scene.itemStride;
        const dst = i * 4;
        geom[dst] = data[src];
        geom[dst + 1] = data[src + 1];
        depth[i] = scene.zValueForPoint(data, src);
      }
    }
    renderer.device.queue.writeBuffer(renderer.geomBuffer, start * 16, geom);
    renderer.device.queue.writeBuffer(renderer.depthBuffer, start * 4, depth);
  }
}

function writeUniform(
  renderer: GpuState,
  buffer: GPUBuffer,
  scene: Scene,
  color: Rgba,
  pointSize: number,
  useDepthColor: boolean,
  useIndices: boolean,
): void {
  const bytes = new ArrayBuffer(GPU_UNIFORM_BYTES);
  const floats = new Float32Array(bytes);
  const uints = new Uint32Array(bytes);
  floats[0] = scene.worldView.width;
  floats[1] = scene.worldView.height;
  floats[2] = renderer.canvas.width;
  floats[3] = renderer.canvas.height;
  floats[4] = color[0];
  floats[5] = color[1];
  floats[6] = color[2];
  floats[7] = color[3];
  floats[8] = pointSize;
  uints[9] = useDepthColor ? 1 : 0;
  uints[10] = useIndices ? 1 : 0;
  uints[11] = 0;
  renderer.device.queue.writeBuffer(buffer, 0, bytes);
}

function resizeCanvasToDisplaySize(canvas: HTMLCanvasElement): void {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function shaderSource(): string {
  return /* wgsl */ `
struct Uniforms {
  worldSize : vec2f,
  canvasSize : vec2f,
  color : vec4f,
  pointSize : f32,
  useDepthColor : u32,
  useIndices : u32,
  pad : u32,
};

@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var<storage, read> geom : array<vec4f>;
@group(0) @binding(2) var<storage, read> itemDepth : array<f32>;
@group(0) @binding(3) var<storage, read> hits : array<u32>;

struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0) color : vec4f,
  @location(1) uv : vec2f,
};

fn corner(vi : u32) -> vec2f {
  var c = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
  );
  return c[vi];
}

fn itemIndex(ii : u32) -> u32 {
  if (u.useIndices == 1u) {
    return hits[ii];
  }
  return ii;
}

fn depthGradient(t : f32) -> vec3f {
  let nearColor = vec3f(0.18, 0.45, 0.82);
  let midLow = vec3f(0.18, 0.66, 0.92);
  let midHigh = vec3f(0.48, 0.92, 0.82);
  let farColor = vec3f(1.0, 0.88, 0.32);
  if (t < 0.33) {
    return mix(nearColor, midLow, t / 0.33);
  }
  if (t < 0.66) {
    return mix(midLow, midHigh, (t - 0.33) / 0.33);
  }
  return mix(midHigh, farColor, (t - 0.66) / 0.34);
}

fn shade(item : u32) -> vec4f {
  if (u.useDepthColor == 1u) {
    return vec4f(depthGradient(itemDepth[item]), u.color.a);
  }
  return u.color;
}

fn toClip(world : vec2f) -> vec2f {
  return vec2f((world.x / u.worldSize.x) * 2.0 - 1.0, 1.0 - (world.y / u.worldSize.y) * 2.0);
}

@vertex
fn vsBox(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VSOut {
  let item = itemIndex(ii);
  let box = geom[item];
  let world = mix(box.xy, box.zw, corner(vi));
  var out : VSOut;
  out.pos = vec4f(toClip(world), 0.0, 1.0);
  out.color = shade(item);
  out.uv = vec2f(0.0, 0.0);
  return out;
}

@vertex
fn vsPoint(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VSOut {
  let item = itemIndex(ii);
  let c = corner(vi);
  let clip = toClip(geom[item].xy);
  let offset = (c - vec2f(0.5)) * u.pointSize * 2.0 / u.canvasSize;
  var out : VSOut;
  out.pos = vec4f(clip.x + offset.x, clip.y + offset.y, 0.0, 1.0);
  out.color = shade(item);
  out.uv = c;
  return out;
}

@fragment
fn fsSolid(in : VSOut) -> @location(0) vec4f {
  return in.color;
}

@fragment
fn fsPoint(in : VSOut) -> @location(0) vec4f {
  let d = in.uv - vec2f(0.5);
  if (dot(d, d) > 0.25) {
    discard;
  }
  return in.color;
}
`;
}
