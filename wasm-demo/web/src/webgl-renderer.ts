import type { Renderer, Rgba, Scene } from './rendering';

type WebGlState = {
  gl: WebGL2RenderingContext;
  pointProgram: WebGLProgram;
  boxProgram: WebGLProgram;
  indexedBoxProgram: WebGLProgram;
  indexedPointProgram: WebGLProgram;
  positionLocation: number;
  colorLocation: WebGLUniformLocation;
  useDepthColorLocation: WebGLUniformLocation;
  pointSizeLocation: WebGLUniformLocation;
  worldSizeLocation: WebGLUniformLocation;
  boxCornerLocation: number;
  boxLocation: number;
  boxDepthLocation: number;
  boxColorLocation: WebGLUniformLocation;
  boxUseDepthColorLocation: WebGLUniformLocation;
  boxWorldSizeLocation: WebGLUniformLocation;
  indexedBoxCornerLocation: number;
  indexedBoxColorLocation: WebGLUniformLocation;
  indexedBoxWorldSizeLocation: WebGLUniformLocation;
  indexedBoxTextureLocation: WebGLUniformLocation;
  indexedBoxHitTextureLocation: WebGLUniformLocation;
  indexedBoxTextureWidthLocation: WebGLUniformLocation;
  indexedBoxHitTextureWidthLocation: WebGLUniformLocation;
  indexedPointColorLocation: WebGLUniformLocation;
  indexedPointWorldSizeLocation: WebGLUniformLocation;
  indexedPointSizeLocation: WebGLUniformLocation;
  indexedPointTextureLocation: WebGLUniformLocation;
  indexedPointHitTextureLocation: WebGLUniformLocation;
  indexedPointTextureWidthLocation: WebGLUniformLocation;
  indexedPointHitTextureWidthLocation: WebGLUniformLocation;
  unitQuadBuffer: WebGLBuffer;
  pointBuffer: WebGLBuffer;
  boxBuffer: WebGLBuffer;
  itemTexture: WebGLTexture;
  hitIndexTexture: WebGLTexture;
  itemTextureWidth: number;
  hitTextureWidth: number;
  maxTextureSize: number;
  itemCount: number;
  hitCount: number;
};

export function createWebGlRenderer(canvas: HTMLCanvasElement): Renderer {
  const state = createState(canvas);
  return {
    canvas,
    uploadItems: (data, scene) => uploadItems(state, data, scene),
    uploadHits: (hitIndices) => uploadHits(state, hitIndices),
    render: (scene) => render(state, canvas, scene),
  };
}

function render(state: WebGlState, canvas: HTMLCanvasElement, scene: Scene): void {
  resizeCanvasToDisplaySize(canvas);

  const { gl } = state;
  const { background } = scene.colors;
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(background[0], background[1], background[2], background[3]);
  gl.clear(gl.COLOR_BUFFER_BIT);

  const is3d = scene.dimension === '3d';
  if (scene.geometry === 'boxes') {
    drawBoxes(state, state.itemCount, scene.colors.box, scene.worldView, is3d);
    drawIndexedBoxes(state, state.hitCount, scene.colors.hitBox, scene.worldView);
  } else {
    drawPoints(state, state.itemCount, scene.colors.point, scene.pointSize, scene.worldView, is3d);
    drawIndexedPoints(state, state.hitCount, scene.colors.hit, scene.hitSize, scene.worldView);
  }
}

function drawPoints(
  state: WebGlState,
  count: number,
  color: Rgba,
  pointSize: number,
  worldView: { width: number; height: number },
  useDepthColor: boolean,
): void {
  if (count === 0) {
    return;
  }

  const { gl } = state;
  gl.useProgram(state.pointProgram);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.pointBuffer);
  gl.enableVertexAttribArray(state.positionLocation);
  gl.vertexAttribPointer(state.positionLocation, 3, gl.FLOAT, false, 0, 0);
  gl.uniform4f(state.colorLocation, color[0], color[1], color[2], color[3]);
  gl.uniform1i(state.useDepthColorLocation, useDepthColor ? 1 : 0);
  gl.uniform1f(state.pointSizeLocation, pointSize * (window.devicePixelRatio || 1));
  gl.uniform2f(state.worldSizeLocation, worldView.width, worldView.height);
  gl.drawArrays(gl.POINTS, 0, count);
}

function drawBoxes(
  state: WebGlState,
  count: number,
  color: Rgba,
  worldView: { width: number; height: number },
  useDepthColor: boolean,
): void {
  if (count === 0) {
    return;
  }

  const { gl } = state;
  gl.useProgram(state.boxProgram);

  gl.bindBuffer(gl.ARRAY_BUFFER, state.unitQuadBuffer);
  gl.enableVertexAttribArray(state.boxCornerLocation);
  gl.vertexAttribPointer(state.boxCornerLocation, 2, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(state.boxCornerLocation, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, state.boxBuffer);
  gl.enableVertexAttribArray(state.boxLocation);
  gl.vertexAttribPointer(state.boxLocation, 4, gl.FLOAT, false, 20, 0);
  gl.vertexAttribDivisor(state.boxLocation, 1);
  gl.enableVertexAttribArray(state.boxDepthLocation);
  gl.vertexAttribPointer(state.boxDepthLocation, 1, gl.FLOAT, false, 20, 16);
  gl.vertexAttribDivisor(state.boxDepthLocation, 1);

  gl.uniform4f(state.boxColorLocation, color[0], color[1], color[2], color[3]);
  gl.uniform1i(state.boxUseDepthColorLocation, useDepthColor ? 1 : 0);
  gl.uniform2f(state.boxWorldSizeLocation, worldView.width, worldView.height);
  gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, count);
  gl.vertexAttribDivisor(state.boxLocation, 0);
  gl.vertexAttribDivisor(state.boxDepthLocation, 0);
}

function drawIndexedBoxes(
  state: WebGlState,
  count: number,
  color: Rgba,
  worldView: { width: number; height: number },
): void {
  if (count === 0) {
    return;
  }

  const { gl } = state;
  gl.useProgram(state.indexedBoxProgram);

  gl.bindBuffer(gl.ARRAY_BUFFER, state.unitQuadBuffer);
  gl.enableVertexAttribArray(state.indexedBoxCornerLocation);
  gl.vertexAttribPointer(state.indexedBoxCornerLocation, 2, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(state.indexedBoxCornerLocation, 0);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, state.itemTexture);
  gl.uniform1i(state.indexedBoxTextureLocation, 0);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, state.hitIndexTexture);
  gl.uniform1i(state.indexedBoxHitTextureLocation, 1);

  gl.uniform1i(state.indexedBoxTextureWidthLocation, state.itemTextureWidth);
  gl.uniform1i(state.indexedBoxHitTextureWidthLocation, state.hitTextureWidth);
  gl.uniform4f(state.indexedBoxColorLocation, color[0], color[1], color[2], color[3]);
  gl.uniform2f(state.indexedBoxWorldSizeLocation, worldView.width, worldView.height);
  gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, count);
}

function drawIndexedPoints(
  state: WebGlState,
  count: number,
  color: Rgba,
  pointSize: number,
  worldView: { width: number; height: number },
): void {
  if (count === 0) {
    return;
  }

  const { gl } = state;
  gl.useProgram(state.indexedPointProgram);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, state.itemTexture);
  gl.uniform1i(state.indexedPointTextureLocation, 0);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, state.hitIndexTexture);
  gl.uniform1i(state.indexedPointHitTextureLocation, 1);

  gl.uniform1i(state.indexedPointTextureWidthLocation, state.itemTextureWidth);
  gl.uniform1i(state.indexedPointHitTextureWidthLocation, state.hitTextureWidth);
  gl.uniform4f(state.indexedPointColorLocation, color[0], color[1], color[2], color[3]);
  gl.uniform2f(state.indexedPointWorldSizeLocation, worldView.width, worldView.height);
  gl.uniform1f(state.indexedPointSizeLocation, pointSize * (window.devicePixelRatio || 1));
  gl.drawArrays(gl.POINTS, 0, count);
}

function uploadItems(state: WebGlState, data: Float64Array<ArrayBufferLike>, scene: Scene): void {
  state.itemCount = scene.itemCount;
  if (scene.geometry === 'boxes') {
    uploadBoxes(state, data, scene);
  } else {
    uploadPoints(state, data, scene);
  }
  uploadItemTexture(state, data, scene);
}

function uploadPoints(state: WebGlState, data: Float64Array<ArrayBufferLike>, scene: Scene): void {
  const { gl } = state;
  gl.bindBuffer(gl.ARRAY_BUFFER, state.pointBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, toFloat32Points(data, scene), gl.STATIC_DRAW);
}

function uploadBoxes(state: WebGlState, data: Float64Array<ArrayBufferLike>, scene: Scene): void {
  const { gl } = state;
  gl.bindBuffer(gl.ARRAY_BUFFER, state.boxBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, toFloat32Boxes(data, scene), gl.STATIC_DRAW);
}

function uploadItemTexture(state: WebGlState, data: Float64Array<ArrayBufferLike>, scene: Scene): void {
  const { itemCount, itemStride } = scene;
  const { width, height } = textureSizeForCount(state, itemCount);
  const texels = new Float32Array(width * height * 4);
  if (scene.geometry === 'boxes') {
    const maxXOffset = scene.dimension === '3d' ? 3 : 2;
    const maxYOffset = scene.dimension === '3d' ? 4 : 3;
    for (let i = 0; i < itemCount; i++) {
      const src = i * itemStride;
      const dst = i * 4;
      texels[dst] = data[src];
      texels[dst + 1] = data[src + 1];
      texels[dst + 2] = data[src + maxXOffset];
      texels[dst + 3] = data[src + maxYOffset];
    }
  } else {
    for (let i = 0; i < itemCount; i++) {
      const src = i * itemStride;
      const dst = i * 4;
      texels[dst] = data[src];
      texels[dst + 1] = data[src + 1];
    }
  }

  const { gl } = state;
  state.itemTextureWidth = width;
  gl.bindTexture(gl.TEXTURE_2D, state.itemTexture);
  setDataTextureParameters(gl);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, texels);
}

function uploadHits(state: WebGlState, hitIndices: Uint32Array<ArrayBufferLike>): void {
  state.hitCount = hitIndices.length;
  const { width, height } = textureSizeForCount(state, hitIndices.length);
  const texels = new Uint32Array(width * height);
  texels.set(hitIndices);

  const { gl } = state;
  state.hitTextureWidth = width;
  gl.bindTexture(gl.TEXTURE_2D, state.hitIndexTexture);
  setDataTextureParameters(gl);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32UI, width, height, 0, gl.RED_INTEGER, gl.UNSIGNED_INT, texels);
}

function textureSizeForCount(state: WebGlState, count: number): { width: number; height: number } {
  const width = Math.max(1, Math.min(state.maxTextureSize, Math.ceil(Math.sqrt(Math.max(1, count)))));
  return {
    width,
    height: Math.max(1, Math.ceil(count / width)),
  };
}

function setDataTextureParameters(gl: WebGL2RenderingContext): void {
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

function toFloat32Points(data: Float64Array<ArrayBufferLike>, scene: Scene): Float32Array {
  const { itemCount, itemStride } = scene;
  const out = new Float32Array(itemCount * 3);
  for (let i = 0; i < itemCount; i++) {
    const offset = i * itemStride;
    const outOffset = i * 3;
    out[outOffset] = data[offset];
    out[outOffset + 1] = data[offset + 1];
    out[outOffset + 2] = scene.zValueForPoint(data, offset);
  }
  return out;
}

function toFloat32Boxes(data: Float64Array<ArrayBufferLike>, scene: Scene): Float32Array {
  const { itemCount, itemStride } = scene;
  const out = new Float32Array(itemCount * 5);
  for (let i = 0; i < itemCount; i++) {
    const offset = i * itemStride;
    const maxOffset = scene.dimension === '3d' ? offset + 3 : offset + 2;
    const outOffset = i * 5;
    out[outOffset] = data[offset];
    out[outOffset + 1] = data[offset + 1];
    out[outOffset + 2] = data[maxOffset];
    out[outOffset + 3] = data[maxOffset + 1];
    out[outOffset + 4] = scene.zValueForBox(data, offset);
  }
  return out;
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

function createState(canvas: HTMLCanvasElement): WebGlState {
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
    in vec3 a_position;
    uniform float u_pointSize;
    uniform vec2 u_worldSize;
    out float v_depth;

    void main() {
      vec2 clip = vec2(
        (a_position.x / u_worldSize.x) * 2.0 - 1.0,
        1.0 - (a_position.y / u_worldSize.y) * 2.0
      );
      gl_Position = vec4(clip, 0.0, 1.0);
      gl_PointSize = u_pointSize;
      v_depth = a_position.z;
    }`,
    `#version 300 es
    precision highp float;

    uniform vec4 u_color;
    uniform bool u_useDepthColor;
    in float v_depth;
    out vec4 outColor;

    vec3 depthColor(float t) {
      vec3 nearColor = vec3(0.18, 0.45, 0.82);
      vec3 midLowColor = vec3(0.18, 0.66, 0.92);
      vec3 midHighColor = vec3(0.48, 0.92, 0.82);
      vec3 farColor = vec3(1.0, 0.88, 0.32);
      if (t < 0.33) {
        return mix(nearColor, midLowColor, t / 0.33);
      }
      if (t < 0.66) {
        return mix(midLowColor, midHighColor, (t - 0.33) / 0.33);
      }
      return mix(midHighColor, farColor, (t - 0.66) / 0.34);
    }

    void main() {
      vec2 delta = gl_PointCoord - vec2(0.5);
      if (dot(delta, delta) > 0.25) {
        discard;
      }
      vec4 color = u_useDepthColor ? vec4(depthColor(v_depth), u_color.a) : u_color;
      outColor = color;
    }`,
  );

  const boxProgram = createProgram(
    gl,
    `#version 300 es
    in vec2 a_corner;
    in vec4 a_box;
    in float a_depth;
    uniform vec2 u_worldSize;
    out float v_depth;

    void main() {
      vec2 world = mix(a_box.xy, a_box.zw, a_corner);
      vec2 clip = vec2(
        (world.x / u_worldSize.x) * 2.0 - 1.0,
        1.0 - (world.y / u_worldSize.y) * 2.0
      );
      gl_Position = vec4(clip, 0.0, 1.0);
      v_depth = a_depth;
    }`,
    `#version 300 es
    precision highp float;

    uniform vec4 u_color;
    uniform bool u_useDepthColor;
    in float v_depth;
    out vec4 outColor;

    vec3 depthColor(float t) {
      vec3 nearColor = vec3(0.18, 0.45, 0.82);
      vec3 midLowColor = vec3(0.18, 0.66, 0.92);
      vec3 midHighColor = vec3(0.48, 0.92, 0.82);
      vec3 farColor = vec3(1.0, 0.88, 0.32);
      if (t < 0.33) {
        return mix(nearColor, midLowColor, t / 0.33);
      }
      if (t < 0.66) {
        return mix(midLowColor, midHighColor, (t - 0.33) / 0.33);
      }
      return mix(midHighColor, farColor, (t - 0.66) / 0.34);
    }

    void main() {
      outColor = u_useDepthColor ? vec4(depthColor(v_depth), u_color.a) : u_color;
    }`,
  );

  const indexedBoxProgram = createProgram(
    gl,
    `#version 300 es
    precision highp float;
    precision highp usampler2D;

    in vec2 a_corner;
    uniform sampler2D u_boxTexture;
    uniform usampler2D u_hitTexture;
    uniform int u_boxTextureWidth;
    uniform int u_hitTextureWidth;
    uniform vec2 u_worldSize;

    vec4 fetchBox(uint index) {
      int i = int(index);
      return texelFetch(u_boxTexture, ivec2(i % u_boxTextureWidth, i / u_boxTextureWidth), 0);
    }

    uint fetchHit(int instanceIndex) {
      return texelFetch(u_hitTexture, ivec2(instanceIndex % u_hitTextureWidth, instanceIndex / u_hitTextureWidth), 0).r;
    }

    void main() {
      vec4 box = fetchBox(fetchHit(gl_InstanceID));
      vec2 world = mix(box.xy, box.zw, a_corner);
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

  const indexedPointProgram = createProgram(
    gl,
    `#version 300 es
    precision highp float;
    precision highp usampler2D;

    uniform sampler2D u_itemTexture;
    uniform usampler2D u_hitTexture;
    uniform int u_itemTextureWidth;
    uniform int u_hitTextureWidth;
    uniform vec2 u_worldSize;
    uniform float u_pointSize;

    vec2 fetchPoint(uint index) {
      int i = int(index);
      return texelFetch(u_itemTexture, ivec2(i % u_itemTextureWidth, i / u_itemTextureWidth), 0).xy;
    }

    uint fetchHit(int instanceIndex) {
      return texelFetch(u_hitTexture, ivec2(instanceIndex % u_hitTextureWidth, instanceIndex / u_hitTextureWidth), 0).r;
    }

    void main() {
      vec2 world = fetchPoint(fetchHit(gl_VertexID));
      vec2 clip = vec2(
        (world.x / u_worldSize.x) * 2.0 - 1.0,
        1.0 - (world.y / u_worldSize.y) * 2.0
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

  const unitQuadBuffer = gl.createBuffer();
  const pointBuffer = gl.createBuffer();
  const boxBuffer = gl.createBuffer();
  const itemTexture = gl.createTexture();
  const hitIndexTexture = gl.createTexture();
  if (!unitQuadBuffer || !pointBuffer || !boxBuffer || !itemTexture || !hitIndexTexture) {
    throw new Error('failed to create WebGL buffers');
  }

  gl.bindBuffer(gl.ARRAY_BUFFER, unitQuadBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1]), gl.STATIC_DRAW);

  const colorLocation = gl.getUniformLocation(pointProgram, 'u_color');
  const useDepthColorLocation = gl.getUniformLocation(pointProgram, 'u_useDepthColor');
  const pointSizeLocation = gl.getUniformLocation(pointProgram, 'u_pointSize');
  const worldSizeLocation = gl.getUniformLocation(pointProgram, 'u_worldSize');
  if (!colorLocation || !useDepthColorLocation || !pointSizeLocation || !worldSizeLocation) {
    throw new Error('failed to resolve WebGL uniforms');
  }
  const boxColorLocation = gl.getUniformLocation(boxProgram, 'u_color');
  const boxUseDepthColorLocation = gl.getUniformLocation(boxProgram, 'u_useDepthColor');
  const boxWorldSizeLocation = gl.getUniformLocation(boxProgram, 'u_worldSize');
  if (!boxColorLocation || !boxUseDepthColorLocation || !boxWorldSizeLocation) {
    throw new Error('failed to resolve WebGL box uniforms');
  }
  const indexedBoxColorLocation = gl.getUniformLocation(indexedBoxProgram, 'u_color');
  const indexedBoxWorldSizeLocation = gl.getUniformLocation(indexedBoxProgram, 'u_worldSize');
  const indexedBoxTextureLocation = gl.getUniformLocation(indexedBoxProgram, 'u_boxTexture');
  const indexedBoxHitTextureLocation = gl.getUniformLocation(indexedBoxProgram, 'u_hitTexture');
  const indexedBoxTextureWidthLocation = gl.getUniformLocation(indexedBoxProgram, 'u_boxTextureWidth');
  const indexedBoxHitTextureWidthLocation = gl.getUniformLocation(indexedBoxProgram, 'u_hitTextureWidth');
  if (
    !indexedBoxColorLocation ||
    !indexedBoxWorldSizeLocation ||
    !indexedBoxTextureLocation ||
    !indexedBoxHitTextureLocation ||
    !indexedBoxTextureWidthLocation ||
    !indexedBoxHitTextureWidthLocation
  ) {
    throw new Error('failed to resolve WebGL indexed box uniforms');
  }
  const indexedPointColorLocation = gl.getUniformLocation(indexedPointProgram, 'u_color');
  const indexedPointWorldSizeLocation = gl.getUniformLocation(indexedPointProgram, 'u_worldSize');
  const indexedPointSizeLocation = gl.getUniformLocation(indexedPointProgram, 'u_pointSize');
  const indexedPointTextureLocation = gl.getUniformLocation(indexedPointProgram, 'u_itemTexture');
  const indexedPointHitTextureLocation = gl.getUniformLocation(indexedPointProgram, 'u_hitTexture');
  const indexedPointTextureWidthLocation = gl.getUniformLocation(indexedPointProgram, 'u_itemTextureWidth');
  const indexedPointHitTextureWidthLocation = gl.getUniformLocation(indexedPointProgram, 'u_hitTextureWidth');
  if (
    !indexedPointColorLocation ||
    !indexedPointWorldSizeLocation ||
    !indexedPointSizeLocation ||
    !indexedPointTextureLocation ||
    !indexedPointHitTextureLocation ||
    !indexedPointTextureWidthLocation ||
    !indexedPointHitTextureWidthLocation
  ) {
    throw new Error('failed to resolve WebGL indexed point uniforms');
  }

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  return {
    gl,
    pointProgram,
    boxProgram,
    indexedBoxProgram,
    indexedPointProgram,
    positionLocation: gl.getAttribLocation(pointProgram, 'a_position'),
    colorLocation,
    useDepthColorLocation,
    pointSizeLocation,
    worldSizeLocation,
    boxCornerLocation: gl.getAttribLocation(boxProgram, 'a_corner'),
    boxLocation: gl.getAttribLocation(boxProgram, 'a_box'),
    boxDepthLocation: gl.getAttribLocation(boxProgram, 'a_depth'),
    boxColorLocation,
    boxUseDepthColorLocation,
    boxWorldSizeLocation,
    indexedBoxCornerLocation: gl.getAttribLocation(indexedBoxProgram, 'a_corner'),
    indexedBoxColorLocation,
    indexedBoxWorldSizeLocation,
    indexedBoxTextureLocation,
    indexedBoxHitTextureLocation,
    indexedBoxTextureWidthLocation,
    indexedBoxHitTextureWidthLocation,
    indexedPointColorLocation,
    indexedPointWorldSizeLocation,
    indexedPointSizeLocation,
    indexedPointTextureLocation,
    indexedPointHitTextureLocation,
    indexedPointTextureWidthLocation,
    indexedPointHitTextureWidthLocation,
    unitQuadBuffer,
    pointBuffer,
    boxBuffer,
    itemTexture,
    hitIndexTexture,
    itemTextureWidth: 1,
    hitTextureWidth: 1,
    maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
    itemCount: 0,
    hitCount: 0,
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
