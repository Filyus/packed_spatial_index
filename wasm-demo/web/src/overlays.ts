import { mustQuery } from './dom';
import { itemStrideFor, type Dimension, type Geometry, type WorldView } from './rendering';

export type QueryRect = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

export type QueryPoint = {
  x: number;
  y: number;
};

/** Read-only view of the frame state the overlays draw from. */
export type OverlayState = {
  mode: 'range' | 'nearest';
  resultMode: 'all' | 'any' | 'first';
  dimension: Dimension;
  geometry: Geometry;
  query: QueryRect | null;
  queryPoint: QueryPoint | null;
  anyResult: boolean | null;
  hits: Uint32Array<ArrayBufferLike>;
  items: Float64Array<ArrayBufferLike>;
  worldView: WorldView;
};

export type Overlays = {
  render(state: OverlayState): void;
};

/**
 * The HTML overlay layer drawn on top of the canvas: the query rectangle, the
 * nearest-query marker, the 3D depth shade, and the first-hit highlight. Owns
 * its DOM elements and maps world coordinates to the stage's pixel space.
 */
export function createOverlays(stage: HTMLElement): Overlays {
  const queryShadeEls = [
    mustQuery<HTMLDivElement>('#queryShadeTop'),
    mustQuery<HTMLDivElement>('#queryShadeRight'),
    mustQuery<HTMLDivElement>('#queryShadeBottom'),
    mustQuery<HTMLDivElement>('#queryShadeLeft'),
  ] as const;
  const queryRectEl = mustQuery<HTMLDivElement>('#queryRect');
  const queryPointEl = mustQuery<HTMLDivElement>('#queryPoint');
  const firstHitEl = mustQuery<HTMLDivElement>('#firstHit');

  function worldToCanvasX(x: number, worldView: WorldView): number {
    return (x / worldView.width) * stage.clientWidth;
  }

  function worldToCanvasY(y: number, worldView: WorldView): number {
    return (y / worldView.height) * stage.clientHeight;
  }

  function renderQueryOverlay(state: OverlayState): void {
    const { mode, query, queryPoint, anyResult, worldView } = state;
    if (mode !== 'range' || !query) {
      queryRectEl.style.display = 'none';
      hideQueryShade();
    } else {
      const x0 = worldToCanvasX(query.minX, worldView);
      const y0 = worldToCanvasY(query.minY, worldView);
      const x1 = worldToCanvasX(query.maxX, worldView);
      const y1 = worldToCanvasY(query.maxY, worldView);
      const left = Math.min(x0, x1);
      const top = Math.min(y0, y1);
      const width = Math.abs(x1 - x0);
      const height = Math.abs(y1 - y0);
      queryRectEl.style.display = 'block';
      queryRectEl.classList.toggle('is-match', anyResult === true);
      queryRectEl.classList.toggle('is-miss', anyResult === false);
      queryRectEl.style.transform = `translate(${left}px, ${top}px)`;
      queryRectEl.style.width = `${width}px`;
      queryRectEl.style.height = `${height}px`;
      if (state.dimension === '3d') {
        renderQueryShade(left, top, width, height);
      } else {
        hideQueryShade();
      }
    }

    if (mode !== 'nearest' || !queryPoint) {
      queryPointEl.style.display = 'none';
    } else {
      queryPointEl.style.display = 'block';
      queryPointEl.style.transform = `translate(${worldToCanvasX(queryPoint.x, worldView)}px, ${worldToCanvasY(queryPoint.y, worldView)}px)`;
    }
  }

  function renderQueryShade(left: number, top: number, width: number, height: number): void {
    const stageWidth = stage.clientWidth;
    const stageHeight = stage.clientHeight;
    setShadeRect(queryShadeEls[0], 0, 0, stageWidth, top);
    setShadeRect(queryShadeEls[1], left + width, top, Math.max(0, stageWidth - left - width), height);
    setShadeRect(queryShadeEls[2], 0, top + height, stageWidth, Math.max(0, stageHeight - top - height));
    setShadeRect(queryShadeEls[3], 0, top, left, height);
  }

  function hideQueryShade(): void {
    for (const shade of queryShadeEls) {
      shade.style.display = 'none';
    }
  }

  function renderFirstHitOverlay(state: OverlayState): void {
    const { mode, resultMode, geometry, dimension, hits, items, worldView } = state;
    if (mode !== 'range' || resultMode !== 'first' || hits.length === 0) {
      firstHitEl.style.display = 'none';
      return;
    }

    const stride = itemStrideFor(dimension, geometry);
    const offset = hits[0] * stride;
    let x0: number;
    let y0: number;
    let x1: number;
    let y1: number;
    if (geometry === 'boxes') {
      x0 = worldToCanvasX(items[offset], worldView);
      y0 = worldToCanvasY(items[offset + 1], worldView);
      const maxOffset = dimension === '3d' ? offset + 3 : offset + 2;
      x1 = worldToCanvasX(items[maxOffset], worldView);
      y1 = worldToCanvasY(items[maxOffset + 1], worldView);
      firstHitEl.classList.remove('is-point');
      firstHitEl.classList.add('is-box');
    } else {
      const x = worldToCanvasX(items[offset], worldView);
      const y = worldToCanvasY(items[offset + 1], worldView);
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
    const minSize = geometry === 'boxes' ? 10 : 16;
    const padding = geometry === 'boxes' ? 4 : 0;
    const width = Math.max(minSize, Math.abs(x1 - x0) + padding * 2);
    const height = Math.max(minSize, Math.abs(y1 - y0) + padding * 2);
    const left = centerX - width * 0.5;
    const top = centerY - height * 0.5;
    firstHitEl.style.display = 'block';
    firstHitEl.style.transform = `translate(${left}px, ${top}px)`;
    firstHitEl.style.width = `${width}px`;
    firstHitEl.style.height = `${height}px`;
  }

  return {
    render(state: OverlayState): void {
      renderQueryOverlay(state);
      renderFirstHitOverlay(state);
    },
  };
}

function setShadeRect(element: HTMLDivElement, left: number, top: number, width: number, height: number): void {
  if (width <= 0 || height <= 0) {
    element.style.display = 'none';
    return;
  }
  element.style.display = 'block';
  element.style.transform = `translate(${left}px, ${top}px)`;
  element.style.width = `${width}px`;
  element.style.height = `${height}px`;
}
