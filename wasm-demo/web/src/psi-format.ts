import { itemStrideFor, type Geometry } from './rendering';

// See FORMAT.md. The .psi buffer is the raw index serialization: a 64-byte
// header, then level_bounds, box records, and indices.
const PSI_MAGIC = [0x50, 0x53, 0x49, 0x4e, 0x44, 0x45, 0x58, 0x00]; // "PSINDEX\0"
const PSI_HEADER_LEN = 64;

export type PsiHeader = {
  flags: number;
  nodeSize: number;
  numItems: number;
  numNodes: number;
  levelCount: number;
  is3d: boolean;
  isF32: boolean;
  recordBytes: number;
  fieldBytes: number;
};

export function parsePsiHeader(view: DataView): PsiHeader {
  if (view.byteLength < PSI_HEADER_LEN) {
    throw new Error('File is too small to be a .psi index');
  }
  for (let i = 0; i < PSI_MAGIC.length; i++) {
    if (view.getUint8(i) !== PSI_MAGIC[i]) {
      throw new Error('Not a packed_spatial_index (.psi) file: bad magic');
    }
  }
  const formatVersion = Number(view.getBigUint64(8, true));
  const headerLen = Number(view.getBigUint64(16, true));
  if (formatVersion !== 1 || headerLen !== PSI_HEADER_LEN) {
    throw new Error(`Unsupported .psi format (version ${formatVersion}, header ${headerLen})`);
  }
  const flags = Number(view.getBigUint64(24, true));
  if (flags > 3) {
    throw new Error(`Unsupported .psi flags ${flags}`);
  }
  const is3d = flags === 1 || flags === 3;
  const isF32 = flags === 2 || flags === 3;
  const dims = is3d ? 3 : 2;
  const fieldBytes = isF32 ? 4 : 8;
  return {
    flags,
    nodeSize: Number(view.getBigUint64(32, true)),
    numItems: Number(view.getBigUint64(40, true)),
    numNodes: Number(view.getBigUint64(48, true)),
    levelCount: Number(view.getBigUint64(56, true)),
    is3d,
    isF32,
    recordBytes: dims * 2 * fieldBytes,
    fieldBytes,
  };
}

// Rebuild the render dataset from the leaf box records. Per FORMAT.md the first
// `numItems` nodes are the leaves (in packed order), and each leaf's `indices`
// entry is its original insertion index. Points are stored as degenerate boxes,
// so an all-degenerate leaf set is rendered as points.
export function reconstructItems(
  buffer: ArrayBuffer,
  header: PsiHeader,
): { items: Float64Array; geometry: Geometry } {
  const { numItems, numNodes, levelCount, recordBytes, fieldBytes, is3d } = header;
  const view = new DataView(buffer);
  const dims = is3d ? 3 : 2;
  const boxesStart = PSI_HEADER_LEN + 8 * levelCount;
  const indicesStart = boxesStart + recordBytes * numNodes;
  const readField = header.isF32
    ? (offset: number) => view.getFloat32(offset, true)
    : (offset: number) => view.getFloat64(offset, true);

  let isPoints = true;
  for (let i = 0; i < numItems && isPoints; i++) {
    const recordOffset = boxesStart + i * recordBytes;
    for (let d = 0; d < dims; d++) {
      if (readField(recordOffset + d * fieldBytes) !== readField(recordOffset + (dims + d) * fieldBytes)) {
        isPoints = false;
        break;
      }
    }
  }

  const geometry: Geometry = isPoints ? 'points' : 'boxes';
  const stride = itemStrideFor(is3d ? '3d' : '2d', geometry);
  const fieldCount = geometry === 'points' ? dims : dims * 2;
  const items = new Float64Array(numItems * stride);
  for (let i = 0; i < numItems; i++) {
    const recordOffset = boxesStart + i * recordBytes;
    const original = Number(view.getBigUint64(indicesStart + i * 8, true));
    const dst = original * stride;
    for (let f = 0; f < fieldCount; f++) {
      items[dst + f] = readField(recordOffset + f * fieldBytes);
    }
  }
  return { items, geometry };
}
