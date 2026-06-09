use std::ops::ControlFlow;

use js_sys::{Float64Array, Object, Reflect, Uint8Array, Uint32Array};
use packed_spatial_index::{
    Box2D, Box3D, Index2DBuilder, Index3DBuilder, NeighborWorkspace, Point2D, Point3D,
    SimdIndex2D, SimdIndex3D,
};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmIndex2D {
    index: SimdIndex2D,
    neighbor_workspace: NeighborWorkspace,
    search_stack: Vec<usize>,
    hit_indices: Vec<u32>,
    len: usize,
}

#[wasm_bindgen]
pub struct WasmIndex3D {
    index: SimdIndex3D,
    neighbor_workspace: NeighborWorkspace,
    search_stack: Vec<usize>,
    hit_indices: Vec<u32>,
    len: usize,
}

#[wasm_bindgen]
impl WasmIndex2D {
    #[wasm_bindgen(constructor)]
    pub fn new(boxes: &Float64Array, node_size: usize) -> Result<WasmIndex2D, JsValue> {
        build_from_boxes2d(&boxes.to_vec(), node_size)
    }

    pub fn from_points(points: &Float64Array, node_size: usize) -> Result<WasmIndex2D, JsValue> {
        let coords = points.to_vec();
        if !coords.len().is_multiple_of(2) {
            return Err(JsValue::from_str(
                "points must be a flat [x0, y0, x1, y1, ...] array",
            ));
        }

        let len = coords.len() / 2;
        let mut builder = Index2DBuilder::new(len).node_size(node_size);
        for pair in coords.chunks_exact(2) {
            let x = pair[0];
            let y = pair[1];
            validate_point2d(x, y)?;
            builder.add(Box2D::from_point(Point2D::new(x, y)));
        }

        let index = builder
            .finish_simd()
            .map_err(|err| JsValue::from_str(&err.to_string()))?;

        Ok(wrap_index2d(index))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn node_size(&self) -> usize {
        self.index.node_size()
    }

    pub fn extent(&self) -> Result<Object, JsValue> {
        let out = Object::new();
        match self.index.extent() {
            Some(bounds) => {
                Reflect::set(&out, &"empty".into(), &JsValue::FALSE)?;
                Reflect::set(&out, &"minX".into(), &JsValue::from_f64(bounds.min_x))?;
                Reflect::set(&out, &"minY".into(), &JsValue::from_f64(bounds.min_y))?;
                Reflect::set(&out, &"maxX".into(), &JsValue::from_f64(bounds.max_x))?;
                Reflect::set(&out, &"maxY".into(), &JsValue::from_f64(bounds.max_y))?;
            }
            None => {
                Reflect::set(&out, &"empty".into(), &JsValue::TRUE)?;
            }
        }
        Ok(out)
    }

    pub fn to_bytes(&self) -> Uint8Array {
        Uint8Array::from(self.index.to_bytes().as_slice())
    }

    pub fn from_bytes(bytes: &Uint8Array) -> Result<WasmIndex2D, JsValue> {
        let index = SimdIndex2D::from_bytes(&bytes.to_vec())
            .map_err(|err| JsValue::from_str(&err.to_string()))?;
        Ok(wrap_index2d(index))
    }

    pub fn search(
        &mut self,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> Result<Uint32Array, JsValue> {
        validate_query2d(min_x, min_y, max_x, max_y)?;

        self.search_u32(Box2D::new(min_x, min_y, max_x, max_y))?;
        Ok(Uint32Array::from(self.hit_indices.as_slice()))
    }

    pub fn search_profile(
        &mut self,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> Result<Object, JsValue> {
        validate_query2d(min_x, min_y, max_x, max_y)?;

        let started = now_ms();
        self.search_u32(Box2D::new(min_x, min_y, max_x, max_y))?;
        let traversed = now_ms();

        let array = Uint32Array::from(self.hit_indices.as_slice());
        let copied = now_ms();

        let result = Object::new();
        Reflect::set(&result, &"hits".into(), &array.into())?;
        Reflect::set(
            &result,
            &"traverseMs".into(),
            &JsValue::from_f64(traversed - started),
        )?;
        Reflect::set(
            &result,
            &"convertMs".into(),
            &JsValue::from_f64(0.0),
        )?;
        Reflect::set(
            &result,
            &"copyMs".into(),
            &JsValue::from_f64(copied - traversed),
        )?;
        Reflect::set(
            &result,
            &"totalMs".into(),
            &JsValue::from_f64(copied - started),
        )?;
        Ok(result)
    }

    pub fn any(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Result<bool, JsValue> {
        validate_query2d(min_x, min_y, max_x, max_y)?;
        Ok(self.index.any(Box2D::new(min_x, min_y, max_x, max_y)))
    }

    pub fn first(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Result<i32, JsValue> {
        validate_query2d(min_x, min_y, max_x, max_y)?;
        match self.index.first(Box2D::new(min_x, min_y, max_x, max_y)) {
            Some(hit) => {
                i32::try_from(hit).map_err(|_| JsValue::from_str("hit index does not fit in i32"))
            }
            None => Ok(-1),
        }
    }

    fn search_u32(&mut self, query: Box2D) -> Result<(), JsValue> {
        self.hit_indices.clear();
        if self
            .index
            .extent()
            .is_some_and(|extent| query.contains(extent))
        {
            fill_all_indices_u32(self.len, &mut self.hit_indices)?;
            return Ok(());
        }

        let hits = &mut self.hit_indices;
        let flow = self
            .index
            .visit_avx512(query, &mut self.search_stack, |hit| {
                let Ok(hit) = u32::try_from(hit) else {
                    return ControlFlow::Break(JsValue::from_str("hit index does not fit in u32"));
                };
                hits.push(hit);
                ControlFlow::Continue(())
            });
        match flow {
            ControlFlow::Continue(()) => Ok(()),
            ControlFlow::Break(err) => Err(err),
        }
    }

    pub fn neighbors(
        &mut self,
        x: f64,
        y: f64,
        max_results: usize,
    ) -> Result<Uint32Array, JsValue> {
        validate_point2d(x, y)?;

        let hits = self.index.neighbors_with(
            Point2D::new(x, y),
            max_results,
            f64::INFINITY,
            &mut self.neighbor_workspace,
        );
        hits_to_uint32(hits)
    }

    pub fn neighbors_within(
        &mut self,
        x: f64,
        y: f64,
        max_results: usize,
        max_distance: f64,
    ) -> Result<Uint32Array, JsValue> {
        validate_point2d(x, y)?;
        validate_max_distance(max_distance)?;

        let hits = self.index.neighbors_with(
            Point2D::new(x, y),
            max_results,
            max_distance,
            &mut self.neighbor_workspace,
        );
        hits_to_uint32(hits)
    }
}

#[wasm_bindgen]
impl WasmIndex3D {
    #[wasm_bindgen(constructor)]
    pub fn new(boxes: &Float64Array, node_size: usize) -> Result<WasmIndex3D, JsValue> {
        build_from_boxes3d(&boxes.to_vec(), node_size)
    }

    pub fn from_points(points: &Float64Array, node_size: usize) -> Result<WasmIndex3D, JsValue> {
        let coords = points.to_vec();
        if !coords.len().is_multiple_of(3) {
            return Err(JsValue::from_str(
                "points must be a flat [x0, y0, z0, x1, y1, z1, ...] array",
            ));
        }

        let len = coords.len() / 3;
        let mut builder = Index3DBuilder::new(len).node_size(node_size);
        for triple in coords.chunks_exact(3) {
            let x = triple[0];
            let y = triple[1];
            let z = triple[2];
            validate_point3d(x, y, z)?;
            builder.add(Box3D::from_point(Point3D::new(x, y, z)));
        }

        let index = builder
            .finish_simd()
            .map_err(|err| JsValue::from_str(&err.to_string()))?;

        Ok(wrap_index3d(index))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn node_size(&self) -> usize {
        self.index.node_size()
    }

    pub fn extent(&self) -> Result<Object, JsValue> {
        let out = Object::new();
        match self.index.extent() {
            Some(bounds) => {
                Reflect::set(&out, &"empty".into(), &JsValue::FALSE)?;
                Reflect::set(&out, &"minX".into(), &JsValue::from_f64(bounds.min_x))?;
                Reflect::set(&out, &"minY".into(), &JsValue::from_f64(bounds.min_y))?;
                Reflect::set(&out, &"minZ".into(), &JsValue::from_f64(bounds.min_z))?;
                Reflect::set(&out, &"maxX".into(), &JsValue::from_f64(bounds.max_x))?;
                Reflect::set(&out, &"maxY".into(), &JsValue::from_f64(bounds.max_y))?;
                Reflect::set(&out, &"maxZ".into(), &JsValue::from_f64(bounds.max_z))?;
            }
            None => {
                Reflect::set(&out, &"empty".into(), &JsValue::TRUE)?;
            }
        }
        Ok(out)
    }

    pub fn to_bytes(&self) -> Uint8Array {
        Uint8Array::from(self.index.to_bytes().as_slice())
    }

    pub fn from_bytes(bytes: &Uint8Array) -> Result<WasmIndex3D, JsValue> {
        let index = SimdIndex3D::from_bytes(&bytes.to_vec())
            .map_err(|err| JsValue::from_str(&err.to_string()))?;
        Ok(wrap_index3d(index))
    }

    pub fn search(
        &mut self,
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
    ) -> Result<Uint32Array, JsValue> {
        validate_query3d(min_x, min_y, min_z, max_x, max_y, max_z)?;

        self.search_u32(Box3D::new(min_x, min_y, min_z, max_x, max_y, max_z))?;
        Ok(Uint32Array::from(self.hit_indices.as_slice()))
    }

    pub fn search_profile(
        &mut self,
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
    ) -> Result<Object, JsValue> {
        validate_query3d(min_x, min_y, min_z, max_x, max_y, max_z)?;

        let started = now_ms();
        self.search_u32(Box3D::new(min_x, min_y, min_z, max_x, max_y, max_z))?;
        let traversed = now_ms();

        let array = Uint32Array::from(self.hit_indices.as_slice());
        let copied = now_ms();

        let result = Object::new();
        Reflect::set(&result, &"hits".into(), &array.into())?;
        Reflect::set(
            &result,
            &"traverseMs".into(),
            &JsValue::from_f64(traversed - started),
        )?;
        Reflect::set(
            &result,
            &"convertMs".into(),
            &JsValue::from_f64(0.0),
        )?;
        Reflect::set(
            &result,
            &"copyMs".into(),
            &JsValue::from_f64(copied - traversed),
        )?;
        Reflect::set(
            &result,
            &"totalMs".into(),
            &JsValue::from_f64(copied - started),
        )?;
        Ok(result)
    }

    pub fn any(
        &self,
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
    ) -> Result<bool, JsValue> {
        validate_query3d(min_x, min_y, min_z, max_x, max_y, max_z)?;
        Ok(self
            .index
            .any(Box3D::new(min_x, min_y, min_z, max_x, max_y, max_z)))
    }

    pub fn first(
        &self,
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
    ) -> Result<i32, JsValue> {
        validate_query3d(min_x, min_y, min_z, max_x, max_y, max_z)?;
        match self
            .index
            .first(Box3D::new(min_x, min_y, min_z, max_x, max_y, max_z))
        {
            Some(hit) => {
                i32::try_from(hit).map_err(|_| JsValue::from_str("hit index does not fit in i32"))
            }
            None => Ok(-1),
        }
    }

    fn search_u32(&mut self, query: Box3D) -> Result<(), JsValue> {
        self.hit_indices.clear();
        if self
            .index
            .extent()
            .is_some_and(|extent| query.contains(extent))
        {
            fill_all_indices_u32(self.len, &mut self.hit_indices)?;
            return Ok(());
        }

        let hits = &mut self.hit_indices;
        let flow = self
            .index
            .visit_avx512(query, &mut self.search_stack, |hit| {
                let Ok(hit) = u32::try_from(hit) else {
                    return ControlFlow::Break(JsValue::from_str("hit index does not fit in u32"));
                };
                hits.push(hit);
                ControlFlow::Continue(())
            });
        match flow {
            ControlFlow::Continue(()) => Ok(()),
            ControlFlow::Break(err) => Err(err),
        }
    }

    pub fn neighbors(
        &mut self,
        x: f64,
        y: f64,
        z: f64,
        max_results: usize,
    ) -> Result<Uint32Array, JsValue> {
        validate_point3d(x, y, z)?;

        let hits = self.index.neighbors_with(
            Point3D::new(x, y, z),
            max_results,
            f64::INFINITY,
            &mut self.neighbor_workspace,
        );
        hits_to_uint32(hits)
    }

    pub fn neighbors_within(
        &mut self,
        x: f64,
        y: f64,
        z: f64,
        max_results: usize,
        max_distance: f64,
    ) -> Result<Uint32Array, JsValue> {
        validate_point3d(x, y, z)?;
        validate_max_distance(max_distance)?;

        let hits = self.index.neighbors_with(
            Point3D::new(x, y, z),
            max_results,
            max_distance,
            &mut self.neighbor_workspace,
        );
        hits_to_uint32(hits)
    }
}

fn build_from_boxes2d(coords: &[f64], node_size: usize) -> Result<WasmIndex2D, JsValue> {
    if !coords.len().is_multiple_of(4) {
        return Err(JsValue::from_str(
            "boxes must be a flat [min_x, min_y, max_x, max_y, ...] array",
        ));
    }

    let len = coords.len() / 4;
    let mut builder = Index2DBuilder::new(len).node_size(node_size);
    for quad in coords.chunks_exact(4) {
        let min_x = quad[0];
        let min_y = quad[1];
        let max_x = quad[2];
        let max_y = quad[3];
        validate_query2d(min_x, min_y, max_x, max_y)?;
        builder.add(Box2D::new(min_x, min_y, max_x, max_y));
    }

    let index = builder
        .finish_simd()
        .map_err(|err| JsValue::from_str(&err.to_string()))?;

    Ok(wrap_index2d(index))
}

fn build_from_boxes3d(coords: &[f64], node_size: usize) -> Result<WasmIndex3D, JsValue> {
    if !coords.len().is_multiple_of(6) {
        return Err(JsValue::from_str(
            "boxes must be a flat [min_x, min_y, min_z, max_x, max_y, max_z, ...] array",
        ));
    }

    let len = coords.len() / 6;
    let mut builder = Index3DBuilder::new(len).node_size(node_size);
    for sext in coords.chunks_exact(6) {
        let min_x = sext[0];
        let min_y = sext[1];
        let min_z = sext[2];
        let max_x = sext[3];
        let max_y = sext[4];
        let max_z = sext[5];
        validate_query3d(min_x, min_y, min_z, max_x, max_y, max_z)?;
        builder.add(Box3D::new(min_x, min_y, min_z, max_x, max_y, max_z));
    }

    let index = builder
        .finish_simd()
        .map_err(|err| JsValue::from_str(&err.to_string()))?;

    Ok(wrap_index3d(index))
}

fn wrap_index2d(index: SimdIndex2D) -> WasmIndex2D {
    let len = index.num_items();
    WasmIndex2D {
        index,
        neighbor_workspace: NeighborWorkspace::new(),
        search_stack: Vec::new(),
        hit_indices: Vec::new(),
        len,
    }
}

fn wrap_index3d(index: SimdIndex3D) -> WasmIndex3D {
    let len = index.num_items();
    WasmIndex3D {
        index,
        neighbor_workspace: NeighborWorkspace::new(),
        search_stack: Vec::new(),
        hit_indices: Vec::new(),
        len,
    }
}

fn fill_all_indices_u32(len: usize, out: &mut Vec<u32>) -> Result<(), JsValue> {
    let len =
        u32::try_from(len).map_err(|_| JsValue::from_str("hit index does not fit in u32"))?;
    out.extend(0..len);
    Ok(())
}

fn hits_to_uint32(hits: &[usize]) -> Result<Uint32Array, JsValue> {
    let out = hits_to_u32_vec(hits)?;
    Ok(Uint32Array::from(out.as_slice()))
}

fn hits_to_u32_vec(hits: &[usize]) -> Result<Vec<u32>, JsValue> {
    let mut out = Vec::with_capacity(hits.len());
    for &hit in hits {
        let hit = u32::try_from(hit)
            .map_err(|_| JsValue::from_str("hit index does not fit in Uint32Array"))?;
        out.push(hit);
    }
    Ok(out)
}

fn validate_query2d(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Result<(), JsValue> {
    if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
        return Err(JsValue::from_str(
            "query coordinates must be finite numbers",
        ));
    }
    if min_x > max_x || min_y > max_y {
        return Err(JsValue::from_str(
            "query bounds must satisfy min_x <= max_x and min_y <= max_y",
        ));
    }
    Ok(())
}

fn validate_point2d(x: f64, y: f64) -> Result<(), JsValue> {
    if !x.is_finite() || !y.is_finite() {
        return Err(JsValue::from_str(
            "query point coordinates must be finite numbers",
        ));
    }
    Ok(())
}

fn validate_point3d(x: f64, y: f64, z: f64) -> Result<(), JsValue> {
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return Err(JsValue::from_str(
            "query point coordinates must be finite numbers",
        ));
    }
    Ok(())
}

fn validate_max_distance(max_distance: f64) -> Result<(), JsValue> {
    if !max_distance.is_finite() {
        return Err(JsValue::from_str(
            "query point and max_distance must be finite numbers",
        ));
    }
    if max_distance < 0.0 {
        return Err(JsValue::from_str("max_distance must be non-negative"));
    }
    Ok(())
}

fn validate_query3d(
    min_x: f64,
    min_y: f64,
    min_z: f64,
    max_x: f64,
    max_y: f64,
    max_z: f64,
) -> Result<(), JsValue> {
    if !min_x.is_finite()
        || !min_y.is_finite()
        || !min_z.is_finite()
        || !max_x.is_finite()
        || !max_y.is_finite()
        || !max_z.is_finite()
    {
        return Err(JsValue::from_str(
            "query coordinates must be finite numbers",
        ));
    }
    if min_x > max_x || min_y > max_y || min_z > max_z {
        return Err(JsValue::from_str(
            "query bounds must satisfy min_x <= max_x, min_y <= max_y, and min_z <= max_z",
        ));
    }
    Ok(())
}

fn now_ms() -> f64 {
    js_sys::global()
        .dyn_into::<web_sys::Window>()
        .ok()
        .and_then(|window| window.performance())
        .map_or_else(js_sys::Date::now, |performance| performance.now())
}
