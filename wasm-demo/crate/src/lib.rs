use js_sys::{Float64Array, Object, Reflect, Uint8Array, Uint32Array};
use packed_spatial_index::{
    Box2D, Index2DBuilder, NeighborWorkspace, Point2D, SearchWorkspace, SimdIndex2D,
};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmIndex2D {
    index: SimdIndex2D,
    workspace: SearchWorkspace,
    neighbor_workspace: NeighborWorkspace,
    len: usize,
}

#[wasm_bindgen]
impl WasmIndex2D {
    #[wasm_bindgen(constructor)]
    pub fn new(boxes: &Float64Array, node_size: usize) -> Result<WasmIndex2D, JsValue> {
        build_from_boxes(&boxes.to_vec(), node_size)
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
            if !x.is_finite() || !y.is_finite() {
                return Err(JsValue::from_str(
                    "point coordinates must be finite numbers",
                ));
            }
            builder.add(Box2D::from_point(Point2D::new(x, y)));
        }

        let index = builder
            .finish_simd()
            .map_err(|err| JsValue::from_str(&err.to_string()))?;

        Ok(wrap_index(index))
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
        Ok(wrap_index(index))
    }

    pub fn search(
        &mut self,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> Result<Uint32Array, JsValue> {
        validate_query(min_x, min_y, max_x, max_y)?;

        let hits = self
            .index
            .search_with(Box2D::new(min_x, min_y, max_x, max_y), &mut self.workspace);
        hits_to_uint32(hits)
    }

    pub fn search_profile(
        &mut self,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> Result<Object, JsValue> {
        validate_query(min_x, min_y, max_x, max_y)?;

        let started = now_ms();
        let hits = self
            .index
            .search_with(Box2D::new(min_x, min_y, max_x, max_y), &mut self.workspace);
        let traversed = now_ms();

        let mut out = Vec::with_capacity(hits.len());
        for &hit in hits {
            let hit = u32::try_from(hit)
                .map_err(|_| JsValue::from_str("hit index does not fit in Uint32Array"))?;
            out.push(hit);
        }
        let converted = now_ms();

        let array = Uint32Array::from(out.as_slice());
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
            &JsValue::from_f64(converted - traversed),
        )?;
        Reflect::set(
            &result,
            &"copyMs".into(),
            &JsValue::from_f64(copied - converted),
        )?;
        Reflect::set(
            &result,
            &"totalMs".into(),
            &JsValue::from_f64(copied - started),
        )?;
        Ok(result)
    }

    pub fn any(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Result<bool, JsValue> {
        validate_query(min_x, min_y, max_x, max_y)?;
        Ok(self.index.any(Box2D::new(min_x, min_y, max_x, max_y)))
    }

    pub fn first(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Result<i32, JsValue> {
        validate_query(min_x, min_y, max_x, max_y)?;
        match self.index.first(Box2D::new(min_x, min_y, max_x, max_y)) {
            Some(hit) => {
                i32::try_from(hit).map_err(|_| JsValue::from_str("hit index does not fit in i32"))
            }
            None => Ok(-1),
        }
    }

    pub fn neighbors(
        &mut self,
        x: f64,
        y: f64,
        max_results: usize,
    ) -> Result<Uint32Array, JsValue> {
        if !x.is_finite() || !y.is_finite() {
            return Err(JsValue::from_str(
                "query point coordinates must be finite numbers",
            ));
        }

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
        if !x.is_finite() || !y.is_finite() || !max_distance.is_finite() {
            return Err(JsValue::from_str(
                "query point and max_distance must be finite numbers",
            ));
        }
        if max_distance < 0.0 {
            return Err(JsValue::from_str("max_distance must be non-negative"));
        }

        let hits = self.index.neighbors_with(
            Point2D::new(x, y),
            max_results,
            max_distance,
            &mut self.neighbor_workspace,
        );
        hits_to_uint32(hits)
    }
}

fn build_from_boxes(coords: &[f64], node_size: usize) -> Result<WasmIndex2D, JsValue> {
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
        validate_query(min_x, min_y, max_x, max_y)?;
        builder.add(Box2D::new(min_x, min_y, max_x, max_y));
    }

    let index = builder
        .finish_simd()
        .map_err(|err| JsValue::from_str(&err.to_string()))?;

    Ok(wrap_index(index))
}

fn wrap_index(index: SimdIndex2D) -> WasmIndex2D {
    let len = index.num_items();
    WasmIndex2D {
        index,
        workspace: SearchWorkspace::new(),
        neighbor_workspace: NeighborWorkspace::new(),
        len,
    }
}

fn hits_to_uint32(hits: &[usize]) -> Result<Uint32Array, JsValue> {
    let mut out = Vec::with_capacity(hits.len());
    for &hit in hits {
        let hit = u32::try_from(hit)
            .map_err(|_| JsValue::from_str("hit index does not fit in Uint32Array"))?;
        out.push(hit);
    }
    Ok(Uint32Array::from(out.as_slice()))
}

fn validate_query(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Result<(), JsValue> {
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

fn now_ms() -> f64 {
    js_sys::global()
        .dyn_into::<web_sys::Window>()
        .ok()
        .and_then(|window| window.performance())
        .map_or_else(js_sys::Date::now, |performance| performance.now())
}
