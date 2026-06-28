# Нейминг запросов в core

Заметка фиксирует решение по словам `search`, `overlap` и `intersect` внутри
core API. Это не документ про `geo`: внешний GIS-словарь не должен определять
форму core.

## Принятый публичный API

Для range/region overlap в core публично используется одна короткая семья:

- `search`
- `search_iter`
- `search_into`
- `search_with`
- `any`
- `first`
- `visit`

Эти методы принимают обычный AABB query по значению:

```rust
index.search(Box2D::new(...));
index.search(Box3D::new(...));
```

И те же методы принимают borrowed region geometry:

```rust
index.search(&triangle);
index.search(&polygon);
index.search(&frustum);

index.search_into(&triangle, &mut out);
index.search_with(&frustum, &mut workspace);
index.any(&polygon);
index.first(&triangle);
index.visit(&frustum, |id| ...);
index.search_iter(&triangle);
```

Старые публичные `search_overlaps`, `search_overlaps_into`,
`search_overlaps_with`, `any_overlaps`, `first_overlap`, `visit_overlaps`
удалены. Они создавали вторую параллельную систему имен рядом с уже
существующими `search` / `any` / `first` / `visit`.

## Где core API намеренно несимметричен

### `search(query)` vs `query.overlaps_box(box)`

Это главная оставшаяся несимметрия, и она полезная.

`search` на индексе называет действие пользователя: найти элементы индекса.
Слово `overlap` там не нужно, потому что в core `search` уже исторически означает
AABB overlap search.

`overlaps_box` на query type называет внутренний predicate contract: принимает ли
эта query geometry конкретный AABB. Здесь suffix `_box` нужен, потому что query
может быть не box: triangle, polygon или frustum.

```rust
index.search(&query);
query.overlaps_box(candidate_box);
query.contains_box(subtree_box);
```

Иными словами: публичный action короткий, predicate layer точный.

### `Box2D::overlaps(other)` vs `Overlaps2D::overlaps_box(box)`

`Box2D::overlaps(other)` симметричен: box overlap box.

`Overlaps2D::overlaps_box(box)` не обязан быть симметричным в реализации и
семантике. Индекс спрашивает query geometry: "принимаешь ли ты этот AABB?".
Поэтому `_box` здесь делает контракт яснее, чем абстрактное `overlaps`.

То же самое для 3D:

```rust
Box3D::overlaps(other_box);
query.overlaps_box(candidate_box);
```

### 2D vs 3D различаются типами, не именами методов

Методы остаются одинаковыми:

```rust
index2d.search(Box2D::new(...));
index2d.search(&triangle);
index2d.search(&polygon);

index3d.search(Box3D::new(...));
index3d.search(&frustum);
```

Размерность выражена типами:

- `Index2D` / `Index3D`
- `Box2D` / `Box3D`
- `Overlaps2D` / `Overlaps3D`
- скрытые dispatch traits `SearchQuery2D` / `SearchQuery3D`

В имени метода не появляется `2d`, `3d`, `circle`, `sphere`, `polygon` или
`frustum`, потому что shape-specific смысл уже живет в типе query.

### `Ray2D::intersects_box` vs `Overlaps2D::overlaps_box`

Raycast остается отдельным словарем.

`Ray2D::intersects_box` / `Ray3D::intersects_box` отвечают на вопрос:
пересекает ли луч AABB вдоль своего параметра. Для raycast слово `intersects`
естественно: результат связан с hit distance и traversal по лучу.

`Overlaps2D::overlaps_box` / `Overlaps3D::overlaps_box` отвечают на другой
вопрос: должна ли region query принять AABB как overlap candidate. Это
broad-phase overlap, не ray hit.

Переименовывать ray в `overlaps_box` ради симметрии было бы хуже: луч не
"overlaps" box как область.

## Почему не `search_overlaps`

`search_overlaps` точнее проговаривает relation, но в core он проигрывает как
публичная форма:

- рядом уже есть короткие `search`, `any`, `first`, `visit`;
- box query и region query тогда получают разные имена при одинаковой операции;
- `search_overlaps(Box2D)` дублирует `search(Box2D)`;
- пользователю приходится помнить, когда нужен short API, а когда generic API.

После того как `search(Box2D)` и `search(&triangle)` dispatch-ятся через одну
короткую семью, отдельный `search_overlaps` больше не несет полезной роли.

## Почему не shape-specific methods

Старые named conveniences удалены:

- `search_triangle`
- `search_polygon`
- `search_frustum`
- соответствующие `*_into`, `any_*`, `visit_*`

Они почти не экономят символы, но создают вторую систему имен:

```rust
index.search(&triangle);
index.search(&polygon);
index.search(&frustum);
```

Так короче и согласованнее: форма запроса стабильная, shape-specific смысл живет
в query type и его документации.

## Почему не `search_intersects`

`intersects` звучит математически шире, но в core оно хуже связывается с уже
существующим box API и predicate layer:

- `Box2D::overlaps`
- `Box3D::overlaps`
- `Triangle2D::overlaps_box`
- `ConvexPolygon2D::overlaps_box`
- `Frustum3D::overlaps_box`

Если публичный метод назвать `search_intersects`, получится разрыв:

```rust
index.search_intersects(&query); // публично "intersects"
query.overlaps_box(bbox);        // контракт реально "overlaps"
```

Для core broad-phase лучше одно relation word: **overlap**. Но в публичном
action-method оно опущено ради короткой универсальной формы `search`.

## Рабочая формула

В core:

- public action API: `search`, `search_into`, `search_with`, `search_iter`,
  `any`, `first`, `visit`;
- region predicate API: `Overlaps2D` / `Overlaps3D`, `overlaps_box`,
  `contains_box`;
- raycast API: `raycast`, `visit_raycast`, `intersects_box`.

