# Нейминг запросов в core

Заметка фиксирует решение по словам `search`, `overlap` и `intersect` внутри
core API. Это не документ про `geo`: внешний GIS-словарь не должен определять
форму core.

## Принятый словарь

Core AABB hot path остается коротким:

- `search(Box2D)` / `search(Box3D)`
- `search_into`
- `search_with`
- `any`
- `first`
- `visit`

Generic region API говорит через `overlap`:

- `search_overlaps`
- `search_overlaps_into`
- `any_overlaps`
- `visit_overlaps`
- `Overlaps2D` / `Overlaps3D`
- `overlaps_box`
- `contains_box`

Shape-specific helpers остаются именованными:

- `search_triangle`
- `search_polygon`
- `search_frustum`

Raycast остается отдельным словарем:

- `raycast`
- `visit_raycast`
- `Ray2D::intersects_box`
- `Ray3D::intersects_box`

## Где core API несимметричен

### `search(Box2D)` vs `search_overlaps(&query)`

Это намеренная несимметрия.

`search(Box2D)` - главный primitive индекса. Он короткий, привычный и находится
на hot path. Полное имя вроде `search_overlapping_box` или `search_box_overlaps`
сделало бы самый частый вызов тяжелее без выигрыша в ясности.

`search_overlaps(&query)` - generic extension point для query geometry. Здесь
слово `overlaps` полезно, потому что вызывающий передает не box query напрямую,
а объект, реализующий overlap-контракт:

```rust
index.search(Box2D::new(...));

index.search_overlaps(&triangle);
index.search_overlaps(&polygon);
index.search_overlaps(&frustum);
```

То есть короткое имя оставлено за базовым AABB query, а явное имя - за generic
region query.

### `search_overlaps` vs `search_triangle` / `search_polygon` / `search_frustum`

Это тоже намеренная несимметрия.

`search_overlaps` - общий механизм для любого типа, который реализует
`Overlaps2D` или `Overlaps3D`.

Именованные методы - не просто aliases. Они дают discoverability и фиксируют
shape-specific смысл:

- `search_triangle` - точный 2D triangle-vs-box SAT region query.
- `search_polygon` - выпуклый polygon-vs-box region query.
- `search_frustum` - conservative 3D frustum culling query.

Их стоит держать рядом с generic API, потому что пользователь часто ищет
конкретную форму, а не trait extension point.

### `Overlaps2D::overlaps_box` vs `Box2D::overlaps`

Здесь есть небольшая асимметрия имени, но она полезная.

`Box2D::overlaps(other)` симметричен: box overlap box.

`Overlaps2D::overlaps_box(bx)` не обязательно симметричен в реализации и
семантике. Query type может быть triangle, polygon или frustum; индекс
спрашивает у него: "принимаешь ли ты этот AABB?". Поэтому suffix `_box` полезен:
он явно называет primitive, против которого тестируется query geometry.

### `search_overlaps` vs `overlaps_box`

Эта пара теперь симметрична по смыслу:

```rust
index.search_overlaps(&query);
query.overlaps_box(node_or_item_box);
```

Именно это было причиной не оставлять форму `search_intersects(&query)` с
внутренним контрактом `overlaps_box`: она создавала бы два слова для одной и той
же broad-phase операции.

### `Ray2D::intersects_box` vs `Overlaps2D::overlaps_box`

Raycast - отдельная операция, не region overlap.

`Ray2D::intersects_box` отвечает на вопрос: пересекает ли луч AABB вдоль своего
параметра. Для raycast слово `intersects` естественно: результат связан с hit
distance и traversal по лучу.

`Overlaps2D::overlaps_box` отвечает на другой вопрос: должна ли region query
принять AABB как candidate/overlap. Это broad-phase overlap, не ray hit.

Переименовывать ray в `overlaps_box` ради симметрии было бы хуже: луч не
"overlaps" box как область.

## Почему не `search_intersects`

`intersects` звучит математически шире, но в core оно хуже связывается с уже
существующим box API:

- `Box2D::overlaps`
- `Box3D::overlaps`
- `Triangle2D::overlaps_box`
- `ConvexPolygon2D::overlaps_box`
- `Frustum3D::overlaps_box`

Если generic API назвать `search_intersects`, получится разрыв:

```rust
index.search_intersects(&query); // публично "intersects"
query.overlaps_box(bbox);        // контракт реально "overlaps"
```

Для core broad-phase лучше одно слово: **overlap**.

## Что не делаем

Не переименовываем `search(Box2D)` в более длинное имя: это главный primitive.

Не удаляем `search_triangle`, `search_polygon`, `search_frustum`: generic API не
заменяет discoverability и shape-specific документацию.

Не переименовываем `Ray2D::intersects_box` / `Ray3D::intersects_box`: для raycast
это правильный термин.

## Рабочая формула

В core:

- box/range/region traversal говорит на языке **overlap**;
- raycast говорит на языке **intersect**;
- короткое `search` зарезервировано за самым частым AABB query.
