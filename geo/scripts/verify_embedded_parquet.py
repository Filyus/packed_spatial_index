#!/usr/bin/env python3
"""Cross-check the geoparquet#279 prototype with a second, independent reader.

`examples/embedded_in_parquet.rs` proves the spliced file is readable with the
Rust `parquet` crate. This script confirms the same file is readable with
pyarrow (the C++ Arrow Parquet reader the GeoParquet Python stack uses), so the
"transparent to existing tooling" claim does not rest on a single engine.

Dev / CI verification only — it is excluded from the published crate (see the
`exclude` in geo/Cargo.toml) and is not part of the library.

Usage:
    python geo/scripts/verify_embedded_parquet.py
Requires: pyarrow (`pip install pyarrow`) and a workspace that can `cargo run`.
Exit code 0 on success, non-zero if any check fails.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pyarrow not installed — run `pip install pyarrow`")

GEO = Path(__file__).resolve().parents[1]  # geo/scripts -> geo
EXPECTED_ROWS = 100_000


def main() -> int:
    # ignore_cleanup_errors: on Windows pyarrow may still hold the file handle
    # when the temp dir is torn down (WinError 32); the checks have run by then.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out = Path(tmp) / "embedded.parquet"
        # Produce the index-embedded GeoParquet via the Rust example.
        subprocess.run(
            ["cargo", "run", "--release", "--example", "embedded_in_parquet", "--", str(out)],
            cwd=GEO,
            check=True,
        )

        pf = pq.ParquetFile(str(out))
        rows = pf.metadata.num_rows
        table = pf.read()  # full materialize — must not choke on the embedded bytes
        geo = pf.metadata.metadata.get(b"geo")

        ok = True

        def check(label: str, cond: bool) -> None:
            nonlocal ok
            ok = ok and cond
            print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

        print(f"\npyarrow {pa.__version__} cross-check of the embedded-index GeoParquet:")
        check(f"footer num_rows == {EXPECTED_ROWS}", rows == EXPECTED_ROWS)
        check(f"read() materialized {EXPECTED_ROWS} rows", table.num_rows == EXPECTED_ROWS)
        check("columns == ['geometry', 'name']", table.column_names == ["geometry", "name"])
        check("geo metadata preserved", geo is not None)
        if geo is not None:
            check("geo.primary_column == 'geometry'",
                  json.loads(geo).get("primary_column") == "geometry")

        print("RESULT:", "all checks passed" if ok else "FAILED")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
