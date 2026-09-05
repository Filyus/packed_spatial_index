//! `query --pick`: the click's ordered broad phase as a CLI verb.

#![cfg(all(feature = "geojson", feature = "parquet"))]

fn grid_geojson() -> String {
    let points = [
        ("a", 0.0, 0.0, 0.0),
        ("b", 10.0, 0.0, 0.0),
        ("c", 20.0, 0.0, 0.0),
        ("d", 20.0, 0.0, 10.0),
    ];
    let features: Vec<String> = points
        .iter()
        .map(|(id, x, y, z)| {
            format!(
                r#"{{"type":"Feature","id":"{id}","geometry":{{"type":"Point","coordinates":[{x},{y},{z}]}},"properties":{{}}}}"#
            )
        })
        .collect();
    format!(
        r#"{{"type":"FeatureCollection","features":[{}]}}"#,
        features.join(",")
    )
}

#[test]
fn pick_orders_on_ray_first_near_to_far_and_limit_truncates() {
    let dir = std::env::temp_dir().join(format!(
        "psi_pick_cli_{}_{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("grid.geojson");
    let output = dir.join("grid.psi");
    std::fs::write(&input, grid_geojson()).unwrap();

    let bin = env!("CARGO_BIN_EXE_gp2psindex");
    let build = std::process::Command::new(bin)
        .arg("build")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "build failed\nstderr:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );

    let run = |args: &[&str]| {
        let out = std::process::Command::new(bin)
            .arg("query")
            .arg(&input)
            .arg(&output)
            .args(args)
            .output()
            .unwrap();
        (
            out.status,
            String::from_utf8_lossy(&out.stdout).to_string(),
            String::from_utf8_lossy(&out.stderr).to_string(),
        )
    };

    let (status, stdout, stderr) = run(&[
        "--pick",
        "-100,0,0,1,0,0",
        "--half-angle",
        "5",
        "--limit",
        "10",
    ]);
    assert!(status.success(), "pick failed\nstderr:\n{stderr}");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 4, "{stdout}");
    // The three points on the ray, near-to-far, then the graze 10 units off.
    assert!(
        lines[0].starts_with(r#"{"entry":0,"distanceSquared":0,"entryT":100}"#),
        "{stdout}"
    );
    assert!(
        lines[1].starts_with(r#"{"entry":1,"distanceSquared":0,"entryT":110}"#),
        "{stdout}"
    );
    assert!(
        lines[2].starts_with(r#"{"entry":2,"distanceSquared":0,"entryT":120}"#),
        "{stdout}"
    );
    assert!(
        lines[3].starts_with(r#"{"entry":3,"distanceSquared":100,"entryT":inf}"#),
        "{stdout}"
    );

    let (status, stdout, _) = run(&[
        "--pick",
        "-100,0,0,1,0,0",
        "--half-angle",
        "5",
        "--limit",
        "2",
    ]);
    assert!(status.success());
    assert_eq!(stdout.lines().count(), 2, "{stdout}");

    // A zero direction is refused, and --count is refused with --pick.
    let (status, _, stderr) = run(&["--pick", "-100,0,0,0,0,0", "--half-angle", "5"]);
    assert!(!status.success());
    assert!(stderr.contains("not be all zeros"), "{stderr}");
    let (status, _, stderr) = run(&["--pick", "-100,0,0,1,0,0", "--half-angle", "5", "--count"]);
    assert!(!status.success());
    assert!(stderr.contains("drop --count"), "{stderr}");

    let _ = std::fs::remove_dir_all(&dir);
}
