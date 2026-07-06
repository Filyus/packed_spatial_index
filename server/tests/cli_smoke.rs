use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use packed_spatial_index_geo::{
    ConvertRequest, PayloadPlan, PropertyProjection, open_geojson_slice,
};
use tempfile::tempdir;

fn sample_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "west",
                "geometry": {"type": "Point", "coordinates": [-5.0, 1.0]},
                "properties": {"name": "west", "kind": "sample"}
            },
            {
                "type": "Feature",
                "id": "east",
                "geometry": {"type": "Point", "coordinates": [25.0, 3.0]},
                "properties": {"name": "east", "kind": "sample"}
            }
        ]
    }"#
}

fn write_artifact(path: &Path, payload: PayloadPlan) {
    let mut source = open_geojson_slice(sample_geojson()).unwrap();
    let bytes = source
        .convert(ConvertRequest {
            payload,
            ..ConvertRequest::default()
        })
        .unwrap();
    fs::write(path, bytes).unwrap();
}

struct ServerProcess {
    child: Child,
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn free_port() -> u16 {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.local_addr().unwrap().port()
}

fn get(port: u16, path: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    let status = response
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok())
        .unwrap();
    (status, response)
}

fn start_server(catalog: &Path, port: u16) -> ServerProcess {
    let child = Command::new(env!("CARGO_BIN_EXE_psindex-server"))
        .arg("--catalog")
        .arg(catalog)
        .arg("--addr")
        .arg(format!("127.0.0.1:{port}"))
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut process = ServerProcess { child };
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(status) = process.child.try_wait().unwrap() {
            let mut stderr = String::new();
            if let Some(pipe) = process.child.stderr.as_mut() {
                let _ = pipe.read_to_string(&mut stderr);
            }
            panic!("server exited early with {status}: {stderr}");
        }
        if let Ok((200, body)) = std::panic::catch_unwind(|| get(port, "/health"))
            && body.contains("\"status\":\"ok\"")
        {
            return process;
        }
        assert!(
            Instant::now() < deadline,
            "server did not become ready on port {port}"
        );
        thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn binary_serves_generated_local_artifacts() {
    let dir = tempdir().unwrap();
    let data_dir = dir.path().join("data");
    fs::create_dir(&data_dir).unwrap();
    write_artifact(
        &data_dir.join("places-feature-json.psindex"),
        PayloadPlan::FeatureJson {
            properties: PropertyProjection::AllNonGeometry,
        },
    );
    write_artifact(
        &data_dir.join("places-row-wkb.psindex"),
        PayloadPlan::RowWkb,
    );

    let catalog = dir.path().join("catalog.toml");
    fs::write(
        &catalog,
        r#"
        [[collections]]
        id = "places_json"
        title = "Places JSON"
        artifact = "data/places-feature-json.psindex"

        [[collections]]
        id = "places_wkb"
        title = "Places WKB"
        artifact = "data/places-row-wkb.psindex"
        "#,
    )
    .unwrap();

    let port = free_port();
    let _server = start_server(&catalog, port);

    let (status, body) = get(port, "/collections");
    assert_eq!(status, 200);
    assert!(body.contains("\"places_json\""));
    assert!(body.contains("\"places_wkb\""));

    let (status, body) = get(
        port,
        "/collections/places_json/items?bbox=-10,0,0,2&limit=10",
    );
    assert_eq!(status, 200);
    assert!(body.contains("\"FeatureCollection\""));
    assert!(body.contains("\"west\""));

    let (status, body) = get(
        port,
        "/collections/places_wkb/hits?bbox=-10,0,0,2&payload=full",
    );
    assert_eq!(status, 200);
    assert!(body.contains("\"kind\":\"row_wkb\""));
    assert!(body.contains("\"wkbBase64\""));

    let (status, _) = get(port, "/collections/places_wkb/items?bbox=-10,0,0,2");
    assert_eq!(status, 422);

    let (status, _) = get(port, "/collections/places_json/hits?bbox=10,0,0,2");
    assert_eq!(status, 400);
}
