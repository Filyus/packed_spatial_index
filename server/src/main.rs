use std::{net::SocketAddr, path::PathBuf};

use clap::Parser;
use packed_spatial_index_server::{Catalog, ServerState, serve_with_cors};

/// Run a local native PSINDEX artifact server.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Path to the TOML catalog.
    #[arg(short, long, default_value = "psindex-server.toml")]
    catalog: PathBuf,
    /// Override the catalog bind address.
    #[arg(long)]
    addr: Option<SocketAddr>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let catalog = Catalog::from_path(&args.catalog)?;
    let addr = args.addr.unwrap_or(catalog.server.addr);
    let cors = catalog.server.cors.clone();
    let state = ServerState::from_catalog(catalog)?;
    if !addr.ip().is_loopback() {
        // A legitimate thing to want for a LAN demo, so this warns rather than
        // refusing -- but the server has no authentication, and every
        // configured artifact is readable by anyone who can reach the port.
        tracing::warn!(
            %addr,
            "binding a non-loopback address; this server has no authentication"
        );
    }
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "starting PSINDEX server");
    serve_with_cors(listener, state, &cors).await?;
    Ok(())
}
