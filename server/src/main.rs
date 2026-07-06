use std::{net::SocketAddr, path::PathBuf};

use clap::Parser;
use packed_spatial_index_server::{Catalog, ServerState, serve};

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
    let state = ServerState::from_catalog(catalog)?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "starting PSINDEX server");
    serve(listener, state).await?;
    Ok(())
}
