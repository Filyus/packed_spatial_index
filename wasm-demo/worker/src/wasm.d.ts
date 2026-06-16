// Cloudflare module workers import a `.wasm` file as a compiled module.
declare module "*.wasm" {
  const mod: WebAssembly.Module;
  export default mod;
}
