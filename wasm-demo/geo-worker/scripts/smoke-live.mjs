const base = (process.argv[2] || process.env.WORKER_URL || "").replace(/\/$/, "");

if (!base) {
  console.error("usage: npm run smoke:live -- https://psi-geo-r2-demo.<subdomain>.workers.dev");
  process.exit(2);
}

const bbox = "64,23,71,29";

async function get(path) {
  const res = await fetch(`${base}${path}`);
  const text = await res.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  if (!res.ok) {
    throw new Error(`${path} -> ${res.status}: ${text}`);
  }
  return {
    body,
    reads: res.headers.get("x-psi-reads"),
    bytes: res.headers.get("x-psi-bytes"),
  };
}

const health = await get("/health");
if (health.body.status !== "ok") {
  throw new Error(`/health returned ${JSON.stringify(health.body)}`);
}

const collections = await get("/collections");
if (!Array.isArray(collections.body) || !collections.body.some((c) => c.id === "cities")) {
  throw new Error(`/collections did not list cities: ${JSON.stringify(collections.body)}`);
}

const search = await get(`/collections/cities/search?bbox=${bbox}&limit=3&payload=summary`);
if (!search.body.numberReturned || !search.body.reads || !search.body.bytes || !search.reads || !search.bytes) {
  throw new Error(`/search response missed matches or counters: ${JSON.stringify(search.body)}`);
}

const items = await get(`/collections/cities/items?bbox=${bbox}&limit=3`);
if (items.body.type !== "FeatureCollection" || !items.body.features?.length || !items.reads || !items.bytes) {
  throw new Error(`/items did not return a FeatureCollection: ${JSON.stringify(items.body)}`);
}

console.log("health:", health.body);
console.log("collections:", collections.body.map((c) => c.id).join(", "));
console.log("search example:", JSON.stringify(search.body, null, 2));
console.log("items example:", JSON.stringify(items.body, null, 2));
