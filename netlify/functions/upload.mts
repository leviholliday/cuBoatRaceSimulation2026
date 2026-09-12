import type { Context, Config } from "@netlify/functions";
import { getStore } from "@netlify/blobs";

// One machine's simulator output, zipped and POSTed here from a terminal
// (see scripts/upload_results.py). Guarded by a shared token so random
// internet traffic cannot fill the storage or overwrite results -- this is
// the one endpoint here that actually writes anything, so it is the one
// worth guarding even though the data itself (hull dimensions for a class
// project) is not sensitive.
export default async (req: Request, context: Context) => {
  if (req.method !== "POST") {
    return new Response("POST only", { status: 405 });
  }

  const token = req.headers.get("x-upload-token");
  if (!token || token !== Netlify.env.get("UPLOAD_TOKEN")) {
    return new Response("unauthorized", { status: 401 });
  }

  const tag = (req.headers.get("x-machine-tag") || "unknown")
    .replace(/[^a-zA-Z0-9_-]/g, "-")
    .slice(0, 40);
  const filename = (req.headers.get("x-filename") || "results.zip")
    .replace(/[^a-zA-Z0-9_.-]/g, "-")
    .slice(0, 100);

  const body = await req.arrayBuffer();
  if (body.byteLength === 0) {
    return new Response("empty body", { status: 400 });
  }

  const uploadedAt = new Date().toISOString();
  const key = `${tag}/${uploadedAt.replace(/[:.]/g, "-")}-${filename}`;

  const store = getStore("results", { consistency: "strong" });
  await store.set(key, body, {
    metadata: { tag, filename, size: body.byteLength, uploadedAt },
  });

  // A small hand-maintained index, so the list page does not need to call
  // getMetadata() once per blob. Blob writes have no locking (last write
  // wins), so two uploads landing in the exact same instant could in theory
  // clobber each other's index update -- the blob data itself is never at
  // risk either way, only its entry in this list, and that is an acceptable
  // trade for four machines uploading a few times a day rather than a
  // high-traffic service.
  const existing = (await store.get("_index", { type: "json" })) || [];
  existing.push({ key, tag, filename, size: body.byteLength, uploadedAt });
  await store.setJSON("_index", existing);

  return new Response(JSON.stringify({ ok: true, key }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
};

export const config: Config = {
  path: "/api/upload",
};
