import type { Context, Config } from "@netlify/functions";
import { getStore } from "@netlify/blobs";

// One machine's simulator output, zipped and POSTed here from a terminal
// (see scripts/upload_results.py). Guarded by a shared token so random
// internet traffic cannot fill the storage or overwrite results -- this is
// the one endpoint here that actually writes anything, so it is the one
// worth guarding even though the data itself (hull dimensions for a class
// project) is not sensitive.
//
// A real overnight run's out/mc easily runs tens of MB, well past what a
// standard Netlify Function can accept in one request body (it runs on
// Lambda under the hood, which caps this around 6 MB -- a bigger request
// gets its connection cut mid-upload, seen client-side as a broken pipe).
// The client (upload_results.py) splits the zip into chunks under that
// limit instead; each chunk lands here as its own small request, stored as
// its own temporary blob, and the final chunk triggers reassembly -- which
// reads those blobs back out of storage rather than over the wire, so it
// is not bound by the same request-size limit.
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
  const uploadId = (req.headers.get("x-upload-id") || "single")
    .replace(/[^a-zA-Z0-9_-]/g, "-")
    .slice(0, 64);
  const chunkIndex = parseInt(req.headers.get("x-chunk-index") || "0", 10);
  const chunkTotal = parseInt(req.headers.get("x-chunk-total") || "1", 10);

  const body = await req.arrayBuffer();
  if (body.byteLength === 0) {
    return new Response("empty body", { status: 400 });
  }

  const store = getStore("results", { consistency: "strong" });
  const chunkKey = `_chunks/${tag}/${uploadId}/${chunkIndex}`;
  await store.set(chunkKey, body);

  if (chunkIndex + 1 < chunkTotal) {
    return new Response(JSON.stringify({ ok: true, chunk: chunkIndex, of: chunkTotal }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  // Last chunk in: read every piece back and reassemble in order. These
  // reads come from Blobs storage, not the incoming request, so the
  // combined size here is not limited by the per-request cap above.
  const parts: ArrayBuffer[] = [];
  let totalSize = 0;
  for (let i = 0; i < chunkTotal; i++) {
    const part = await store.get(`_chunks/${tag}/${uploadId}/${i}`, { type: "arrayBuffer" });
    if (!part) {
      return new Response(`missing chunk ${i} of ${chunkTotal} -- retry the upload`, { status: 409 });
    }
    parts.push(part);
    totalSize += part.byteLength;
  }
  const combined = new Uint8Array(totalSize);
  let offset = 0;
  for (const part of parts) {
    combined.set(new Uint8Array(part), offset);
    offset += part.byteLength;
  }

  const uploadedAt = new Date().toISOString();
  const key = `${tag}/${uploadedAt.replace(/[:.]/g, "-")}-${filename}`;
  await store.set(key, combined, {
    metadata: { tag, filename, size: combined.byteLength, uploadedAt },
  });

  // Clean up the temporary chunk blobs now that the combined one exists.
  await Promise.all(
    Array.from({ length: chunkTotal }, (_, i) => store.delete(`_chunks/${tag}/${uploadId}/${i}`))
  );

  // A small hand-maintained index, so the list page does not need to call
  // getMetadata() once per blob. Blob writes have no locking (last write
  // wins), so two uploads landing in the exact same instant could in theory
  // clobber each other's index update -- the blob data itself is never at
  // risk either way, only its entry in this list, and that is an acceptable
  // trade for four machines uploading a few times a day rather than a
  // high-traffic service.
  const existing = (await store.get("_index", { type: "json" })) || [];
  existing.push({ key, tag, filename, size: combined.byteLength, uploadedAt });
  await store.setJSON("_index", existing);

  return new Response(JSON.stringify({ ok: true, key }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
};

export const config: Config = {
  path: "/api/upload",
};
