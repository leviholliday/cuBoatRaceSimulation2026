import type { Context, Config } from "@netlify/functions";
import { getStore } from "@netlify/blobs";

// Streams one uploaded zip back down. Keyed by the exact key list.mts
// handed out -- there is nothing to enumerate or guess your way into.
export default async (req: Request, context: Context) => {
  const url = new URL(req.url);
  const key = url.searchParams.get("key");
  if (!key) {
    return new Response("missing key", { status: 400 });
  }

  const store = getStore("results", { consistency: "strong" });
  const data = await store.get(key, { type: "arrayBuffer" });
  if (!data) {
    return new Response("not found", { status: 404 });
  }

  const filename = key.split("/").pop() || "download.zip";
  return new Response(data, {
    status: 200,
    headers: {
      "content-type": "application/zip",
      "content-disposition": `attachment; filename="${filename}"`,
    },
  });
};

export const config: Config = {
  path: "/api/download",
};
