import type { Context, Config } from "@netlify/functions";
import { getStore } from "@netlify/blobs";

// Read-only: what has been uploaded so far, newest first. No token check --
// site-wide password protection would be the right thing to gate this with,
// but that Netlify feature is not available on this account's plan (both
// attempts to enable it returned 422). Left open deliberately rather than
// silently unprotected: the URL is not published anywhere, and the data
// behind it is hull dimensions for a class project, not anything sensitive.
// If that changes, add the same token check upload.mts uses here too.
export default async (req: Request, context: Context) => {
  const store = getStore("results", { consistency: "strong" });
  const index: any[] = (await store.get("_index", { type: "json" })) || [];
  index.sort((a, b) => (a.uploadedAt < b.uploadedAt ? 1 : -1));
  return new Response(JSON.stringify(index), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
};

export const config: Config = {
  path: "/api/list",
};
