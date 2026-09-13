import type { Context, Config } from "@netlify/functions";
import { getStore } from "@netlify/blobs";

// Hands each guest computer the next unused seed, so no two machines
// repeat each other's search. Starts above the seeds Levi assigned by hand
// (5 and 50-55). Same shared token as uploads, so random traffic can't
// burn through numbers.
const FIRST_SEED = 60;

export default async (req: Request, context: Context) => {
  if (req.method !== "POST") {
    return new Response("POST only", { status: 405 });
  }
  const token = req.headers.get("x-upload-token");
  if (!token || token !== Netlify.env.get("UPLOAD_TOKEN")) {
    return new Response("unauthorized", { status: 401 });
  }

  let name = "unknown";
  try {
    name = String((await req.json()).name || "unknown").slice(0, 40);
  } catch {}

  const store = getStore("results", { consistency: "strong" });

  // Read-then-write with an etag check: if two computers claim at the same
  // moment, the second write is rejected and it retries with the new count,
  // instead of both walking away with the same seed.
  for (let attempt = 0; attempt < 10; attempt++) {
    const current = await store.getWithMetadata("_seeds", { type: "json" });
    const data = (current?.data as any) ?? { next: FIRST_SEED, claims: [] };
    const seed: number = data.next;
    const updated = {
      next: seed + 1,
      claims: [...data.claims, { seed, name, claimedAt: new Date().toISOString() }],
    };
    const result: any = current
      ? await store.setJSON("_seeds", updated, { onlyIfMatch: current.etag })
      : await store.setJSON("_seeds", updated, { onlyIfNew: true });
    if (!result || result.modified !== false) {
      return Response.json({ ok: true, seed });
    }
    await new Promise((r) => setTimeout(r, 50 + Math.random() * 150));
  }
  return new Response("busy -- try again", { status: 503 });
};

export const config: Config = {
  path: "/api/seed",
};
