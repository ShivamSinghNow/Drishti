const MAX_BYTES = 12 * 1024 * 1024;

export const config = {
  api: {
    bodyParser: false,
  },
};

async function readRawBody(req) {
  const chunks = [];
  let total = 0;

  for await (const chunk of req) {
    total += chunk.length;
    if (total > MAX_BYTES) {
      throw new Error("Image must be smaller than 12 MB.");
    }
    chunks.push(chunk);
  }

  return Buffer.concat(chunks);
}

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }

  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const modalUrl = process.env.DRISHTI_MODAL_API_URL || process.env.MODAL_API_URL;
  if (!modalUrl) {
    res.status(503).json({
      error: "Demo API is not configured. Set DRISHTI_MODAL_API_URL in Vercel.",
    });
    return;
  }

  try {
    const body = await readRawBody(req);
    const upstream = await fetch(`${modalUrl.replace(/\/$/, "")}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": req.headers["content-type"] || "application/octet-stream",
      },
      body,
    });

    const payload = await upstream.text();
    res.status(upstream.status);
    res.setHeader("Content-Type", upstream.headers.get("content-type") || "application/json");
    res.setHeader("Cache-Control", "no-store");
    res.end(payload);
  } catch (error) {
    res.status(502).json({
      error: error instanceof Error ? error.message : "Failed to reach inference service.",
    });
  }
}
