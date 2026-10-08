import { Router } from "express";
import { z } from "zod";

const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || "http://localhost:8000";

const QuerySchema = z.object({
  question: z.string().trim().min(3).max(500),
});

const router = Router();

// This layer stays thin on purpose: validate the request shape, forward to
// the Python RAG core (retrieval + rerank + generation live there, since
// that ecosystem is Python-first), and pass the response through. No
// business logic duplicated here.
router.post("/", async (req, res, next) => {
  const parsed = QuerySchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: parsed.error.flatten() });
  }

  try {
    const upstream = await fetch(`${PYTHON_SERVICE_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: parsed.data.question }),
      // Generous: answers come from a local LLM, which can take 20s+ on a
      // cold start.
      signal: AbortSignal.timeout(120_000),
    });

    const body = await upstream.json();
    if (!upstream.ok) {
      return res.status(upstream.status).json(body);
    }
    res.json(body);
  } catch (err) {
    next(err);
  }
});

export default router;
