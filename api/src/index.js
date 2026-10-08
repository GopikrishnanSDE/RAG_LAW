import "dotenv/config";
import path from "node:path";
import { fileURLToPath } from "node:url";

import cors from "cors";
import express from "express";
import { rateLimit } from "express-rate-limit";

import { errorHandler } from "./middleware/errorHandler.js";
import queryRouter from "./routes/query.js";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
// Chat UI. Served before the rate limiter so loading the page doesn't
// count against the per-minute query budget.
app.use(express.static(path.join(path.dirname(fileURLToPath(import.meta.url)), "../public")));
app.use(express.json({ limit: "10kb" }));
app.use(
  rateLimit({
    windowMs: 60 * 1000,
    limit: 20,
    standardHeaders: true,
    legacyHeaders: false,
  })
);

app.get("/health", (_req, res) => res.json({ status: "ok" }));
app.use("/api/query", queryRouter);

app.use(errorHandler);

app.listen(PORT, () => {
  console.log(`RAG-Law API listening on port ${PORT}`);
});
