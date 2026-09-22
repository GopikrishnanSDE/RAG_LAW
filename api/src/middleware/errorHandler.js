export function errorHandler(err, _req, res, _next) {
  const isTimeout = err.name === "TimeoutError" || err.name === "AbortError";
  console.error(err);
  res.status(isTimeout ? 504 : 502).json({
    error: isTimeout ? "Upstream RAG service timed out" : "Upstream RAG service error",
  });
}
