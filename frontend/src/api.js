const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function queryAnswer(question, { temperature, maxTokens, history, signal } = {}) {
  const body = {
    question,
    return_context: true,
    history: history || [],
  };
  if (temperature !== undefined) body.temperature = temperature;
  if (maxTokens !== undefined) body.max_tokens = maxTokens;

  const res = await fetch(`${API_BASE}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `HTTP ${res.status}`);
  }

  return res.json();
}
