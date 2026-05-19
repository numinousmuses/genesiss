const bridge = window.genesiss?.bridgeUrl ?? "http://127.0.0.1:8765";

export interface Variant {
  name: string;
  base: string;
  hf_repo: string;
  ollama_tag: string;
  context: number;
}

export interface ExecResult {
  ok: boolean;
  error?: string | null;
  traceback?: string | null;
  stl_b64?: string | null;
  step_b64?: string | null;
  log: string;
}

export async function listModels(): Promise<{ variants: Variant[]; default: string }> {
  const r = await fetch(`${bridge}/models`);
  if (!r.ok) throw new Error(`models: ${r.status}`);
  return r.json();
}

export async function generate(prompt: string, model: string): Promise<{ code: string; model: string }> {
  const r = await fetch(`${bridge}/generate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt, model }),
  });
  if (!r.ok) throw new Error(`generate: ${r.status} ${await r.text()}`);
  return r.json();
}

export async function execCode(code: string): Promise<ExecResult> {
  const r = await fetch(`${bridge}/exec`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ code, export_format: "stl" }),
  });
  if (!r.ok) throw new Error(`exec: ${r.status} ${await r.text()}`);
  return r.json();
}

export function streamGenerate(
  prompt: string,
  model: string,
  onToken: (t: string) => void,
  onDone: () => void,
  onError: (e: string) => void,
): WebSocket {
  const wsUrl = bridge.replace(/^http/, "ws") + "/ws/generate";
  const ws = new WebSocket(wsUrl);
  ws.onopen = () => ws.send(JSON.stringify({ prompt, model }));
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "token") onToken(m.text);
    else if (m.type === "done") onDone();
    else if (m.type === "error") onError(m.message);
  };
  ws.onerror = () => onError("websocket error");
  return ws;
}
