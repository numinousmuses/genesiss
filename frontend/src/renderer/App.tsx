import { useEffect, useState } from "react";
import { PromptInput } from "./components/PromptInput";
import { CodeView } from "./components/CodeView";
import { ModelViewer } from "./components/ModelViewer";
import { ModelPicker } from "./components/ModelPicker";
import { execCode, generate, listModels, type Variant } from "./lib/api";

export function App() {
  const [variants, setVariants] = useState<Variant[]>([]);
  const [model, setModel] = useState<string>("");
  const [code, setCode] = useState<string>("# Describe a part and hit Generate.\n");
  const [stlB64, setStlB64] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listModels()
      .then((r) => {
        setVariants(r.variants);
        setModel(r.default);
      })
      .catch((e) => setError(String(e)));
  }, []);

  async function onSubmit(prompt: string) {
    setError(null);
    setBusy(true);
    try {
      const g = await generate(prompt, model);
      setCode(g.code);
      const ex = await execCode(g.code);
      if (!ex.ok) {
        setError(ex.error ?? "exec failed");
      } else {
        setStlB64(ex.stl_b64 ?? null);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onRunCode() {
    setError(null);
    setBusy(true);
    try {
      const ex = await execCode(code);
      if (!ex.ok) setError(ex.error ?? "exec failed");
      else setStlB64(ex.stl_b64 ?? null);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <h1>Genesiss</h1>
        <ModelPicker variants={variants} value={model} onChange={setModel} />
        <div style={{ marginLeft: "auto", color: "var(--muted)", fontSize: 12 }}>
          {busy ? "working…" : "idle"}
        </div>
      </header>

      <main className="workspace">
        <section className="pane">
          <div className="pane-header">
            <span>CadQuery</span>
            <button onClick={onRunCode} disabled={busy}>
              Run
            </button>
          </div>
          <div className="code">
            <CodeView value={code} onChange={setCode} />
          </div>
          {error && <div className="error">{error}</div>}
          <PromptInput disabled={busy} onSubmit={onSubmit} />
        </section>

        <section className="pane">
          <div className="pane-header">3D preview</div>
          <div className="viewer">
            <ModelViewer stlB64={stlB64} />
          </div>
        </section>
      </main>
    </div>
  );
}
