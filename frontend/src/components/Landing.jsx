import { useState, useEffect, useRef } from 'react';
import { Icon } from './Icons';
import { CategoryGlyph } from './Badge';
import { categories } from './data';

export function Landing({ onSubmit, onPickCategory }) {
  const [val, setVal] = useState("");
  const ref = useRef(null);
  useEffect(() => { ref.current?.focus(); }, []);
  const submit = (e) => {
    e.preventDefault();
    if (val.trim()) onSubmit(val.trim());
  };
  return (
    <div className="landing">
      <div className="landing-inner">
        <div className="landing-hero">
          <div className="landing-mark">
            <span className="brand-mark">A</span>
          </div>
          <h1>Atlas <em>workbench</em></h1>
          <p>A GraphRAG interface for Alzheimer's disease research. Ask about biomarkers, drugs, genes, phenotypes, or pathways — get grounded answers with traceable evidence.</p>
        </div>

        <form className="landing-input" onSubmit={submit}>
          <textarea
            ref={ref}
            value={val}
            onChange={(e) => setVal(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(e); }
            }}
            placeholder="Ask about Alzheimer's biomarkers, drugs, genes, phenotypes, pathways..."
            rows={2}
          />
          <div className="landing-input-row">
            <div className="input-hint">Shift+Enter for newline · Enter to submit</div>
            <button type="submit" className="btn-send" disabled={!val.trim()}>
              <Icon.Send /> Submit
            </button>
          </div>
        </form>

        <div className="cat-grid">
          {categories.map(c => (
            <div key={c.id} className="cat-card">
              <div className="cat-head">
                <CategoryGlyph id={c.id} />
                <div>
                  <div className="cat-label">{c.label}</div>
                  <div className="cat-blurb">{c.blurb}</div>
                </div>
              </div>
              <div className="cat-prompts">
                {c.prompts.map((p, i) => (
                  <button key={i} className="cat-prompt" onClick={() => onPickCategory(p)}>
                    <span>{p}</span>
                    <Icon.Arrow />
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="landing-foot">
          <span className="landing-back">Alzheimer's Knowledge Graph RAG</span>
        </div>
      </div>
    </div>
  );
}
