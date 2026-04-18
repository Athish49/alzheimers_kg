import { useState } from 'react';
import { Badge } from './Badge';
import { Icon } from './Icons';

// ——— BIOMARKER ———
export function BiomarkerView({ data }) {
  const fluids = Object.keys(data.biomarkers || {});
  const [fluid, setFluid] = useState(fluids[0] || "CSF");
  const current = data.biomarkers?.[fluid] || { increased: [], decreased: [] };
  const maxEffect = Math.max(
    ...[...current.increased, ...current.decreased].map(b => Math.abs(b.effect)),
    1
  );

  const Row = ({ b, dir }) => {
    const pct = Math.min(100, (Math.abs(b.effect) / maxEffect) * 100);
    return (
      <div className="ev-row">
        <div className="ev-row-main">
          <div className="ev-name">{b.name}</div>
          <div className="ev-meta">{b.class}</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div className="ev-bar"><span style={{
            width: `${pct}%`,
            background: dir === "up" ? "var(--up)" : "var(--down)",
          }} /></div>
          <div style={{
            fontFamily: "var(--font-mono)", fontSize: 12, minWidth: 44, textAlign: "right",
            color: dir === "up" ? "var(--up)" : "var(--down)", fontWeight: 600,
          }}>
            {dir === "up" ? "\u2191" : "\u2193"} {Math.abs(b.effect).toFixed(1)}
          </div>
        </div>
        <div className="ev-pvalue">p {b.p}</div>
      </div>
    );
  };

  return (
    <div className="ctx-section">
      <div className="tabs">
        {fluids.map(f => (
          <button key={f} className={"tab " + (fluid === f ? "active" : "")} onClick={() => setFluid(f)}>
            {f}
            <span className="tab-count">
              {((data.biomarkers[f]?.increased?.length || 0) + (data.biomarkers[f]?.decreased?.length || 0))}
            </span>
          </button>
        ))}
      </div>

      {current.increased.length > 0 && (
        <>
          <div className="ev-label">Increased in AD <span>({current.increased.length})</span></div>
          <div className="ev-list">
            {[...current.increased].sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect)).map((b, i) => <Row key={i} b={b} dir="up" />)}
          </div>
        </>
      )}
      {current.decreased.length > 0 && (
        <>
          <div className="ev-label" style={{ marginTop: 20 }}>Decreased in AD <span>({current.decreased.length})</span></div>
          <div className="ev-list">
            {[...current.decreased].sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect)).map((b, i) => <Row key={i} b={b} dir="down" />)}
          </div>
        </>
      )}
      {current.increased.length === 0 && current.decreased.length === 0 && (
        <div className="ev-empty">No biomarkers retrieved for {fluid}.</div>
      )}
    </div>
  );
}

// ——— DRUG_TRIAL ———
const statusOrder = ["Approved", "Phase 3", "Phase 1-2", "Discontinued"];

export function DrugView({ data }) {
  const [expanded, setExpanded] = useState(null);
  const drugs = [...(data.drugs || [])].sort((a, b) =>
    statusOrder.indexOf(a.status) - statusOrder.indexOf(b.status)
  );
  const counts = statusOrder.map(s => ({ s, n: drugs.filter(d => d.status === s).length }));

  return (
    <div className="ctx-section">
      <div className="status-bar">
        {counts.map(c => (
          <div key={c.s} className={"status-chip " + (c.n === 0 ? "dim" : "")}>
            <span className={"status-dot status-" + c.s.replace(/\s+/g, "-").replace(/[^a-zA-Z0-9-]/g, "")} />
            <span>{c.s}</span>
            <span className="status-n">{c.n}</span>
          </div>
        ))}
      </div>

      <div className="ev-list" style={{ marginTop: 16 }}>
        {drugs.map((d, i) => {
          const open = expanded === i;
          const dim = d.status === "Discontinued";
          return (
            <div key={i} className={"drug-card " + (dim ? "dim" : "")}>
              <button className="drug-head" onClick={() => setExpanded(open ? null : i)}>
                <div style={{ flex: 1, textAlign: "left" }}>
                  <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
                    <div className="drug-name">{d.name}</div>
                    <div className="ev-meta">{d.chebi}</div>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6, flexWrap: "wrap" }}>
                    <Badge variant={d.status === "Approved" ? "intent" : d.status === "Discontinued" ? "muted" : "outline"}>
                      {d.phase}
                    </Badge>
                    <span className="ev-meta" style={{ fontFamily: "var(--font-sans)" }}>{d.type}</span>
                    <span className="ev-meta">&middot; {d.trials} trials</span>
                  </div>
                </div>
                <span style={{ color: "var(--fg-subtle)", transform: open ? "rotate(90deg)" : "none", transition: "transform 150ms" }}>
                  <Icon.ChevRight />
                </span>
              </button>
              {open && d.pathways && (
                <div className="drug-body">
                  <div className="ev-label small">Pathway targets</div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    {d.pathways.map((p, j) => (
                      <div key={j} className="pw-row">
                        <div className="pw-name">
                          {p.name}
                          {p.primary && <Badge variant="accent" style={{ marginLeft: 8 }}>primary</Badge>}
                        </div>
                        <div className="ev-meta">{p.go}</div>
                        <Badge variant="outline">{p.action}</Badge>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ——— PHENOTYPE ———
const freqWeight = { "Very frequent": 4, "Frequent": 3, "Occasional": 2, "Rare": 1 };

export function PhenotypeView({ data }) {
  const [sort, setSort] = useState("frequency");
  const list = [...(data.phenotypes || [])].sort((a, b) => {
    if (sort === "frequency") return (freqWeight[b.frequency] || 0) - (freqWeight[a.frequency] || 0);
    if (sort === "alpha") return a.name.localeCompare(b.name);
    if (sort === "onset") return a.onset.localeCompare(b.onset);
    return 0;
  });
  const freqStyle = f => {
    const w = freqWeight[f] || 0;
    if (w === 4) return { background: "var(--fg)", color: "var(--bg)" };
    if (w === 3) return { background: "var(--fg-secondary)", color: "var(--bg)" };
    if (w === 2) return { background: "var(--bg-sunken)", color: "var(--fg)" };
    return { background: "var(--bg-muted)", color: "var(--fg-subtle)" };
  };

  return (
    <div className="ctx-section">
      <div className="sort-bar">
        <span className="sort-label">Sort by</span>
        {["frequency", "onset", "alpha"].map(k => (
          <button key={k} className={"sort-btn " + (sort === k ? "active" : "")} onClick={() => setSort(k)}>
            {k === "alpha" ? "A\u2013Z" : k}
          </button>
        ))}
      </div>
      <div className="ev-list">
        {list.map((p, i) => (
          <div key={i} className="pheno-row">
            <div className="pheno-main">
              <div className="ev-name">{p.name}</div>
              <div className="ev-meta">{p.hpo}</div>
            </div>
            <div className="pheno-tags">
              <Badge variant="outline">{p.onset}</Badge>
              <span className="pheno-freq" style={freqStyle(p.frequency)}>{p.frequency}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ——— PATHWAY ———
export function PathwayView({ data }) {
  return (
    <div className="ctx-section">
      {(data.drugPathways || []).map((g, i) => (
        <div key={i} style={{ marginBottom: 20 }}>
          <div className="ev-label">
            <span>{g.drug}</span>
            <span>{g.pathways.length} pathways</span>
          </div>
          <div className="ev-list">
            {g.pathways.map((p, j) => (
              <div key={j} className="pw-row">
                <div className="pw-name">
                  {p.name}
                  {p.primary && <Badge variant="accent" style={{ marginLeft: 8 }}>primary</Badge>}
                </div>
                <div className="ev-meta">{p.go}</div>
                <Badge variant="outline">{p.action}</Badge>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ——— GENE_PROTEIN ———
export function GeneView({ data }) {
  const genes = [...(data.genes || [])].sort((a, b) => {
    if (a.highRisk && !b.highRisk) return -1;
    if (!a.highRisk && b.highRisk) return 1;
    return 0;
  });
  const evStyle = e => e === "GWAS" ? "intent" : e === "linkage" ? "solid" : "outline";
  return (
    <div className="ctx-section">
      <div className="ev-list">
        {genes.map((g, i) => (
          <div key={i} className={"gene-card " + (g.highRisk ? "high" : "")}>
            <div className="gene-head">
              <div>
                <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
                  <div className="gene-symbol">{g.gene}</div>
                  <div className="ev-meta">{g.hgnc}</div>
                  {g.highRisk && <Badge variant="accent">high-risk</Badge>}
                </div>
                <div style={{ fontSize: 13, color: "var(--fg-secondary)", marginTop: 4 }}>
                  encodes <span style={{ color: "var(--fg)", fontWeight: 500 }}>{g.protein}</span>
                  <span className="ev-meta" style={{ marginLeft: 8 }}>{g.uniprot}</span>
                </div>
              </div>
              <Badge variant={evStyle(g.evidence)}>{g.evidence}</Badge>
            </div>
            {g.pathways && g.pathways.length > 0 && (
              <div className="gene-pathways">
                {g.pathways.map((p, j) => (
                  <span key={j} className="gene-pathway">
                    {p.name} <span className="ev-meta">{p.go}</span>
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ——— GENERAL / composite ———
export function CompositeView({ data }) {
  const tabs = [];
  if (data.biomarkers) tabs.push({ key: "bio", label: "Biomarkers", n: Object.values(data.biomarkers).reduce((s, f) => s + (f.increased?.length || 0) + (f.decreased?.length || 0), 0) });
  if (data.drugs) tabs.push({ key: "drug", label: "Drugs", n: data.drugs.length });
  if (data.phenotypes) tabs.push({ key: "phe", label: "Phenotypes", n: data.phenotypes.length });
  if (data.genes) tabs.push({ key: "gene", label: "Genes", n: data.genes.length });
  const [active, setActive] = useState(tabs[0]?.key);

  return (
    <div className="ctx-section">
      <div className="tabs">
        {tabs.map(t => (
          <button key={t.key} className={"tab " + (active === t.key ? "active" : "")} onClick={() => setActive(t.key)}>
            {t.label}
            <span className="tab-count">{t.n}</span>
          </button>
        ))}
      </div>
      <div style={{ marginTop: 12 }}>
        {active === "bio" && <BiomarkerView data={data} />}
        {active === "drug" && <DrugView data={data} />}
        {active === "phe" && <PhenotypeView data={data} />}
        {active === "gene" && <GeneView data={data} />}
      </div>
    </div>
  );
}
