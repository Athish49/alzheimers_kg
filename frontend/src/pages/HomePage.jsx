import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import '../styles/home.css';

/* ─── Pre-canned demo data ─── */

const DEMOS = [
  {
    id: 'biomarker',
    chip: 'CSF Biomarkers',
    query: "What CSF biomarkers are elevated in Alzheimer’s disease?",
    intent: 'BIOMARKER',
    strategy: 'AD_BIOMARKERS_V2',
    latency: '1.1s',
    answer: (
      <>
        The following CSF biomarkers are <strong>increased</strong> in Alzheimer&apos;s disease.
        Tau-total and tau-phospho reflect neuronal loss and tau pathology; NFL marks neurodegeneration.
        Effect sizes drawn from graph-indexed meta-analyses, with <code>p &lt; 0.0001</code> for top markers.
      </>
    ),
  },
  {
    id: 'drug',
    chip: 'Phase 3 Drugs',
    query: "What drugs are currently in Phase 3 trials for AD?",
    intent: 'DRUG_TRIAL',
    strategy: 'AD_DRUGS_V2',
    latency: '0.9s',
    answer: (
      <>
        The graph tracks <strong>29 active Phase 3 compounds</strong> for Alzheimer&apos;s disease,
        including lecanemab, donanemab, and remternetug — all targeting amyloid clearance.
        Five compounds have FDA approval; 42 discontinued trials are also indexed.
      </>
    ),
  },
  {
    id: 'phenotype',
    chip: 'AD Symptoms',
    query: "What are the most frequent symptoms of Alzheimer’s disease?",
    intent: 'PHENOTYPE',
    strategy: 'AD_PHENOTYPES_V2',
    latency: '0.8s',
    answer: (
      <>
        The most frequently documented clinical phenotypes are <strong>Memory impairment</strong>,{' '}
        <strong>Cognitive impairment</strong>, and <strong>Aphasia</strong>.
        All are indexed with HPO IDs and frequency weights from clinical cohort studies.
      </>
    ),
  },
  {
    id: 'gene',
    chip: 'APOE Gene',
    query: "What protein does APOE encode?",
    intent: 'GENE_PROTEIN',
    strategy: 'AD_GENES_V2',
    latency: '1.3s',
    answer: (
      <>
        Gene <strong>APOE</strong> encodes <strong>apolipoprotein E</strong> — the strongest genetic
        risk factor for late-onset AD. APOE &epsilon;4 carriers show 3&ndash;12&times; elevated risk.
        The graph also tracks MAPT, PSEN1, and PSEN2 as high-risk loci with GWAS evidence.
      </>
    ),
  },
];

/* ─── Evidence panel components ─── */

function BiomarkerPanel() {
  const rows = [
    { name: 'tau-total (CSF)',   cat: 'tau',              effect: '11.7', pval: '0.0001', bar: 100 },
    { name: 'NFL (CSF)',         cat: 'neurodegeneration', effect: '5.2',  pval: '0.0001', bar: 44  },
    { name: 'tau-phospho (CSF)', cat: 'tau',              effect: '1.7',  pval: '0.0001', bar: 15  },
    { name: 'YKL-40 (CSF)',      cat: 'inflammation',     effect: '1.4',  pval: '0.0001', bar: 12  },
    { name: 'Aβ42 (CSF)',   cat: 'amyloid',          effect: '1.1',  pval: '0.115',  bar: 9   },
  ];
  return (
    <>
      <div className="ctx-tabs">
        <span className="ctx-tab ctx-tab-active">CSF <span className="ctx-tab-count">22</span></span>
        <span className="ctx-tab">Plasma <span className="ctx-tab-count">17</span></span>
        <span className="ctx-tab">Other <span className="ctx-tab-count">12</span></span>
      </div>
      <div className="ctx-section-label">INCREASED IN AD (5)</div>
      {rows.map((r, i) => (
        <div key={i} className="ctx-bio-row">
          <div className="ctx-bio-meta">
            <div className="ctx-bio-name">{r.name}</div>
            <div className="ctx-bio-cat">{r.cat}</div>
          </div>
          <div className="ctx-bar-wrap">
            <div className="ctx-bar"><div className="ctx-bar-fill ctx-bar-up" style={{ width: `${r.bar}%` }} /></div>
          </div>
          <div className="ctx-effect arrow-up">↑ {r.effect}</div>
          <div className="ctx-pval">p {r.pval}</div>
        </div>
      ))}
    </>
  );
}

function DrugPanel() {
  const drugs = [
    { name: 'Lecanemab',    phase: 'Phase 3', type: 'Anti-amyloid antibody' },
    { name: 'Donanemab',    phase: 'Phase 3', type: 'Anti-amyloid antibody' },
    { name: 'Remternetug',  phase: 'Phase 3', type: 'Anti-amyloid antibody' },
    { name: 'Semaglutide',  phase: 'Phase 3', type: 'GLP-1 agonist'         },
  ];
  return (
    <>
      <div className="ctx-tabs" style={{ flexWrap: 'wrap' }}>
        <span className="ctx-tab">Approved <span className="ctx-tab-count">5</span></span>
        <span className="ctx-tab ctx-tab-active">Phase 3 <span className="ctx-tab-count">29</span></span>
        <span className="ctx-tab">Phase 1–2 <span className="ctx-tab-count">74</span></span>
        <span className="ctx-tab">Discontinued <span className="ctx-tab-count">42</span></span>
      </div>
      {drugs.map((d, i) => (
        <div key={i} className="ctx-drug-card">
          <div>
            <div className="ctx-drug-name">{d.name}</div>
            <div className="ctx-drug-meta">
              <span className="ctx-drug-phase">{d.phase}</span>
              <span className="ctx-drug-sep">&middot;</span>
              <span className="ctx-drug-type">{d.type}</span>
            </div>
          </div>
          <span className="ctx-chevron">›</span>
        </div>
      ))}
    </>
  );
}

function PhenotypePanel() {
  const items = [
    { name: 'Memory impairment',   hpo: 'HP:0002354', freq: 100 },
    { name: 'Cognitive impairment', hpo: 'HP:0100543', freq: 88  },
    { name: 'Aphasia',             hpo: 'HP:0002381', freq: 72  },
    { name: 'Behavioral changes',  hpo: 'HP:0000708', freq: 54  },
  ];
  return (
    <>
      <div className="ctx-sort-row">
        <span className="ctx-sort-label">SORT BY</span>
        <span className="ctx-sort-btn ctx-sort-active">Frequency</span>
        <span className="ctx-sort-btn">Onset</span>
        <span className="ctx-sort-btn">A–Z</span>
      </div>
      {items.map((p, i) => (
        <div key={i} className="ctx-pheno-card">
          <div className="ctx-pheno-text">
            <div className="ctx-pheno-name">{p.name}</div>
            <div className="ctx-pheno-id">{p.hpo}</div>
          </div>
          <div className="ctx-pheno-bar-wrap">
            <div className="ctx-pheno-bar">
              <div className="ctx-pheno-bar-fill" style={{ width: `${p.freq}%` }} />
            </div>
          </div>
        </div>
      ))}
    </>
  );
}

function GenePanel() {
  const genes = [
    { gene: 'APOE',  hgnc: '613',  protein: 'apolipoprotein E',                       pr: 'PR:000004155' },
    { gene: 'MAPT',  hgnc: '6893', protein: 'microtubule-associated protein tau',      pr: 'PR:000010173' },
    { gene: 'PSEN1', hgnc: '9508', protein: 'presenilin-1',                            pr: 'PR:000013344' },
    { gene: 'PSEN2', hgnc: '9509', protein: 'presenilin-2',                            pr: 'PR:000013345' },
  ];
  return (
    <>
      {genes.map((g, i) => (
        <div key={i} className="ctx-gene-card">
          <div className="ctx-gene-head">
            <span className="ctx-gene-name">{g.gene}</span>
            <span className="ctx-gene-hgnc">HGNC:{g.hgnc}</span>
            <span className="ctx-gene-risk">high-risk</span>
            <span className="ctx-gwas-badge">GWAS</span>
          </div>
          <div className="ctx-gene-encodes">
            encodes <strong>{g.protein}</strong>{' '}
            <span className="ctx-gene-prid">{g.pr}</span>
          </div>
        </div>
      ))}
    </>
  );
}

const PANELS = { biomarker: BiomarkerPanel, drug: DrugPanel, phenotype: PhenotypePanel, gene: GenePanel };

const CHK = {
  yes:     <span className="chk-yes">✓</span>,
  no:      <span className="chk-no">—</span>,
  partial: <span className="chk-partial">~</span>,
};

/* ─── Page component ─── */

export function HomePage() {
  const [activeIdx, setActiveIdx] = useState(0);
  const demo = DEMOS[activeIdx];
  const Panel = PANELS[demo.id];

  return (
    <div className="home-page" style={{ overflow: 'auto', height: '100vh' }}>
      {/* Nav */}
      <header className="nav">
        <div className="nav-inner">
          <Link className="brand" to="/">
            <span className="brand-mark">A</span>
            <span>Atlas</span>
          </Link>
          <nav className="nav-links">
            <a className="nav-link" href="#problem">Problem</a>
            <a className="nav-link" href="#why">Why Atlas</a>
            <a className="nav-link" href="#how">How it works</a>
            <a className="nav-link" href="#capabilities">Capabilities</a>
            <a className="nav-link" href="#who">Who it&apos;s for</a>
          </nav>
          <div style={{ display: 'flex', gap: 8 }}>
            <a className="btn btn-ghost" href="#">Docs</a>
            <Link className="btn btn-primary" to="/app">Open Atlas &rarr;</Link>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="hero">
        <span className="eyebrow"><span className="dot"></span> Knowledge graph &middot; Alzheimer&apos;s disease</span>
        <h1 className="hero-title">A research interface <em>for the shape</em> of Alzheimer&apos;s data.</h1>
        <p className="hero-sub">
          Ask in plain English, get answers traced to the graph.
        </p>
        <p className="hero-sub-detail">
          A GraphRAG workbench over a curated Alzheimer&apos;s knowledge graph &mdash; biomarker effect sizes, trial phases, phenotype frequencies, and gene-to-pathway evidence, all grounded and traceable.
        </p>
        <div className="hero-cta">
          <Link className="btn btn-primary" to="/app">Open the workbench &rarr;</Link>
          <a className="btn btn-outline" href="#how">See how it works</a>
        </div>

        {/* Demo section intro */}
        <div className="demo-intro">
          <div className="demo-intro-label">Interactive demo</div>
          <p className="demo-intro-head">Four query types. <em>Each one shows its work.</em></p>
          <p className="demo-intro-sub">Select a category on the right to see how Atlas classifies intent, routes to a retrieval strategy, and renders structured evidence alongside every answer — nothing invented, everything traceable.</p>
        </div>

        {/* Hero mock preview + floating sidebar */}
        <div className="demo-wrapper">
          <div className="hero-visual" aria-hidden="true">
            <div className="visual-chrome">
              <span className="tl"></span><span className="tl"></span><span className="tl"></span>
              <div style={{ marginLeft: 12, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--fg-subtle)' }}>
                atlas.session &middot; {demo.strategy}
              </div>
              <div className="latency-badge">
                <span className="latency-dot" />
                answered in {demo.latency}
              </div>
            </div>
            <div className="visual-body">
              <div className="visual-chat demo-fade" key={`chat-${demo.id}`}>
                <div className="home-msg-user">{demo.query}</div>
                <div className="cls-row">
                  <span className="badge badge-intent">{demo.intent}</span>
                  <span className="badge badge-strategy">{demo.strategy}</span>
                </div>
                <div className="visual-answer prose">
                  <p>{demo.answer}</p>
                </div>
              </div>
              <div className="visual-ctx demo-fade" key={`ctx-${demo.id}`}>
                <div className="home-ctx-label">Evidence &middot; {demo.intent}</div>
                <Panel />
              </div>
            </div>
          </div>

          <div className="demo-sidebar" role="tablist" aria-label="Demo query types">
            {DEMOS.map((d, i) => (
              <button
                key={d.id}
                role="tab"
                aria-selected={activeIdx === i}
                className={`demo-sidebar-item${activeIdx === i ? ' demo-sidebar-active' : ''}`}
                onClick={() => setActiveIdx(i)}
              >
                {d.chip}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Problem */}
      <section className="band" id="problem">
        <div className="wrap">
          <div className="section-head">
            <div className="section-label">The problem</div>
            <h2>Alzheimer&apos;s research lives in <em>ten thousand</em> disconnected rows.</h2>
            <p className="section-lede">Biomarker studies, trial registries, gene databases, phenotype ontologies, pathway annotations &mdash; all authoritative, all siloed. Answering a single question usually means opening six browser tabs and reconciling them in your head.</p>
          </div>
          <div className="problem-grid">
            <div className="prob-card">
              <div className="prob-num">01</div>
              <h3 className="prob-title">Fragmented sources</h3>
              <p className="prob-body">HPO for phenotypes. CHEBI for drugs. UniProt for proteins. GWAS catalogs for risk. Each with its own IDs, its own update cadence, its own semantics.</p>
            </div>
            <div className="prob-card">
              <div className="prob-num">02</div>
              <h3 className="prob-title">Chat tools don&apos;t show their work</h3>
              <p className="prob-body">General-purpose assistants will happily hallucinate a p-value. Researchers need to see the specific nodes and edges behind every claim &mdash; not trust a paragraph.</p>
            </div>
            <div className="prob-card">
              <div className="prob-num">03</div>
              <h3 className="prob-title">Dashboards don&apos;t answer questions</h3>
              <p className="prob-body">Classical BI tools show you what you already asked to see. They can&apos;t respond to &ldquo;which Phase 3 drugs target APOE-adjacent pathways?&rdquo; without a custom build.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Why Atlas */}
      <section className="band" id="why">
        <div className="wrap">
          <div className="section-head">
            <div className="section-label">Why Atlas</div>
            <h2>Why not just <em>ChatGPT?</em></h2>
            <p className="section-lede">General assistants give you an answer. Atlas gives you an answer <em>and shows its work</em>. The difference matters when a p-value is on the line.</p>
          </div>

          <div className="compare-table">
            <div className="compare-grid">
              <div className="compare-feat-head"></div>
              <div className="compare-col-head compare-atlas-head">Atlas</div>
              <div className="compare-col-head">General chatbot</div>
              <div className="compare-col-head">BI dashboard</div>

              {[
                { feat: 'Grounded in curated source data',      atlas: 'yes', chat: 'no',      bi: 'yes'     },
                { feat: 'Answers novel, open-ended questions',  atlas: 'yes', chat: 'partial', bi: 'no'      },
                { feat: 'Shows evidence behind each claim',     atlas: 'yes', chat: 'no',      bi: 'no'      },
                { feat: 'Every claim traceable to a node',      atlas: 'yes', chat: 'no',      bi: 'no'      },
                { feat: 'Domain-structured evidence views',     atlas: 'yes', chat: 'no',      bi: 'no'      },
              ].map((row, i) => (
                <React.Fragment key={i}>
                  <div className="compare-feat">{row.feat}</div>
                  <div className={`compare-cell compare-atlas-cell chk-${row.atlas}`}>{CHK[row.atlas]}</div>
                  <div className={`compare-cell chk-${row.chat}`}>{CHK[row.chat]}</div>
                  <div className={`compare-cell chk-${row.bi}`}>{CHK[row.bi]}</div>
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="band" id="how">
        <div className="wrap">
          <div className="section-head">
            <div className="section-label">How Atlas works</div>
            <h2>Questions are <em>classified</em>, routed, and grounded in the graph.</h2>
            <p className="section-lede">Every query passes through an intent classifier that picks the right retrieval strategy, pulls the relevant subgraph, and hands structured context to the language model. Nothing is invented &mdash; every claim traces back to a node.</p>
          </div>
          <div className="how-grid">
            <div className="how-steps">
              <div className="step">
                <div className="step-num">01</div>
                <div>
                  <h4 className="step-title">Classify intent</h4>
                  <p className="step-body">Your question is labelled as one of seven types: biomarker, drug &amp; trial, phenotype, pathway, gene-protein, general AD, or other. The classifier surfaces its notes so you can verify it understood you.</p>
                </div>
              </div>
              <div className="step">
                <div className="step-num">02</div>
                <div>
                  <h4 className="step-title">Select a retrieval strategy</h4>
                  <p className="step-body">Each intent routes to a named strategy (<code>AD_BIOMARKERS_V2</code>, <code>AD_DRUGS_V2</code>, &hellip;) &mdash; a specific subgraph traversal tuned for that class of question.</p>
                </div>
              </div>
              <div className="step">
                <div className="step-num">03</div>
                <div>
                  <h4 className="step-title">Retrieve the subgraph</h4>
                  <p className="step-body">The strategy pulls the relevant nodes and edges &mdash; biomarker values with fluid, direction, effect size, p-value; drugs with phase, CHEBI ID, and pathway links; and so on.</p>
                </div>
              </div>
              <div className="step">
                <div className="step-num">04</div>
                <div>
                  <h4 className="step-title">Synthesise, but show the evidence</h4>
                  <p className="step-body">The language model writes a grounded answer. The retrieved context is rendered alongside it as structured data &mdash; so every number in the prose corresponds to a row you can inspect.</p>
                </div>
              </div>
            </div>

            <div className="diagram" aria-hidden="true">
              <div className="diag-row">
                <span className="tag">Query</span>
                <span className="val">&ldquo;Which Phase 3 drugs target the amyloid pathway?&rdquo;</span>
              </div>
              <div className="diag-arrow">&darr;</div>
              <div className="diag-row">
                <span className="tag">Intent</span>
                <span className="val"><span className="badge badge-intent">DRUG_TRIAL</span></span>
              </div>
              <div className="diag-sub">Notes: matched pathway keyword &ldquo;amyloid&rdquo;; phase filter &ldquo;Phase 3&rdquo;.</div>
              <div className="diag-arrow">&darr;</div>
              <div className="diag-row">
                <span className="tag">Strategy</span>
                <span className="val"><span className="badge badge-strategy">AD_DRUGS_V2</span></span>
              </div>
              <div className="diag-arrow">&darr;</div>
              <div className="diag-row">
                <span className="tag">Subgraph</span>
                <span className="val" style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  <span className="badge badge-strategy">Drug &times;7</span>
                  <span className="badge badge-strategy">Pathway &times;4</span>
                  <span className="badge badge-strategy">Trial &times;12</span>
                </span>
              </div>
              <div className="diag-arrow">&darr;</div>
              <div className="diag-row" style={{ background: 'var(--fg)', color: 'var(--bg)', borderColor: 'var(--fg)' }}>
                <span className="tag" style={{ color: 'rgba(255,255,255,0.55)' }}>Answer</span>
                <span className="val" style={{ color: 'var(--bg)' }}>Grounded synthesis + evidence panel</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Capabilities */}
      <section className="band" id="capabilities">
        <div className="wrap">
          <div className="section-head">
            <div className="section-label">Capabilities</div>
            <h2>Six query classes, each with a <em>tailored</em> view.</h2>
            <p className="section-lede">The intent you ask determines how the evidence is rendered. A biomarker question doesn&apos;t look like a drug question &mdash; the structure of the data is different, and the UI follows.</p>
          </div>

          <div className="cap-grid">
            <div className="cap-cell">
              <div className="cap-icon">01 &middot; BIOMARKER</div>
              <h4 className="cap-title">Biomarker evidence</h4>
              <p className="cap-body">Grouped by fluid, split by direction, sorted by effect size. See what goes up, what goes down, and how strongly.</p>
              <div className="cap-example">&ldquo;Which plasma biomarkers are decreased in AD?&rdquo;</div>
            </div>
            <div className="cap-cell">
              <div className="cap-icon">02 &middot; DRUG_TRIAL</div>
              <h4 className="cap-title">Drug landscape</h4>
              <p className="cap-body">Compounds grouped by status &mdash; approved to discontinued &mdash; with phase, CHEBI ID, and pathway targets per drug.</p>
              <div className="cap-example">&ldquo;What amyloid-targeting drugs have FDA approval?&rdquo;</div>
            </div>
            <div className="cap-cell">
              <div className="cap-icon">03 &middot; PHENOTYPE</div>
              <h4 className="cap-title">Clinical features</h4>
              <p className="cap-body">Phenotypes with HPO IDs, onset, and frequency weighting. Sortable by frequency, onset, or alphabetically.</p>
              <div className="cap-example">&ldquo;At what stage does agitation typically appear?&rdquo;</div>
            </div>
            <div className="cap-cell">
              <div className="cap-icon">04 &middot; PATHWAY</div>
              <h4 className="cap-title">Drug&ndash;pathway links</h4>
              <p className="cap-body">Which drugs target which pathways, with action type (inhibition, activation, modulation) and primary-target flags.</p>
              <div className="cap-example">&ldquo;Which biological pathways does lecanemab affect?&rdquo;</div>
            </div>
            <div className="cap-cell">
              <div className="cap-icon">05 &middot; GENE_PROTEIN</div>
              <h4 className="cap-title">Gene&ndash;protein associations</h4>
              <p className="cap-body">Gene&ndash;protein pairs with HGNC and UniProt IDs, GWAS evidence strength, and the pathways the protein participates in.</p>
              <div className="cap-example">&ldquo;Which genes are most strongly associated with AD by GWAS?&rdquo;</div>
            </div>
            <div className="cap-cell">
              <div className="cap-icon">06 &middot; GENERAL_AD</div>
              <h4 className="cap-title">Composite overview</h4>
              <p className="cap-body">A tabbed view combining biomarkers, drugs, phenotypes, and genes &mdash; for questions that don&apos;t fit a single class.</p>
              <div className="cap-example">&ldquo;Give me an overview of the AD landscape.&rdquo;</div>
            </div>
          </div>
        </div>
      </section>

      {/* Who */}
      <section className="band" id="who">
        <div className="wrap">
          <div className="section-head">
            <div className="section-label">Who it&apos;s for</div>
            <h2>Built for people comfortable with <em>p-values</em>.</h2>
            <p className="section-lede">Atlas is not a consumer assistant. It&apos;s an expert tool for the three constituencies that spend the most time reconciling AD evidence.</p>
          </div>
          <div className="audience-grid">
            <div className="aud-card">
              <div className="aud-role">Drug researcher</div>
              <p className="aud-quote">&ldquo;I need to see the <em>landscape</em>. Which compounds are approved, which are still running, what pathway they hit, and where the failures clustered.&rdquo;</p>
              <div className="aud-person">
                <div className="aud-avatar">DR</div>
                <div>
                  <div className="aud-name">Preclinical discovery teams</div>
                  <div className="aud-title">Industry &amp; academic pharma</div>
                </div>
              </div>
            </div>
            <div className="aud-card">
              <div className="aud-role">Clinician</div>
              <p className="aud-quote">&ldquo;The <em>onset</em> and <em>frequency</em> of a phenotype matters more to me than the prose describing it. Give me the HPO tags and let me sort.&rdquo;</p>
              <div className="aud-person">
                <div className="aud-avatar">CL</div>
                <div>
                  <div className="aud-name">Neurologists &amp; memory-clinic staff</div>
                  <div className="aud-title">Academic medical centres</div>
                </div>
              </div>
            </div>
            <div className="aud-card">
              <div className="aud-role">Data scientist</div>
              <p className="aud-quote">&ldquo;Show me <em>exactly</em> what went into the prompt. I want the raw context markdown so I can pipe it into a notebook and tune retrieval.&rdquo;</p>
              <div className="aud-person">
                <div className="aud-avatar">DS</div>
                <div>
                  <div className="aud-name">Computational biology</div>
                  <div className="aud-title">Research informatics</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="cta-band">
        <h2>Start with a question.</h2>
        <p>No setup, no API keys. Type what you want to know and watch the graph answer.</p>
        <Link className="btn btn-primary" to="/app" style={{ padding: '13px 22px', fontSize: 15 }}>Open Atlas &rarr;</Link>
      </section>

      <footer className="home-footer">
        <div>Atlas &middot; An Alzheimer&apos;s GraphRAG research interface</div>
        <div className="f-links">
          <a href="#">Methodology</a>
          <a href="#">Data sources</a>
          <a href="#">Contact</a>
        </div>
      </footer>
    </div>
  );
}
