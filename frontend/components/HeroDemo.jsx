'use client';
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { Icon } from './Icons';

const DEMOS = [
  {
    id: 'biomarker',
    chip: 'CSF Biomarkers',
    query: "What CSF biomarkers are elevated in Alzheimer's disease?",
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
        including lecanemab, donanemab, and remternetug, all targeting amyloid clearance.
        Five compounds have FDA approval; 42 discontinued trials are also indexed.
      </>
    ),
  },
  {
    id: 'phenotype',
    chip: 'AD Symptoms',
    query: "What are the most frequent symptoms of Alzheimer's disease?",
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
        Gene <strong>APOE</strong> encodes <strong>apolipoprotein E</strong>, the strongest genetic
        risk factor for late-onset AD. APOE &epsilon;4 carriers show 3&ndash;12&times; elevated risk.
        The graph also tracks MAPT, PSEN1, and PSEN2 as high-risk loci with GWAS evidence.
      </>
    ),
  },
];

function BiomarkerPanel() {
  const rows = [
    { name: 'tau-total (CSF)',   cat: 'tau',              effect: '11.7', pval: '0.0001', bar: 100 },
    { name: 'NFL (CSF)',         cat: 'neurodegeneration', effect: '5.2',  pval: '0.0001', bar: 44  },
    { name: 'tau-phospho (CSF)', cat: 'tau',              effect: '1.7',  pval: '0.0001', bar: 15  },
    { name: 'YKL-40 (CSF)',      cat: 'inflammation',     effect: '1.4',  pval: '0.0001', bar: 12  },
    { name: 'Aβ42 (CSF)',        cat: 'amyloid',          effect: '1.1',  pval: '0.115',  bar: 9   },
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
    { gene: 'APOE',  hgnc: '613',  protein: 'apolipoprotein E',                  pr: 'PR:000004155' },
    { gene: 'MAPT',  hgnc: '6893', protein: 'microtubule-associated protein tau', pr: 'PR:000010173' },
    { gene: 'PSEN1', hgnc: '9508', protein: 'presenilin-1',                       pr: 'PR:000013344' },
    { gene: 'PSEN2', hgnc: '9509', protein: 'presenilin-2',                       pr: 'PR:000013345' },
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

export function HeroDemo() {
  const [activeIdx, setActiveIdx] = useState(0);

  useEffect(() => {
    fetch('/health').catch(() => {});
  }, []);

  const demo = DEMOS[activeIdx];
  const Panel = PANELS[demo.id];

  return (
    <section className="hero">
      <span className="eyebrow"><span className="dot"></span> Ontology-grounded knowledge graph &middot; Alzheimer&apos;s disease</span>
      <h1 className="hero-title">A research interface for the shape of Alzheimer&apos;s data.</h1>
      <p className="hero-sub">
        Ask in plain English. Every answer is traced to a node and the ontology behind it.
      </p>
      <p className="hero-sub-detail">
      Six biomedical ontologies, one graph. Every number traces to a node.
      </p>
      <div className="hero-cta">
        <Link className="btn btn-primary" href="/app">Open the workbench &rarr;</Link>
        <a className="btn btn-outline" href="#how">See how it works</a>
      </div>

      <div className="demo-intro">
        <div className="demo-intro-label">Interactive demo</div>
        <p className="demo-intro-head">Every query type. Each one shows its work.</p>
        <p className="demo-intro-sub">Pick a category. See how Atlas reads the question, picks a strategy, and grounds every answer in the graph.</p>
      </div>

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
  );
}
