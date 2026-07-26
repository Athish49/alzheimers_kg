'use client';
import React, { useState } from 'react';
import Link from 'next/link';
import { Icon } from './Icons';

/* ─── Pre-canned demo data ─── */

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

/* ─── Evidence panel components ─── */

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

const CHK = {
  yes:     <span className="chk-yes">✓</span>,
  no:      <span className="chk-no">-</span>,
  partial: <span className="chk-partial">~</span>,
};

/* ─── Knowledge Graph Schema ─── */

function KnowledgeGraphSchema() {
  const nodes = [
    { id: 'disease',     label: 'Disease',     x: 450, y: 280, dark: true  },
    { id: 'drug',        label: 'Drug',         x: 720, y: 270, dark: false },
    { id: 'gene',        label: 'Gene',         x: 195, y: 340, dark: false },
    { id: 'protein',     label: 'Protein',      x: 330, y: 430, dark: false },
    { id: 'phenotype',   label: 'Phenotype',    x: 170, y: 230, dark: false },
    { id: 'biomarker',   label: 'Biomarker',    x: 330, y: 100, dark: false },
    { id: 'pathway',     label: 'Pathway',      x: 670, y: 110, dark: false },
    { id: 'trial',       label: 'Trial',        x: 560, y: 390, dark: false },
    { id: 'mechanism',   label: 'Mechanism',    x: 560, y: 490, dark: false },
    { id: 'riskfactor',  label: 'RiskFactor',   x: 120, y: 440, dark: false },
    { id: 'variant',     label: 'Variant',      x: 290, y: 510, dark: false },
    { id: 'study',       label: 'Study',        x: 105, y: 140, dark: false },
    { id: 'fluid',       label: 'Fluid',        x: 145, y: 55,  dark: false },
    { id: 'company',     label: 'Company',      x: 860, y: 145, dark: false },
    { id: 'therapytype', label: 'TherapyType',  x: 870, y: 380, dark: false },
  ];

  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));

  const edges = [
    { from: 'biomarker',  to: 'fluid',      label: 'MEASURED_IN',             dashed: false },
    { from: 'biomarker',  to: 'disease',    label: 'PREDICTS_PROGRESSION_TO', dashed: false },
    { from: 'study',      to: 'disease',    label: 'REPORTS',                 dashed: false },
    { from: 'phenotype',  to: 'disease',    label: 'HAS_PHENOTYPE',           dashed: false },
    { from: 'gene',       to: 'disease',    label: 'ASSOCIATED_WITH_DISEASE', dashed: false },
    { from: 'gene',       to: 'protein',    label: 'ENCODES',                 dashed: false },
    { from: 'gene',       to: 'disease',    label: 'INCREASES_RISK_OF',       dashed: false },
    { from: 'protein',    to: 'disease',    label: 'CLEAVES',                 dashed: false },
    { from: 'riskfactor', to: 'disease',    label: 'INCREASES_RISK_OF',       dashed: false },
    { from: 'variant',    to: 'gene',       label: 'LOCATED_IN',              dashed: false },
    { from: 'drug',       to: 'disease',    label: 'TREATS',                  dashed: false },
    { from: 'drug',       to: 'pathway',    label: 'AFFECTS_PATHWAY',         dashed: false },
    { from: 'drug',       to: 'trial',      label: 'HAS_TRIAL',               dashed: false },
    { from: 'drug',       to: 'company',    label: 'DEVELOPED_BY',            dashed: false },
    { from: 'drug',       to: 'therapytype',label: 'HAS_THERAPY_TYPE',        dashed: false },
    { from: 'trial',      to: 'disease',    label: 'FOR_DISEASE',             dashed: false },
    { from: 'trial',      to: 'mechanism',  label: 'INVOLVES_PATHOLOGY',      dashed: false },
  ];

  const NW = 96;
  const NH = 34;

  function edgePath(e) {
    const a = nodeMap[e.from];
    const b = nodeMap[e.to];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    return {
      x1: a.x + ux * (NW / 2 + 4),
      y1: a.y + uy * (NH / 2 + 4),
      x2: b.x - ux * (NW / 2 + 10),
      y2: b.y - uy * (NH / 2 + 10),
      mx: (a.x + b.x) / 2,
      my: (a.y + b.y) / 2,
    };
  }

  return (
    <section className="band" id="graph-schema">
      <div className="wrap">
        <div className="section-head">
          <div className="section-label">The knowledge graph</div>
          <h2>Every entity connected. Every edge carrying the evidence.</h2>
          <p className="section-lede">Every node carries a canonical ontology ID. Every edge carries the evidence: direction, effect size, phase, or study count. Not just a link.</p>
        </div>
        <div className="kg-schema-wrap">
          <svg
            viewBox="0 0 980 560"
            className="kg-schema-svg"
            aria-label="Knowledge graph schema showing Disease at center connected to Drug, Gene, Protein, Phenotype, Biomarker, Pathway, Trial, Mechanism, RiskFactor, Variant, Study, Fluid, Company, and TherapyType nodes with labeled relationships"
          >
            <defs>
              <marker id="kg-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto" markerUnits="userSpaceOnUse">
                <path d="M0,0.5 L7,3.5 L0,6.5" fill="none" stroke="#aaa" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round"/>
              </marker>
            </defs>

            {edges.map((e, i) => {
              const { x1, y1, x2, y2, mx, my } = edgePath(e);
              const labelW = e.label.length * 5.2 + 8;
              return (
                <g key={i}>
                  <line
                    x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke="#ccc"
                    strokeWidth="1.1"
                    strokeDasharray={e.dashed ? '5,4' : undefined}
                    markerEnd="url(#kg-arrow)"
                  />
                  <rect
                    x={mx - labelW / 2}
                    y={my - 7}
                    width={labelW}
                    height={13}
                    rx="2"
                    fill="#f5f4f0"
                    opacity="0.92"
                  />
                  <text
                    x={mx}
                    y={my}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize="7.5"
                    fill="#999"
                    letterSpacing="0.3"
                    style={{ fontFamily: 'var(--font-mono, monospace)', userSelect: 'none' }}
                  >
                    {e.label}
                  </text>
                </g>
              );
            })}

            {nodes.map(n => (
              <g key={n.id} transform={`translate(${n.x},${n.y})`}>
                <rect
                  x={-NW / 2} y={-NH / 2}
                  width={NW} height={NH}
                  rx="10"
                  fill={n.dark ? '#1a1a1a' : '#fff'}
                  stroke={n.dark ? '#1a1a1a' : '#ddd'}
                  strokeWidth="1.2"
                />
                <text
                  textAnchor="middle"
                  dominantBaseline="middle"
                  x={0}
                  fontSize="12"
                  fontWeight="500"
                  fill={n.dark ? '#fff' : '#1a1a1a'}
                  style={{ fontFamily: 'var(--font-sans, sans-serif)', userSelect: 'none' }}
                >
                  {n.label}
                </text>
              </g>
            ))}
          </svg>
        </div>
      </div>
    </section>
  );
}

/* ─── Architecture diagram ─── */

function ArchDiagram() {
  const arrowD = (
    <marker id="arr-d" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0.5 L6,3.5 L0,6.5" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
    </marker>
  );
  const arrowM = (
    <marker id="arr-m" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0.5 L6,3.5 L0,6.5" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
    </marker>
  );
  return (
    <div
      className="arch-wrap"
      role="img"
      aria-label="Atlas two-phase pipeline: ontologies and curated evidence are reconciled into an Alzheimer's knowledge graph at build time; at query time a question with conversation history passes through a query rewriter, entity linker, intent classifier, graph router, subgraph retrieval, and grounded LLM synthesis to produce a traceable answer."
    >
      {/* ── Desktop SVG ── */}
      <svg className="arch-svg arch-desktop" viewBox="0 0 520 1048">
        <defs>{arrowD}</defs>
        <text x="10" y="18" className="arch-t-phase">PHASE 1 · BUILD TIME</text>
        <rect x="10" y="28" width="225" height="70" rx="11" className="arch-box"/>
        <text x="122.5" y="54" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Biomedical ontologies</text>
        <text x="122.5" y="75" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">Six canonical vocabularies</text>
        <rect x="285" y="28" width="225" height="70" rx="11" className="arch-box"/>
        <text x="397.5" y="54" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Curated AD evidence</text>
        <text x="397.5" y="75" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">AlzForum knowledge base</text>
        <path d="M 122.5,98 L 122.5,116" className="arch-line"/>
        <path d="M 397.5,98 L 397.5,116" className="arch-line"/>
        <path d="M 122.5,116 L 397.5,116" className="arch-line"/>
        <path d="M 260,116 L 260,130" className="arch-line" markerEnd="url(#arr-d)"/>
        <rect x="110" y="130" width="300" height="70" rx="11" className="arch-box"/>
        <text x="260" y="156" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Reconcile to canonical IDs</text>
        <text x="260" y="177" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">One identity per entity</text>
        <path d="M 260,200 L 260,230" className="arch-line" markerEnd="url(#arr-d)"/>
        <rect x="60" y="230" width="400" height="78" rx="11" className="arch-hub"/>
        <text x="260" y="259" textAnchor="middle" dominantBaseline="middle" className="arch-t-hub-title">Alzheimer&apos;s knowledge graph</text>
        <text x="260" y="281" textAnchor="middle" dominantBaseline="middle" className="arch-t-hub-sub">typed property graph · Neo4j</text>
        <path d="M 60,269 L 38,269 L 38,783 L 110,783" className="arch-dash" markerEnd="url(#arr-d)"/>
        <text x="24" y="526" textAnchor="middle" dominantBaseline="middle" className="arch-t-reads" transform="rotate(-90 24 526)">reads the graph</text>
        <text x="10" y="324" className="arch-t-phase">PHASE 2 · QUERY TIME</text>
        <path d="M 260,308 L 260,348" className="arch-line" markerEnd="url(#arr-d)"/>
        <rect x="110" y="348" width="300" height="70" rx="11" className="arch-box"/>
        <text x="260" y="374" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Query rewriter</text>
        <text x="260" y="395" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">LLM Call 1 · coreference resolution</text>
        <path d="M 260,418 L 260,448" className="arch-line" markerEnd="url(#arr-d)"/>
        <rect x="110" y="448" width="300" height="70" rx="11" className="arch-box"/>
        <text x="260" y="474" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Entity linker</text>
        <text x="260" y="495" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">in-memory vocab · synonym lookup</text>
        <path d="M 260,518 L 260,548" className="arch-line" markerEnd="url(#arr-d)"/>
        <rect x="110" y="548" width="300" height="70" rx="11" className="arch-box"/>
        <text x="260" y="574" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Intent classifier</text>
        <text x="260" y="595" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">entity-aware · seven classes</text>
        <path d="M 260,618 L 260,648" className="arch-line" markerEnd="url(#arr-d)"/>
        <rect x="110" y="648" width="300" height="70" rx="11" className="arch-box"/>
        <text x="260" y="674" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Graph router</text>
        <text x="260" y="695" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">targeted → bulk fallback</text>
        <path d="M 260,718 L 260,748" className="arch-line" markerEnd="url(#arr-d)"/>
        <rect x="110" y="748" width="300" height="70" rx="11" className="arch-box"/>
        <text x="260" y="774" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Subgraph retrieval</text>
        <text x="260" y="795" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">Cypher traversal · Neo4j</text>
        <path d="M 260,818 L 260,848" className="arch-line" markerEnd="url(#arr-d)"/>
        <rect x="110" y="848" width="300" height="70" rx="11" className="arch-box"/>
        <text x="260" y="874" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Grounded synthesis</text>
        <text x="260" y="895" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">LLM Call 2 · context-only</text>
        <path d="M 260,918 L 260,948" className="arch-line" markerEnd="url(#arr-d)"/>
        <rect x="110" y="948" width="300" height="70" rx="11" className="arch-box"/>
        <text x="260" y="974" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Answer + evidence</text>
        <text x="260" y="995" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">every claim traces to a node</text>
      </svg>

      {/* ── Mobile SVG ── */}
      <svg className="arch-svg arch-mobile" viewBox="0 0 340 1136">
        <defs>{arrowM}</defs>
        <text x="10" y="18" className="arch-t-phase">PHASE 1 · BUILD TIME</text>
        <rect x="20" y="28" width="300" height="70" rx="11" className="arch-box"/>
        <text x="170" y="54" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Biomedical ontologies</text>
        <text x="170" y="75" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">Six canonical vocabularies</text>
        <path d="M 170,98 L 170,118" className="arch-line" markerEnd="url(#arr-m)"/>
        <rect x="20" y="118" width="300" height="70" rx="11" className="arch-box"/>
        <text x="170" y="144" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Curated AD evidence</text>
        <text x="170" y="165" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">AlzForum knowledge base</text>
        <path d="M 170,188 L 170,218" className="arch-line" markerEnd="url(#arr-m)"/>
        <rect x="20" y="218" width="300" height="70" rx="11" className="arch-box"/>
        <text x="170" y="244" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Reconcile to canonical IDs</text>
        <text x="170" y="265" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">One identity per entity</text>
        <path d="M 170,288 L 170,318" className="arch-line" markerEnd="url(#arr-m)"/>
        <rect x="20" y="318" width="300" height="78" rx="11" className="arch-hub"/>
        <text x="170" y="347" textAnchor="middle" dominantBaseline="middle" className="arch-t-hub-title">Alzheimer&apos;s knowledge graph</text>
        <text x="170" y="369" textAnchor="middle" dominantBaseline="middle" className="arch-t-hub-sub">typed property graph · Neo4j</text>
        <text x="10" y="414" className="arch-t-phase">PHASE 2 · QUERY TIME</text>
        <path d="M 170,396 L 170,428" className="arch-line" markerEnd="url(#arr-m)"/>
        <rect x="20" y="428" width="300" height="70" rx="11" className="arch-box"/>
        <text x="170" y="454" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Query rewriter</text>
        <text x="170" y="475" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">LLM Call 1 · coreference resolution</text>
        <path d="M 170,498 L 170,528" className="arch-line" markerEnd="url(#arr-m)"/>
        <rect x="20" y="528" width="300" height="70" rx="11" className="arch-box"/>
        <text x="170" y="554" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Entity linker</text>
        <text x="170" y="575" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">in-memory vocab · synonym lookup</text>
        <path d="M 170,598 L 170,628" className="arch-line" markerEnd="url(#arr-m)"/>
        <rect x="20" y="628" width="300" height="70" rx="11" className="arch-box"/>
        <text x="170" y="654" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Intent classifier</text>
        <text x="170" y="675" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">entity-aware · seven classes</text>
        <path d="M 170,698 L 170,728" className="arch-line" markerEnd="url(#arr-m)"/>
        <rect x="20" y="728" width="300" height="70" rx="11" className="arch-box"/>
        <text x="170" y="754" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Graph router</text>
        <text x="170" y="775" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">targeted → bulk fallback</text>
        <path d="M 170,798 L 170,828" className="arch-line" markerEnd="url(#arr-m)"/>
        <rect x="20" y="828" width="300" height="70" rx="11" className="arch-box"/>
        <text x="170" y="854" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Subgraph retrieval</text>
        <text x="170" y="875" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">Cypher traversal · Neo4j</text>
        <text x="170" y="912" textAnchor="middle" dominantBaseline="middle" className="arch-t-reads">reads the knowledge graph</text>
        <path d="M 170,920 L 170,936" className="arch-line" markerEnd="url(#arr-m)"/>
        <rect x="20" y="936" width="300" height="70" rx="11" className="arch-box"/>
        <text x="170" y="962" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Grounded synthesis</text>
        <text x="170" y="983" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">LLM Call 2 · context-only</text>
        <path d="M 170,1006 L 170,1036" className="arch-line" markerEnd="url(#arr-m)"/>
        <rect x="20" y="1036" width="300" height="70" rx="11" className="arch-box"/>
        <text x="170" y="1062" textAnchor="middle" dominantBaseline="middle" className="arch-t-title">Answer + evidence</text>
        <text x="170" y="1083" textAnchor="middle" dominantBaseline="middle" className="arch-t-sub">every claim traces to a node</text>
      </svg>
    </div>
  );
}

/* ─── Footer ─── */

function FooterWithCopy() {
  const [copied, setCopied] = useState(false);
  const copyEmail = () => {
    navigator.clipboard.writeText('grathish49@gmail.com');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <footer className="home-footer">
      <div>Atlas &middot; An ontology-grounded Graph RAG interface for Alzheimer&apos;s research</div>
      <div className="f-links">
        <span className="f-built-by">Built by Athish Gopal Rajesh</span>
        <span className="f-divider" />
        <a href="https://athish-gopal-rajesh.vercel.app/" target="_blank" rel="noopener noreferrer">Portfolio &amp; about</a>
        <span>&middot;</span>
        <a href="https://www.linkedin.com/in/athishgr/" target="_blank" rel="noopener noreferrer">LinkedIn</a>
        <span>&middot;</span>
        <span className="f-email-wrap">
          <a href="mailto:grathish49@gmail.com" onClick={(e) => { e.preventDefault(); copyEmail(); }}>grathish49@gmail.com</a>
          <button className="f-copy-btn" onClick={copyEmail} aria-label="Copy email address">
            {copied
              ? <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{color:'var(--fg)'}}><polyline points="2,7 5.5,11 12,3"/></svg>
              : <Icon.Copy />}
          </button>
        </span>
      </div>
    </footer>
  );
}

/* ─── Page component ─── */

export function HomePage() {
  const [activeIdx, setActiveIdx] = useState(0);
  const demo = DEMOS[activeIdx];
  const Panel = PANELS[demo.id];

  return (
    <div className="home-page" style={{ overflowX: 'hidden', overflowY: 'auto', height: '100vh' }}>
      {/* Nav */}
      <header className="nav">
        <div className="nav-inner">
          <Link className="brand" href="/">
            <span className="brand-mark">A</span>
            <span>Atlas</span>
          </Link>
          <nav className="nav-links" aria-label="Main navigation">
            <a className="nav-link" href="#problem">Problem</a>
            <a className="nav-link" href="#how">How Atlas works</a>
            <a className="nav-link" href="#difference">The difference</a>
            <a className="nav-link" href="#hood">Under the hood</a>
            <a className="nav-link" href="#who">Who it&apos;s for</a>
          </nav>
          <div style={{ display: 'flex', gap: 8 }}>
            <Link className="btn btn-primary" href="/app">Open Atlas &rarr;</Link>
          </div>
        </div>
      </header>

      <main id="main-content">
      {/* Hero */}
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

        {/* Demo section intro */}
        <div className="demo-intro">
          <div className="demo-intro-label">Interactive demo</div>
          <p className="demo-intro-head">Every query type. Each one shows its work.</p>
          <p className="demo-intro-sub">Pick a category. See how Atlas reads the question, picks a strategy, and grounds every answer in the graph.</p>
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
            <h2>Alzheimer&apos;s research lives in ten thousand disconnected rows.</h2>
            <p className="section-lede">Six authoritative sources (ontologies, nomenclatures, drug registries, biomarker meta-analyses), each with its own IDs. Answering one question means opening six tabs and reconciling in your head.</p>
          </div>
          <div className="problem-grid">
            <div className="prob-card">
              <div className="prob-num">01</div>
              <h3 className="prob-title">Fragmented sources</h3>
              <p className="prob-body">Thousands of sources. No shared naming standard. The same gene has a different ID in every database.</p>
            </div>
            <div className="prob-card">
              <div className="prob-num">02</div>
              <h3 className="prob-title">Chat tools don&apos;t show their work</h3>
              <p className="prob-body">General assistants hallucinate p-values. Researchers need nodes and edges, not trust in a paragraph.</p>
            </div>
            <div className="prob-card">
              <div className="prob-num">03</div>
              <h3 className="prob-title">Dashboards don&apos;t answer questions</h3>
              <p className="prob-body">BI tools show what you pre-built. They can&apos;t answer &quot;which Phase 3 drugs target APOE-adjacent pathways?&quot; without a custom build.</p>
            </div>
          </div>
        </div>
      </section>

      <KnowledgeGraphSchema />

      {/* How it works */}
      <section className="band" id="how">
        <div className="wrap">
          <div className="section-head">
            <div className="section-label">How Atlas works</div>
            <h2>Questions are classified, routed, and grounded in the graph.</h2>
            <p className="section-lede">Intent classified. Subgraph pulled. Model writes from context only. Every claim traces to a node.</p>
          </div>
          <div className="how-grid">
            <div className="how-steps">
              <div className="step">
                <div className="step-num">01</div>
                <div>
                  <h4 className="step-title">Classify intent</h4>
                  <p className="step-body">Rule-based, sub-millisecond, no API call. Labels one of seven classes; declines out-of-scope. Shows its reasoning.</p>
                </div>
              </div>
              <div className="step">
                <div className="step-num">02</div>
                <div>
                  <h4 className="step-title">Select a retrieval strategy</h4>
                  <p className="step-body">Each intent maps to a named strategy: a subgraph traversal tuned to that question class.</p>
                </div>
              </div>
              <div className="step">
                <div className="step-num">03</div>
                <div>
                  <h4 className="step-title">Retrieve the subgraph</h4>
                  <p className="step-body">Pulls only what&apos;s relevant: biomarkers with effect size and p-value, drugs with phase and pathway links. Typed structure, not free text.</p>
                </div>
              </div>
              <div className="step">
                <div className="step-num">04</div>
                <div>
                  <h4 className="step-title">Synthesise, but show the evidence</h4>
                  <p className="step-body">Model writes at temperature zero from retrieved context only. Evidence renders beside the answer, with every number mapped to a row.</p>
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
            <h2>Six query classes, each with a tailored view.</h2>
            <p className="section-lede">Intent shapes the view. Biomarker evidence looks different from drug evidence. The UI follows the data structure.</p>
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
              <p className="cap-body">Compounds grouped by status from approved to discontinued, with phase, ChEBI ID, and pathway targets per drug.</p>
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
              <p className="cap-body">Which drugs target which pathways, with action type (antibody, small-molecule, inhibitor, gene-therapy) and primary-target flags.</p>
              <div className="cap-example">&ldquo;Which biological pathways does lecanemab affect?&rdquo;</div>
            </div>
            <div className="cap-cell">
              <div className="cap-icon">05 &middot; GENE_PROTEIN</div>
              <h4 className="cap-title">Gene&ndash;protein associations</h4>
              <p className="cap-body">Gene&ndash;protein pairs with HGNC and UniProt IDs, GWAS evidence labels, and the pathways the protein participates in.</p>
              <div className="cap-example">&ldquo;Which genes are most strongly associated with AD by GWAS?&rdquo;</div>
            </div>
            <div className="cap-cell">
              <div className="cap-icon">06 &middot; GENERAL_AD</div>
              <h4 className="cap-title">Composite overview</h4>
              <p className="cap-body">A tabbed view combining biomarkers, drugs, phenotypes, and genes for questions that don&apos;t fit a single class.</p>
              <div className="cap-example">&ldquo;Give me an overview of the AD landscape.&rdquo;</div>
            </div>
          </div>
        </div>
      </section>

      {/* The Difference */}
      {/* Why it works differently */}
      <section className="band" id="difference">
        <div className="wrap">
          <div className="section-head">
            <div className="section-label">Why it works differently</div>
            <h2>Most &ldquo;graph RAG&rdquo; retrieves text. Atlas retrieves structure.</h2>
            <p className="section-lede">Assistants guess. Dashboards only answer what you pre-built. Atlas handles open-ended questions from curated data, with every claim traceable.</p>
          </div>
          <div className="problem-grid">
            <div className="prob-card">
              <div className="prob-num">01</div>
              <h3 className="prob-title">Canonical identity, not fuzzy strings</h3>
              <p className="prob-body">Every entity has a real ontology ID. Aβ42 in a biomarker study and Aβ42 in a trial record are the same node, not two strings that happen to match.</p>
            </div>
            <div className="prob-card">
              <div className="prob-num">02</div>
              <h3 className="prob-title">Edges that carry the evidence</h3>
              <p className="prob-body">Biomarker edges carry effect size and p-value. Drug edges carry phase and approval status. Numbers live in the graph; the model reads them out.</p>
            </div>
            <div className="prob-card">
              <div className="prob-num">03</div>
              <h3 className="prob-title">Reconciled, not concatenated</h3>
              <p className="prob-body">Six ontologies plus AlzForum, joined once at build time. APP gene → APP protein → drug → amyloid pathway. Not re-guessed per query.</p>
            </div>
          </div>

          <div className="compare-table" style={{ marginTop: '2.5rem' }}>
            <div className="compare-grid">
              <div className="compare-feat-head"></div>
              <div className="compare-col-head compare-atlas-head">Atlas</div>
              <div className="compare-col-head">General chatbot</div>
              <div className="compare-col-head">BI dashboard</div>

              {[
                { feat: 'Takes natural-language questions',        atlas: 'yes', chat: 'yes',     bi: 'partial' },
                { feat: 'Answers open-ended, novel questions',     atlas: 'yes', chat: 'yes',     bi: 'no'      },
                { feat: 'Grounded in curated, authoritative data', atlas: 'yes', chat: 'partial', bi: 'yes'     },
                { feat: 'Exact, reproducible figures',             atlas: 'yes', chat: 'partial', bi: 'yes'     },
                { feat: 'Every claim traceable to its source',     atlas: 'yes', chat: 'partial', bi: 'yes'     },
                { feat: 'Structured knowledge with canonical IDs', atlas: 'yes', chat: 'no',      bi: 'no'      },
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
          <p className="compare-legend">&#10003; yes &nbsp;&middot;&nbsp; ~ partial, with caveats &nbsp;&middot;&nbsp; - no</p>
          <p className="section-lede" style={{ textAlign: 'center', maxWidth: '100%', marginTop: '1.5rem' }}>Regular RAG retrieves text. Atlas retrieves structure.</p>
        </div>
      </section>

      {/* Under the hood */}
      <section className="band" id="hood">
        <div className="wrap">
          <div className="section-head">
            <div className="section-label">Under the hood</div>
            <h2>Two phases. One source of truth.</h2>
            <p className="section-lede">Built once. Queried many times. Every answer reads from the same graph.</p>
          </div>
          <ArchDiagram />
        </div>
      </section>

      {/* Who */}
      <section className="band" id="who">
        <div className="wrap">
          <div className="section-head">
            <div className="section-label">Who it&apos;s for</div>
            <h2>Built for people comfortable with p-values.</h2>
            <p className="section-lede">Not a consumer assistant. An expert tool for anyone who lives in AD evidence.</p>
          </div>
          <div className="audience-grid">
            <div className="aud-card">
              <div className="aud-role">Drug researcher</div>
              <p className="aud-quote">&ldquo;I need to see the landscape. Which compounds are approved, which are still running, what pathway they hit, and where the failures clustered.&rdquo;</p>
              <div className="aud-person">
                <div className="aud-avatar">DR</div>
                <div>
                  <div className="aud-name">Preclinical discovery teams</div>
                  <div className="aud-title">Pharma &amp; Biotech Labs</div>
                </div>
              </div>
            </div>
            <div className="aud-card">
              <div className="aud-role">Clinician</div>
              <p className="aud-quote">&ldquo;The onset and frequency of a phenotype matters more to me than the prose describing it. Give me the HPO tags and let me sort.&rdquo;</p>
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
              <p className="aud-quote">&ldquo;Show me exactly what went into the prompt. I want the raw context markdown so I can pipe it into a notebook and tune retrieval.&rdquo;</p>
              <div className="aud-person">
                <div className="aud-avatar">DS</div>
                <div>
                  <div className="aud-name">Computational biology</div>
                  <div className="aud-title">Research &amp; Informatics Companies</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="cta-band">
        <h2>Start with a question.</h2>
        <p>No setup. No API keys. Just ask.</p>
        <Link className="btn btn-primary" href="/app" style={{ padding: '13px 22px', fontSize: 15 }}>Open Atlas &rarr;</Link>
      </section>

      </main>
      <FooterWithCopy />
    </div>
  );
}
