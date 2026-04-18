import { Link } from 'react-router-dom';
import '../styles/home.css';

export function HomePage() {
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
            <a className="nav-link" href="#how">How it works</a>
            <a className="nav-link" href="#who">Who it's for</a>
            <a className="nav-link" href="#capabilities">Capabilities</a>
          </nav>
          <div style={{ display: 'flex', gap: 8 }}>
            <a className="btn btn-ghost" href="#">Docs</a>
            <Link className="btn btn-primary" to="/app">Open Atlas &rarr;</Link>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="hero">
        <span className="eyebrow"><span className="dot"></span> Knowledge graph &middot; Alzheimer's disease</span>
        <h1 className="hero-title">A research interface <em>for the shape</em> of Alzheimer's data.</h1>
        <p className="hero-sub">
          Atlas is a GraphRAG workbench over a curated Alzheimer's knowledge graph. Ask in natural language — get grounded answers backed by biomarker effect sizes, trial phases, phenotype frequencies, and gene-to-pathway evidence you can actually trace.
        </p>
        <div className="hero-cta">
          <Link className="btn btn-primary" to="/app">Open the workbench &rarr;</Link>
          <a className="btn btn-outline" href="#how">See how it works</a>
        </div>

        {/* Hero mock preview */}
        <div className="hero-visual" aria-hidden="true">
          <div className="visual-chrome">
            <span className="tl"></span><span className="tl"></span><span className="tl"></span>
            <div style={{ marginLeft: 12, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--fg-subtle)' }}>atlas.session &middot; AD_BIOMARKERS_V1</div>
          </div>
          <div className="visual-body">
            <div className="visual-chat">
              <div className="home-msg-user">What CSF biomarkers are elevated in Alzheimer's disease?</div>
              <div className="cls-row">
                <span className="badge badge-intent">BIOMARKER</span>
                <span className="badge badge-strategy">AD_BIOMARKERS_V1</span>
                <span className="badge-note">Detected biomarker focus with fluid keyword "CSF"</span>
              </div>
              <div className="visual-answer prose">
                <p>Cerebrospinal fluid biomarkers consistently elevated in Alzheimer's disease reflect neuronal injury and tau pathology. The strongest signals come from <strong>t-tau</strong> and <strong>p-tau181</strong>, while <strong>A&beta;42</strong> is notably decreased.</p>
                <p>Effect sizes are drawn from meta-analyses indexed in the graph, with <code>p &lt; 0.001</code> across reports.</p>
              </div>
            </div>
            <div className="visual-ctx">
              <div className="home-ctx-label">Evidence &middot; CSF &middot; Increased in AD</div>
              <div className="home-ctx-row">
                <div><span className="name">t-tau</span> <span className="meta">Protein</span></div>
                <div className="arrow-up">&uarr; 2.1</div>
                <div className="meta">p&lt;0.001</div>
              </div>
              <div className="home-ctx-row">
                <div><span className="name">p-tau181</span> <span className="meta">Protein</span></div>
                <div className="arrow-up">&uarr; 1.9</div>
                <div className="meta">p&lt;0.001</div>
              </div>
              <div className="home-ctx-row">
                <div><span className="name">NfL</span> <span className="meta">Protein</span></div>
                <div className="arrow-up">&uarr; 1.3</div>
                <div className="meta">p&lt;0.01</div>
              </div>
              <div className="home-ctx-label" style={{ marginTop: 20 }}>CSF &middot; Decreased in AD</div>
              <div className="home-ctx-row">
                <div><span className="name">A&beta;42</span> <span className="meta">Peptide</span></div>
                <div className="arrow-down">&darr; 1.8</div>
                <div className="meta">p&lt;0.001</div>
              </div>
              <div className="home-ctx-row">
                <div><span className="name">A&beta;42/40 ratio</span> <span className="meta">Ratio</span></div>
                <div className="arrow-down">&darr; 1.5</div>
                <div className="meta">p&lt;0.001</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Problem */}
      <section className="band" id="problem">
        <div className="wrap">
          <div className="section-head">
            <div className="section-label">The problem</div>
            <h2>Alzheimer's research lives in <em>ten thousand</em> disconnected rows.</h2>
            <p className="section-lede">Biomarker studies, trial registries, gene databases, phenotype ontologies, pathway annotations — all authoritative, all siloed. Answering a single question usually means opening six browser tabs and reconciling them in your head.</p>
          </div>
          <div className="problem-grid">
            <div className="prob-card">
              <div className="prob-num">01</div>
              <h3 className="prob-title">Fragmented sources</h3>
              <p className="prob-body">HPO for phenotypes. CHEBI for drugs. UniProt for proteins. GWAS catalogs for risk. Each with its own IDs, its own update cadence, its own semantics.</p>
            </div>
            <div className="prob-card">
              <div className="prob-num">02</div>
              <h3 className="prob-title">Chat tools don't show their work</h3>
              <p className="prob-body">General-purpose assistants will happily hallucinate a p-value. Researchers need to see the specific nodes and edges behind every claim — not trust a paragraph.</p>
            </div>
            <div className="prob-card">
              <div className="prob-num">03</div>
              <h3 className="prob-title">Dashboards don't answer questions</h3>
              <p className="prob-body">Classical BI tools show you what you already asked to see. They can't respond to "which Phase 3 drugs target APOE-adjacent pathways?" without a custom build.</p>
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
            <p className="section-lede">Every query passes through an intent classifier that picks the right retrieval strategy, pulls the relevant subgraph, and hands structured context to the language model. Nothing is invented — every claim traces back to a node.</p>
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
                  <p className="step-body">Each intent routes to a named strategy (<code>AD_BIOMARKERS_V1</code>, <code>AD_DRUGS_PATHWAYS_V1</code>, &hellip;) — a specific subgraph traversal tuned for that class of question.</p>
                </div>
              </div>
              <div className="step">
                <div className="step-num">03</div>
                <div>
                  <h4 className="step-title">Retrieve the subgraph</h4>
                  <p className="step-body">The strategy pulls the relevant nodes and edges — biomarker values with fluid, direction, effect size, p-value; drugs with phase, CHEBI ID, and pathway links; and so on.</p>
                </div>
              </div>
              <div className="step">
                <div className="step-num">04</div>
                <div>
                  <h4 className="step-title">Synthesise, but show the evidence</h4>
                  <p className="step-body">The language model writes a grounded answer. The retrieved context is rendered alongside it as structured data — so every number in the prose corresponds to a row you can inspect.</p>
                </div>
              </div>
            </div>

            <div className="diagram" aria-hidden="true">
              <div className="diag-row">
                <span className="tag">Query</span>
                <span className="val">"Which Phase 3 drugs target the amyloid pathway?"</span>
              </div>
              <div className="diag-arrow">&darr;</div>
              <div className="diag-row">
                <span className="tag">Intent</span>
                <span className="val"><span className="badge badge-intent">DRUG_TRIAL</span></span>
              </div>
              <div className="diag-sub">Notes: matched pathway keyword "amyloid"; phase filter "Phase 3".</div>
              <div className="diag-arrow">&darr;</div>
              <div className="diag-row">
                <span className="tag">Strategy</span>
                <span className="val"><span className="badge badge-strategy">AD_DRUGS_PATHWAYS_V1</span></span>
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

      {/* Who */}
      <section className="band" id="who">
        <div className="wrap">
          <div className="section-head">
            <div className="section-label">Who it's for</div>
            <h2>Built for people comfortable with <em>p-values</em>.</h2>
            <p className="section-lede">Atlas is not a consumer assistant. It's an expert tool for the three constituencies that spend the most time reconciling AD evidence.</p>
          </div>
          <div className="audience-grid">
            <div className="aud-card">
              <div className="aud-role">Drug researcher</div>
              <p className="aud-quote">"I need to see the <em>landscape</em>. Which compounds are approved, which are still running, what pathway they hit, and where the failures clustered."</p>
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
              <p className="aud-quote">"The <em>onset</em> and <em>frequency</em> of a phenotype matters more to me than the prose describing it. Give me the HPO tags and let me sort."</p>
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
              <p className="aud-quote">"Show me <em>exactly</em> what went into the prompt. I want the raw context markdown so I can pipe it into a notebook and tune retrieval."</p>
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

      {/* Capabilities */}
      <section className="band" id="capabilities">
        <div className="wrap">
          <div className="section-head">
            <div className="section-label">Capabilities</div>
            <h2>Six query classes, each with a <em>tailored</em> view.</h2>
            <p className="section-lede">The intent you ask determines how the evidence is rendered. A biomarker question doesn't look like a drug question — the structure of the data is different, and the UI follows.</p>
          </div>

          <div className="cap-grid">
            <div className="cap-cell">
              <div className="cap-icon">01 &middot; BIOMARKER</div>
              <h4 className="cap-title">Biomarker evidence</h4>
              <p className="cap-body">Grouped by fluid, split by direction, sorted by effect size. See what goes up, what goes down, and how strongly.</p>
              <div className="cap-example">"Which plasma biomarkers are decreased in AD?"</div>
            </div>
            <div className="cap-cell">
              <div className="cap-icon">02 &middot; DRUG_TRIAL</div>
              <h4 className="cap-title">Drug landscape</h4>
              <p className="cap-body">Compounds grouped by status — approved to discontinued — with phase, CHEBI ID, and pathway targets per drug.</p>
              <div className="cap-example">"What amyloid-targeting drugs have FDA approval?"</div>
            </div>
            <div className="cap-cell">
              <div className="cap-icon">03 &middot; PHENOTYPE</div>
              <h4 className="cap-title">Clinical features</h4>
              <p className="cap-body">Phenotypes with HPO IDs, onset, and frequency weighting. Sortable by frequency, onset, or alphabetically.</p>
              <div className="cap-example">"At what stage does agitation typically appear?"</div>
            </div>
            <div className="cap-cell">
              <div className="cap-icon">04 &middot; PATHWAY</div>
              <h4 className="cap-title">Drug–pathway links</h4>
              <p className="cap-body">Which drugs target which pathways, with action type (inhibition, activation, modulation) and primary-target flags.</p>
              <div className="cap-example">"Which biological pathways does lecanemab affect?"</div>
            </div>
            <div className="cap-cell">
              <div className="cap-icon">05 &middot; GENE_PROTEIN</div>
              <h4 className="cap-title">Gene–protein associations</h4>
              <p className="cap-body">Gene–protein pairs with HGNC and UniProt IDs, GWAS evidence strength, and the pathways the protein participates in.</p>
              <div className="cap-example">"Which genes are most strongly associated with AD by GWAS?"</div>
            </div>
            <div className="cap-cell">
              <div className="cap-icon">06 &middot; GENERAL_AD</div>
              <h4 className="cap-title">Composite overview</h4>
              <p className="cap-body">A tabbed view combining biomarkers, drugs, phenotypes, and genes — for questions that don't fit a single class.</p>
              <div className="cap-example">"Give me an overview of the AD landscape."</div>
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
        <div>Atlas &middot; An Alzheimer's GraphRAG research interface</div>
        <div className="f-links">
          <a href="#">Methodology</a>
          <a href="#">Data sources</a>
          <a href="#">Contact</a>
        </div>
      </footer>
    </div>
  );
}
