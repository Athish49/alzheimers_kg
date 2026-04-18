import { useState } from 'react';
import { Icon } from './Icons';
import { Badge } from './Badge';
import { BiomarkerView, DrugView, PhenotypeView, PathwayView, GeneView, CompositeView } from './ContextViews';

export function ContextPanel({ message, pinnedQuestion, onClose }) {
  const [mode, setMode] = useState("structured");

  if (!message) {
    return (
      <div className="ctx-empty">
        <div className="ctx-empty-inner">
          <Icon.Panel size={18} />
          <div className="ctx-empty-title">Evidence panel</div>
          <div className="ctx-empty-body">Ask a question and graph evidence will appear here.</div>
        </div>
      </div>
    );
  }

  const { intent_type, context, strategy } = message;

  const renderView = () => {
    if (!context) {
      return (
        <div className="ev-notice">No graph nodes were retrieved for this query.</div>
      );
    }
    if (context.composite) return <CompositeView data={context} />;
    switch (intent_type) {
      case "BIOMARKER": return <BiomarkerView data={context} />;
      case "DRUG_TRIAL": return <DrugView data={context} />;
      case "PHENOTYPE": return <PhenotypeView data={context} />;
      case "PATHWAY": return <PathwayView data={context} />;
      case "GENE_PROTEIN": return <GeneView data={context} />;
      case "GENERAL_AD":
      case "OTHER":
        return <CompositeView data={context} />;
      default: return <CompositeView data={context} />;
    }
  };

  const rawMarkdown = () => {
    if (!context) return "(no context)";
    return JSON.stringify(context, null, 2);
  };

  const copyRaw = () => {
    navigator.clipboard?.writeText(rawMarkdown());
  };

  return (
    <>
      <div className="ctx-header">
        <div style={{ minWidth: 0, flex: 1 }}>
          <div className="ctx-kicker">Evidence for</div>
          <div className="ctx-title" title={pinnedQuestion}>{pinnedQuestion}</div>
          <div className="ctx-sub">
            <div style={{ display: 'inline-flex', flexDirection: 'column', gap: 4 }}>
              <span style={{
                fontSize: 9, fontWeight: 600, letterSpacing: "0.06em",
                textTransform: "uppercase", opacity: 0.65, lineHeight: 1,
                fontFamily: "var(--font-sans)", color: "var(--fg-secondary)"
              }}>Intent:</span>
              <Badge variant="intent">{intent_type}</Badge>
            </div>
            <div style={{ display: 'inline-flex', flexDirection: 'column', gap: 4 }}>
              <span style={{
                fontSize: 9, fontWeight: 600, letterSpacing: "0.06em",
                textTransform: "uppercase", opacity: 0.65, lineHeight: 1,
                fontFamily: "var(--font-sans)", color: "var(--fg-secondary)"
              }}>Retrieval Strategy:</span>
              <Badge variant="strategy">{strategy}</Badge>
            </div>
          </div>
        </div>
        <button className="icon-btn" onClick={onClose} title="Close evidence panel">
          <Icon.Close />
        </button>
      </div>

      <div className="ctx-mode">
        <div className="mode-toggle">
          <button className={mode === "structured" ? "active" : ""} onClick={() => setMode("structured")}>Structured</button>
          <button className={mode === "raw" ? "active" : ""} onClick={() => setMode("raw")}>Raw</button>
        </div>
        {mode === "raw" && (
          <button className="mini-btn" onClick={copyRaw}><Icon.Copy /> Copy</button>
        )}
      </div>

      <div className="ctx-body">
        {mode === "structured" ? renderView() : (
          <pre className="raw-context">{rawMarkdown()}</pre>
        )}
      </div>
    </>
  );
}
