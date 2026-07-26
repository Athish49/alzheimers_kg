'use client';
import { useState } from 'react';
import { Icon } from './Icons';
import { Badge, CategoryGlyph } from './Badge';
import { Prose } from './Prose';

export function AssistantMessage({ msg, isActive, onViewEvidence, onFollowup }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(msg.answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <div className={"msg msg-assistant " + (isActive ? "active" : "")}>
      <div className="cls-bar">
        <div style={{ display: 'inline-flex', flexDirection: 'column', gap: 4 }}>
          <span style={{
            fontSize: 9, fontWeight: 600, letterSpacing: "0.06em",
            textTransform: "uppercase", opacity: 0.65, lineHeight: 1,
            fontFamily: "var(--font-sans)", color: "var(--fg-secondary)"
          }}>Intent:</span>
          <Badge variant="intent">
            <CategoryGlyph id={msg.intent_type} />
            <span style={{ marginLeft: 2 }}>{msg.intent_type}</span>
          </Badge>
        </div>
        <div style={{ display: 'inline-flex', flexDirection: 'column', gap: 4 }}>
          <span style={{
            fontSize: 9, fontWeight: 600, letterSpacing: "0.06em",
            textTransform: "uppercase", opacity: 0.65, lineHeight: 1,
            fontFamily: "var(--font-sans)", color: "var(--fg-secondary)"
          }}>Retrieval Strategy:</span>
          <Badge variant="strategy">{msg.strategy}</Badge>
        </div>
      </div>

      <Prose text={msg.answer} />

      <div className="msg-actions">
        {msg.context ? (
          <button className="msg-action primary" onClick={onViewEvidence}>
            <Icon.Panel /> View graph evidence
          </button>
        ) : (
          <span className="msg-noevidence">No graph context retrieved</span>
        )}
        <button className="msg-action" onClick={copy}>
          <Icon.Copy /> {copied ? "Copied" : "Copy"}
        </button>
      </div>

      {msg.followups && msg.followups.length > 0 && (
        <div className="followups">
          <div className="followup-label">Follow-ups</div>
          <div className="followup-chips">
            {msg.followups.map((f, i) => (
              <button key={i} className="followup-chip" onClick={() => onFollowup(f)}>
                {f}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function LoadingMessage({ phase }) {
  const phases = ["Classifying", "Retrieving", "Generating"];
  return (
    <div className="msg msg-assistant loading">
      <div className="cls-bar">
        {phases.map((p, i) => (
          <div key={p} className={"phase " + (i < phase ? "done" : i === phase ? "active" : "")}>
            <span className="phase-dot" />
            <span>{p}</span>
          </div>
        ))}
      </div>
      <div className="skeleton-answer">
        <div className="skeleton-line" style={{ width: "90%" }} />
        <div className="skeleton-line" style={{ width: "75%" }} />
        <div className="skeleton-line" style={{ width: "60%" }} />
      </div>
    </div>
  );
}

export function ErrorMessage({ msg, onRetry }) {
  return (
    <div className="msg msg-error">
      <div className="err-head"><Icon.Alert /> {msg.title || "Error"}</div>
      <div className="err-body">{msg.body || "Something went wrong. Please try again."}</div>
      <button className="msg-action" onClick={onRetry}><Icon.Refresh /> Retry</button>
    </div>
  );
}
