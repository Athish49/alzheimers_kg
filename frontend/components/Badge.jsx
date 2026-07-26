const variants = {
  intent: { background: "var(--fg)", color: "var(--bg)" },
  strategy: { background: "var(--bg-sunken)", color: "var(--fg-secondary)" },
  outline: { background: "transparent", color: "var(--fg-secondary)", border: "1px solid var(--border-strong)" },
  accent: { background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" },
  solid: { background: "var(--fg-secondary)", color: "var(--bg)" },
  muted: { background: "var(--bg-muted)", color: "var(--fg-muted)" },
};

export function Badge({ variant = "strategy", children, mono = true, label, style = {} }) {
  if (label) {
    return (
      <span style={{
        display: "inline-flex", flexDirection: "column", alignItems: "flex-start",
        padding: "4px 10px", borderRadius: 6,
        ...variants[variant],
        ...style,
      }}>
        <span style={{
          fontSize: 9, fontWeight: 600, letterSpacing: "0.06em",
          textTransform: "uppercase", opacity: 0.65, lineHeight: 1,
          fontFamily: "var(--font-sans)",
        }}>{label}</span>
        <span style={{
          fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)",
          fontSize: 11, fontWeight: 500, letterSpacing: "0.02em",
          lineHeight: 1.4, marginTop: 1,
          display: "inline-flex", alignItems: "center", gap: 5,
        }}>{children}</span>
      </span>
    );
  }
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      padding: "3px 8px", borderRadius: 5,
      fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)",
      fontSize: 11, fontWeight: 500, letterSpacing: "0.02em",
      lineHeight: 1.3,
      ...variants[variant],
      ...style,
    }}>{children}</span>
  );
}

const glyphs = {
  BIOMARKER: "◐",
  DRUG_TRIAL: "◇",
  GENE_PROTEIN: "✕",
  PHENOTYPE: "◌",
  PATHWAY: "↬",
  GENERAL_AD: "◯",
};

export function CategoryGlyph({ id }) {
  return (
    <span style={{
      fontFamily: "var(--font-mono)", fontSize: 16,
      color: "var(--fg-secondary)", lineHeight: 1,
    }}>{glyphs[id] || "·"}</span>
  );
}
