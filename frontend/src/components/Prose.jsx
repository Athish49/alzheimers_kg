function renderInline(s) {
  const parts = [];
  const re = /(\*\*[^*]+\*\*)|(`[^`]+`)|(\*[^*]+\*)/g;
  let last = 0;
  let k = 0;
  let m;
  while ((m = re.exec(s)) !== null) {
    if (m.index > last) parts.push(s.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) parts.push(<strong key={k++}>{tok.slice(2, -2)}</strong>);
    else if (tok.startsWith("`")) parts.push(<code key={k++}>{tok.slice(1, -1)}</code>);
    else parts.push(<em key={k++}>{tok.slice(1, -1)}</em>);
    last = m.index + tok.length;
  }
  if (last < s.length) parts.push(s.slice(last));
  return parts;
}

export function Prose({ text }) {
  return (
    <div className="prose">
      {text.split("\n\n").map((p, i) => <p key={i}>{renderInline(p)}</p>)}
    </div>
  );
}
