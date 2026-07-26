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
  const lines = text.split("\n");
  const elements = [];
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim() === "") {
      i++;
      continue;
    }

    const h3 = line.match(/^### (.+)/);
    const h2 = line.match(/^## (.+)/);
    const h1 = line.match(/^# (.+)/);

    if (h3) {
      elements.push(<h3 key={key++}>{renderInline(h3[1])}</h3>);
      i++;
      continue;
    }
    if (h2) {
      elements.push(<h2 key={key++}>{renderInline(h2[1])}</h2>);
      i++;
      continue;
    }
    if (h1) {
      elements.push(<h1 key={key++}>{renderInline(h1[1])}</h1>);
      i++;
      continue;
    }

    if (/^\d+\. /.test(line)) {
      const items = [];
      let startNum = 1;
      while (i < lines.length && /^\d+\. /.test(lines[i])) {
        const numMatch = lines[i].match(/^(\d+)\. /);
        const n = numMatch ? parseInt(numMatch[1], 10) : 1;
        if (items.length === 0) startNum = n;
        items.push(<li key={key++}>{renderInline(lines[i].replace(/^\d+\. /, ""))}</li>);
        i++;
      }
      elements.push(<ol key={key++} start={startNum}>{items}</ol>);
      continue;
    }

    if (/^[-*] /.test(line)) {
      const items = [];
      while (i < lines.length && /^[-*] /.test(lines[i])) {
        items.push(<li key={key++}>{renderInline(lines[i].replace(/^[-*] /, ""))}</li>);
        i++;
      }
      elements.push(<ul key={key++}>{items}</ul>);
      continue;
    }

    const paraLines = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !/^#{1,3} /.test(lines[i]) &&
      !/^\d+\. /.test(lines[i]) &&
      !/^[-*] /.test(lines[i])
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    elements.push(<p key={key++}>{renderInline(paraLines.join(" "))}</p>);
  }

  return <div className="prose">{elements}</div>;
}
