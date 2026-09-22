// Minimal markdown renderer — handles the subset report/visible_answer use:
// ### headers, - bullets, **bold**, `code`, paragraphs. Zero deps.
import React from "react";

function inline(s: string, keyPrefix: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let m: RegExpExecArray | null;
  let last = 0;
  let i = 0;
  while ((m = re.exec(s))) {
    if (m.index > last) parts.push(s.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**"))
      parts.push(<strong key={`${keyPrefix}-${i}`}>{tok.slice(2, -2)}</strong>);
    else parts.push(<code key={`${keyPrefix}-${i}`}>{tok.slice(1, -1)}</code>);
    last = m.index + tok.length;
    i++;
  }
  if (last < s.length) parts.push(s.slice(last));
  return parts;
}

export function Markdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const out: React.ReactNode[] = [];
  let list: string[] = [];
  const flush = () => {
    if (list.length) {
      const items = list;
      out.push(
        <ul key={`ul-${out.length}`}>
          {items.map((li, j) => (
            <li key={j}>{inline(li, `li${out.length}-${j}`)}</li>
          ))}
        </ul>,
      );
      list = [];
    }
  };
  lines.forEach((ln, i) => {
    const t = ln.trimEnd();
    if (/^#{2,4}\s/.test(t)) {
      flush();
      const level = t.match(/^#+/)![0].length;
      const Tag = (`h${Math.min(level + 1, 5)}`) as "h3" | "h4" | "h5";
      out.push(<Tag key={i}>{inline(t.replace(/^#+\s*/, ""), `h${i}`)}</Tag>);
    } else if (/^[-*]\s/.test(t)) {
      list.push(t.replace(/^[-*]\s*/, ""));
    } else if (t === "") {
      flush();
    } else {
      flush();
      out.push(<p key={i}>{inline(t, `p${i}`)}</p>);
    }
  });
  flush();
  return <>{out}</>;
}
