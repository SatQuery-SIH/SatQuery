import { useState, type ReactNode } from "react";

// Controlled <details> that only mounts its body while open — collapsed
// sections contribute nothing to the page's visible text (de-jargon rule)
// and cost no renders/probes until opened.
export function Collapsible({
  summary,
  badge,
  defaultOpen = false,
  className = "",
  children,
}: {
  summary: ReactNode;
  badge?: ReactNode;
  defaultOpen?: boolean;
  className?: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <details className={`collapsible ${className}`} open={open}>
      <summary
        onClick={(e) => {
          e.preventDefault();
          setOpen((o) => !o);
        }}
      >
        <span className="collapsible-label">{summary}</span>
        {badge}
        <span className="collapsible-caret">{open ? "▴" : "▾"}</span>
      </summary>
      {open && <div className="collapsible-body">{children}</div>}
    </details>
  );
}
