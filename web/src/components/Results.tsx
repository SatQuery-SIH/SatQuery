import { useEffect, useState } from "react";
import { api } from "../api";
import { Markdown } from "../md";
import { formatBytes } from "../format";
import type { ArtifactRef, Claim, RunBundle } from "../types";
import { isWithheldClaim } from "../types";

function ProvenanceChips({ c }: { c: Claim }) {
  const p = c.provenance ?? {};
  const bits: [string, string | undefined][] = [
    ["tool", p.tool],
    ["model", p.model],
    ["seat", p.seat],
    ["sha", p.sha256 ?? p.gguf_sha256],
  ];
  return (
    <div className="prov-chips">
      {bits
        .filter(([, v]) => v)
        .map(([k, v]) => (
          <span className="prov-chip" key={k} title={v}>
            {k}={String(v).length > 24 ? `${String(v).slice(0, 10)}…` : v}
          </span>
        ))}
      {c.confidence?.level && (
        <span className={`prov-chip lvl-${c.confidence.level}`}>{c.confidence.level}</span>
      )}
    </div>
  );
}

export function ClaimList({ claims }: { claims: Claim[] }) {
  if (!claims.length) return <div className="muted">no claims in packet</div>;
  return (
    <div className="claim-list">
      {claims.map((c) => {
        const w = isWithheldClaim(c);
        const value = typeof c.value === "string" ? c.value : JSON.stringify(c.value);
        return (
          <div key={c.id} className={`claim ${w ? "claim-withheld" : ""}`}>
            <div className="claim-head">
              <span className="claim-id">{c.id}</span>
              <span className="claim-pred">{c.predicate}</span>
              {w && <span className="withheld-badge">withheld</span>}
            </div>
            <div className="claim-value" title={value}>
              {value}
            </div>
            {c.confidence?.basis && (
              <div className="claim-basis">{c.confidence.basis}</div>
            )}
            <ProvenanceChips c={c} />
          </div>
        );
      })}
    </div>
  );
}

// GET just to read content-length — the server has no HEAD route (FastAPI
// GET routes 405 on HEAD). Abort right after headers; abort on cleanup.
function useContentLength(url: string): number | null {
  const [len, setLen] = useState<number | null>(null);
  useEffect(() => {
    const ac = new AbortController();
    fetch(url, { signal: ac.signal })
      .then((res) => {
        if (res.ok) {
          const cl = res.headers.get("content-length");
          if (cl != null && Number(cl) > 0) setLen(Number(cl));
        }
      })
      .catch(() => {})
      .finally(() => ac.abort());
    return () => ac.abort();
  }, [url]);
  return len;
}

function GeoTiffTile({ a }: { a: ArtifactRef }) {
  const len = useContentLength(api.artifactUrl(a.url));
  return (
    <a
      className="artifact"
      href={api.artifactUrl(a.url)}
      target="_blank"
      rel="noreferrer"
      title={`${a.type} — ${a.path}`}
      download={a.name}
    >
      <div className="artifact-file">⬇ GeoTIFF · download</div>
      <span className="artifact-name">
        {a.name} <em>{a.type}</em>
        {len != null && <span className="artifact-size">{formatBytes(len)}</span>}
      </span>
    </a>
  );
}

const ART_GROUPS: [string, string][] = [
  ["overlay", "overlays"],
  ["image", "images"],
  ["geo_export", "GeoTIFF exports"],
];

export function ArtifactsGrid({ bundle }: { bundle: RunBundle }) {
  const arts = bundle.artifacts ?? [];
  const [lightbox, setLightbox] = useState<ArtifactRef | null>(null);

  useEffect(() => {
    if (!lightbox) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setLightbox(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightbox]);

  if (!arts.length) return null;

  // group by the API `type` field in fixed order; unknown types get their own
  // group labeled with the raw type string. (The API has no "mask" type —
  // geo_export entries ARE the masks; don't invent types from filenames.)
  const groups: { type: string; label: string; items: ArtifactRef[] }[] = [];
  for (const [type, label] of ART_GROUPS) {
    const items = arts.filter((a) => a.type === type);
    if (items.length) groups.push({ type, label, items });
  }
  const known = new Set(ART_GROUPS.map(([t]) => t));
  for (const a of arts) {
    if (known.has(a.type)) continue;
    let g = groups.find((x) => x.type === a.type);
    if (!g) {
      g = { type: a.type, label: a.type, items: [] };
      groups.push(g);
    }
    g.items.push(a);
  }

  const tile = (a: ArtifactRef) => {
    if (/\.(png|jpe?g)$/i.test(a.name)) {
      return (
        <a
          key={a.name}
          className="artifact"
          href={api.artifactUrl(a.url)}
          target="_blank"
          rel="noreferrer"
          title={`${a.type} — ${a.path}`}
          onClick={(e) => {
            // plain left-click → lightbox; modifier/middle clicks still open a tab
            if (e.ctrlKey || e.metaKey || e.shiftKey || e.button !== 0) return;
            e.preventDefault();
            setLightbox(a);
          }}
        >
          <img src={api.artifactUrl(a.url)} alt={a.name} loading="lazy" />
          <span className="artifact-name">
            {a.name} <em>{a.type}</em>
          </span>
        </a>
      );
    }
    if (a.type === "geo_export") return <GeoTiffTile key={a.name} a={a} />;
    return (
      <a
        key={a.name}
        className="artifact"
        href={api.artifactUrl(a.url)}
        target="_blank"
        rel="noreferrer"
        title={`${a.type} — ${a.path}`}
      >
        <div className="artifact-file">⬇ {a.name}</div>
        <span className="artifact-name">
          {a.name} <em>{a.type}</em>
        </span>
      </a>
    );
  };

  return (
    <div className="artifacts">
      {groups.map((g) => (
        <div className="artifact-group" data-testid={`artifact-group-${g.type}`} key={g.type}>
          <h4 className="artifact-group-title">
            {g.label} ({g.items.length})
          </h4>
          <div className="artifacts-grid">{g.items.map(tile)}</div>
        </div>
      ))}
      {lightbox && (
        <div
          className="lightbox"
          role="dialog"
          aria-modal="true"
          data-testid="lightbox"
          onClick={(e) => {
            if (e.target === e.currentTarget) setLightbox(null);
          }}
        >
          <div className="lightbox-body">
            <img src={api.artifactUrl(lightbox.url)} alt={lightbox.name} />
            <div className="lightbox-caption">
              {lightbox.name} · {lightbox.type}
            </div>
            <div className="lightbox-actions">
              <a
                href={api.artifactUrl(lightbox.url)}
                target="_blank"
                rel="noreferrer"
              >
                open original ↗
              </a>
              <button className="btn-mini" autoFocus onClick={() => setLightbox(null)}>
                close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function Results({ bundle }: { bundle: RunBundle }) {
  const claims = bundle.evidence_packet?.claims ?? [];
  const rep = bundle.report ?? {};
  const refused = bundle.plan?.supported === false;
  const answerText = bundle.visible_answer ?? bundle.answer ?? "(no answer)";
  const audit = rep.narration_check as { ok?: boolean } | null | undefined;
  const completeS = bundle.trace?.complete_s;
  const refusal = bundle.plan?.refusal;
  return (
    <div className="results">
      <section className={`answer-card${refused ? " refused" : ""}`} data-testid="answer-card">
        <div className="answer-head">
          <h3>{refused ? "unsupported query — refused by the planner" : "answer"}</h3>
          <span className="answer-chips">
            {audit && (
              <span className={`audit-chip ${audit.ok ? "ok" : "bad"}`}>
                narration audit {audit.ok ? "passed" : "flagged"}
              </span>
            )}
            {completeS != null && <span className="audit-chip">{completeS} s</span>}
          </span>
        </div>
        <div className="answer-body">
          <Markdown text={answerText} />
        </div>
        {refused &&
          refusal != null &&
          !answerText.includes(String(refusal).trim()) && (
            <div className="refusal-reason">planner reason: {String(refusal)}</div>
          )}
      </section>

      <section className="panel">
        <h3>
          evidence packet{" "}
          <span className="muted">
            {claims.length} claims ·{" "}
            {claims.filter(isWithheldClaim).length} withheld
          </span>
        </h3>
        <ClaimList claims={claims} />
        {(bundle.evidence_packet?.limitations ?? []).length > 0 && (
          <div className="limitations">
            <strong>limitations:</strong>
            <ul>
              {(bundle.evidence_packet!.limitations as string[]).map((l, i) => (
                <li key={i}>{l}</li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <ArtifactsGrid bundle={bundle} />

      <details className="panel report">
        <summary>
          report · findings · measurement card · confidence hierarchy
        </summary>
        {rep.findings && <Markdown text={rep.findings} />}
        {rep.measurement && <Markdown text={rep.measurement} />}
        {rep.confidence && <Markdown text={rep.confidence} />}
        {rep.narration_check && (
          <div className="muted">
            narration audit:{" "}
            {(rep.narration_check as { ok?: boolean }).ok ? "passed" : "flagged"}
          </div>
        )}
      </details>
    </div>
  );
}
