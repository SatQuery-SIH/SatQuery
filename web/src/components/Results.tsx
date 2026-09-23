import { useEffect, useState } from "react";
import { api } from "../api";
import { Markdown } from "../md";
import { formatBytes, formatSecs } from "../format";
import {
  composeVerified,
  humanPredicate,
  humanValue,
  reportCheck,
  splitAnswer,
} from "../verified";
import type { ArtifactRef, Claim, RunBundle, UploadResponse } from "../types";
import { isWithheldClaim } from "../types";
import { baseName } from "../liveTrace";
import { Collapsible } from "./Collapsible";
import { Lightbox } from "./Lightbox";

// "answer model" / "remote-sensing model" for the two known seat roles; raw
// values for anything else (collapsed section only — real names allowed here)
const SEAT_WORDS: Record<string, string> = {
  narrator: "answer model",
  canonical: "remote-sensing model",
};

function ProvenanceChips({ c }: { c: Claim }) {
  const p = c.provenance ?? {};
  const bits: [string, string | undefined][] = [
    ["tool", p.tool],
    ["model", p.model ? baseName(p.model) : undefined],
    ["model role", p.seat ? (SEAT_WORDS[p.seat] ?? p.seat) : undefined],
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
  if (!claims.length) return <div className="muted">no claims recorded</div>;
  return (
    <div className="claim-list">
      {claims.map((c) => {
        const w = isWithheldClaim(c);
        const raw = typeof c.value === "string" ? c.value : JSON.stringify(c.value);
        const value = humanValue(c.value);
        return (
          <div key={c.id} className={`claim ${w ? "claim-withheld" : ""}`}>
            <div className="claim-head">
              <span
                className="claim-pred"
                title={`${c.predicate} · ${c.id}`}
              >
                {humanPredicate(c.predicate)}
              </span>
              {w && <span className="withheld-badge">withheld</span>}
            </div>
            <div className="claim-value" title={raw}>
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
  ["geo_export", "map-ready GeoTIFFs"],
];

export function ArtifactsGrid({ bundle }: { bundle: RunBundle }) {
  const arts = bundle.artifacts ?? [];
  const [lightbox, setLightbox] = useState<ArtifactRef | null>(null);

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
        <Lightbox
          src={api.artifactUrl(lightbox.url)}
          name={lightbox.name}
          type={lightbox.type}
          onClose={() => setLightbox(null)}
        />
      )}
    </div>
  );
}

// "image details" rows — works for a bound upload's detected block and for a
// run bundle's trace.gsd record (same gsd vocabulary, different shape)
export function ImageDetails({
  bundle,
  upload,
}: {
  bundle?: RunBundle | null;
  upload?: UploadResponse | null;
}) {
  const g = (bundle?.trace?.gsd ?? null) as Record<string, unknown> | null;
  const d = upload?.detected ?? null;
  const rows: [string, string][] = [];
  const push = (k: string, v: unknown) => {
    if (v != null && v !== "") rows.push([k, String(v)]);
  };
  if (g) {
    push("pixel size", g.gsd_m != null ? `${g.gsd_m} m/px` : "not detected");
    push("source", g.source);
    push("crs", g.crs);
    push("native size", g.native_width && g.native_height ? `${g.native_width}×${g.native_height}` : null);
    push("ingest", g.backend);
  } else if (d) {
    push("pixel size", d.gsd_m != null ? `${d.gsd_m} m/px` : "not detected");
    push("source", d.gsd_source);
    push("crs", d.crs);
    push("bands", d.bands);
    push("dtype", d.dtype);
    push("sar calibrated", d.calibrated == null ? null : d.calibrated ? "yes" : "no");
  }
  const notes = [
    g?.crs_note,
    g?.provenance,
    bundle?.trace?.ingest_note ?? upload?.ingest_note,
    ...(upload?.warnings ?? []),
  ].filter((x): x is string => typeof x === "string" && !!x);
  if (!rows.length && !notes.length) return null;
  return (
    <div className="image-details" data-testid="image-details">
      <div className="detected-grid">
        {rows.map(([k, v]) => (
          <div className="detected-row" key={k}>
            <span className="detected-key">{k}</span>
            <span className="detected-val">{v}</span>
          </div>
        ))}
      </div>
      {notes.length > 0 && (
        <ul className="detail-notes">
          {notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function Results({
  bundle,
  upload,
}: {
  bundle: RunBundle;
  upload?: UploadResponse | null;
}) {
  const claims = bundle.evidence_packet?.claims ?? [];
  const rep = bundle.report ?? {};
  const refused = bundle.plan?.supported === false;
  const parts = splitAnswer(bundle);
  const check = reportCheck(bundle);
  const completeS = bundle.trace?.complete_s;
  const refusal = bundle.plan?.refusal;
  const query = bundle.trace?.query;

  if (refused) {
    const answerText = bundle.visible_answer ?? bundle.answer ?? "";
    return (
      <div className="results">
        <section className="answer-card refused" data-testid="answer-card">
          <div className="answer-head">
            <h3>{"can't answer this with the available tools"}</h3>
            <span className="answer-chips">
              {completeS != null && (
                <span className="audit-chip">{formatSecs(completeS)}</span>
              )}
            </span>
          </div>
          <div className="answer-body">
            <Markdown text={parts.prose || answerText || "(no answer)"} />
          </div>
          {refusal != null &&
            !answerText.includes(String(refusal).trim()) && (
              <div className="refusal-reason">plan reason: {String(refusal)}</div>
            )}
        </section>
      </div>
    );
  }

  const { measured, withheld } = composeVerified(bundle);
  const hasFacts = measured.length + withheld.length > 0;
  const reportChip =
    parts.proseByModel && check
      ? check.ok
        ? { text: "report check: passed", cls: "ok", title: "" }
        : { text: "report check: flagged", cls: "bad", title: check.issues.join("\n") }
      : null;

  return (
    <div className="results">
      <section className="answer-card" data-testid="answer-card">
        <div className="answer-head">
          <h3>{query ? `“${query}”` : "answer"}</h3>
          <span className="answer-chips">
            {reportChip && (
              <span className={`audit-chip ${reportChip.cls}`} title={reportChip.title}>
                {reportChip.text}
              </span>
            )}
            {completeS != null && (
              <span className="audit-chip">{formatSecs(completeS)}</span>
            )}
          </span>
        </div>

        {hasFacts ? (
          <div className="facts">
            {measured.length === 0 && (
              <div className="fact fact-none">
                Nothing here could be measured with confidence.
              </div>
            )}
            {measured.map((f) => (
              <div
                key={f.key}
                className="fact fact-measured"
                data-testid="fact-measured"
                title={f.exact}
              >
                {f.text}
              </div>
            ))}
            {withheld.map((f) => (
              <div
                key={f.key}
                className="fact fact-withheld"
                data-testid="fact-withheld"
                title={f.exact}
              >
                <span className="withheld-tag">withheld</span>{" "}
                {f.text}
              </div>
            ))}
          </div>
        ) : (
          <div className="muted fact-none">
            No tool measurements for this question.
          </div>
        )}

        {parts.prose && (
          <Collapsible
            className="interpretation"
            summary={
              parts.proseByModel
                ? "interpretation — model-written, unverified"
                : "tool note"
            }
            badge={
              parts.flag ? (
                <span className="flag-tag">{parts.flag}</span>
              ) : undefined
            }
            defaultOpen={!hasFacts}
          >
            <div className="answer-body">
              <Markdown text={parts.prose} />
            </div>
            {check && !check.ok && check.issues.length > 0 && (
              <ul className="detail-notes flagged">
                {check.issues.map((i, k) => (
                  <li key={k}>{i}</li>
                ))}
              </ul>
            )}
          </Collapsible>
        )}
      </section>

      <Collapsible
        className="panel"
        summary={
          <>
            evidence &amp; measurements{" "}
            <span className="muted">
              {claims.length} claims ·{" "}
              {claims.filter(isWithheldClaim).length} withheld
            </span>
          </>
        }
      >
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
        {rep.measurement && <Markdown text={rep.measurement} />}
        {rep.confidence && <Markdown text={rep.confidence} />}
        {parts.toolFindings && (
          <>
            <h4 className="tool-log-title">tool log</h4>
            <Markdown text={parts.toolFindings} />
          </>
        )}
      </Collapsible>

      {(bundle.artifacts ?? []).length > 0 && (
        <Collapsible className="panel" summary="downloadable artifacts">
          <ArtifactsGrid bundle={bundle} />
        </Collapsible>
      )}

      {(bundle.trace?.gsd || upload) && (
        <Collapsible className="panel" summary="image details">
          <ImageDetails bundle={bundle} upload={upload} />
        </Collapsible>
      )}
    </div>
  );
}
