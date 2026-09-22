
import { api } from "../api";
import { Markdown } from "../md";
import type { Claim, RunBundle } from "../types";
import { isWithheldClaim } from "../types";
import { TraceView } from "./TraceView";

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
        return (
          <div key={c.id} className={`claim ${w ? "claim-withheld" : ""}`}>
            <div className="claim-head">
              <span className="claim-id">{c.id}</span>
              <span className="claim-pred">{c.predicate}</span>
              {w && <span className="withheld-badge">withheld</span>}
            </div>
            <div className="claim-value">
              {typeof c.value === "string" ? c.value : JSON.stringify(c.value)}
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

export function ArtifactsGrid({ bundle }: { bundle: RunBundle }) {
  const arts = bundle.artifacts ?? [];
  if (!arts.length) return null;
  return (
    <div className="artifacts-grid">
      {arts.map((a) => (
        <a
          key={a.name}
          className="artifact"
          href={api.artifactUrl(a.url)}
          target="_blank"
          rel="noreferrer"
          title={`${a.type} — ${a.path}`}
        >
          {a.name.match(/\.(png|jpe?g)$/i) ? (
            <img src={api.artifactUrl(a.url)} alt={a.name} loading="lazy" />
          ) : (
            <div className="artifact-file">⬇ {a.name}</div>
          )}
          <span className="artifact-name">
            {a.name} <em>{a.type}</em>
          </span>
        </a>
      ))}
    </div>
  );
}

export function Results({ bundle }: { bundle: RunBundle }) {
  const claims = bundle.evidence_packet?.claims ?? [];
  const rep = bundle.report ?? {};
  const refused = bundle.plan?.supported === false;
  return (
    <div className="results">
      <TraceView bundle={bundle} />

      <section className="panel">
        <h3>answer</h3>
        <div className={`answer ${refused ? "refused" : ""}`}>
          <Markdown text={bundle.visible_answer ?? bundle.answer ?? "(no answer)"} />
        </div>
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

      <section className="panel report">
        <h3>report</h3>
        {rep.findings && <Markdown text={rep.findings} />}
        {rep.measurement && <Markdown text={rep.measurement} />}
        {rep.confidence && <Markdown text={rep.confidence} />}
        {rep.narration_check && (
          <div className="muted">
            narration audit:{" "}
            {(rep.narration_check as { ok?: boolean }).ok ? "passed" : "flagged"}
          </div>
        )}
      </section>
    </div>
  );
}
