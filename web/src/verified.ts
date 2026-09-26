// Answer-card composition. The LOUD lines are what the tools MEASURED plus
// what they explicitly WITHHELD — composed only from evidence_packet claims
// and tool_outputs. Model-written prose never feeds these lines ("JSON card
// wins"); it is split out and labeled separately by splitAnswer().
import type { Claim, RunBundle } from "./types";
import { isWithheldClaim } from "./types";
import { humanTool } from "./liveTrace";

export interface Fact {
  key: string;
  text: string;
  // exact bundle value(s) behind a rounded display — shown on hover
  exact?: string;
  // claim id(s) or tool name the fact was composed from
  source: string;
}

export interface VerifiedFacts {
  measured: Fact[];
  withheld: Fact[];
}

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

const rec = (v: unknown): Record<string, unknown> | null =>
  v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;

export function fmtPct(v: number): string {
  if (v === 0) return "0";
  if (Math.abs(v) < 0.1) return v.toPrecision(2);
  return v.toFixed(1);
}
export const fmtInt = (v: number): string => v.toLocaleString("en-US");
export const fmtRatio = (v: number): string => v.toFixed(3);

const QUADRANT: Record<string, string> = {
  NE: "north-east",
  NW: "north-west",
  SE: "south-east",
  SW: "south-west",
  N: "north",
  S: "south",
  E: "east",
  W: "west",
};

const PREDICATE_LABELS: Record<string, string> = {
  change_detected: "change detected",
  changed_pixels: "changed pixels",
  radiometric_delta: "brightness change inside the change mask",
  radiometric_label: "brightness label",
  built_up_direction: "built-up direction",
  iou_vs_gt: "IoU vs reference mask",
  area_m2: "area (m²)",
  area_km2: "area (km²)",
  dominant_quadrant: "dominant quadrant",
  water_method: "water method",
  water_pixels: "water pixels",
  water_calibrated: "SAR calibrated",
  sdwi_label: "SAR water index",
  water_agreement: "optical/SAR water agreement",
  built_up_agreement: "built-up agreement",
  agreement_iou: "agreement IoU",
  agreement_grid: "comparison grid",
  coreg_transform_offset_m: "co-registration offset (m)",
  coreg_same_res: "same resolution",
  coreg_shift_px: "co-registration shift (px)",
  primary_transition: "main land-cover transition",
  largest_change_class: "largest change class",
  trend: "trend",
  cdvqa_answer: "change-QA answer",
  cdvqa_family: "question family",
  canonical_answer: "remote-sensing model answer",
};

export function humanPredicate(p: string): string {
  return PREDICATE_LABELS[p] ?? p.replace(/_/g, " ");
}

const VALUE_WORDS: Record<string, string> = {
  not_determined: "not determined",
  withheld_no_tool: "withheld — no tool for this",
  withheld_uncalibrated: "withheld — SAR not calibrated",
  withheld_misregistered: "withheld — images misaligned",
  both_support: "optical and SAR agree",
  both_absent: "absent in both",
  optical_only: "optical only",
  sar_only: "SAR only",
  disagree: "optical and SAR disagree",
  same_grid: "same grid",
};

// Plain words for claim values; free-text values (with spaces) pass through.
export function humanValue(v: unknown): string {
  if (v === null || v === undefined) return "not identified";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return String(v);
  if (typeof v === "string") {
    if (VALUE_WORDS[v]) return VALUE_WORDS[v];
    return /\s/.test(v) ? v : v.replace(/->/g, " → ").replace(/_/g, " ");
  }
  return JSON.stringify(v);
}

// "pixel shift inconclusive: weak_peak" -> " (weak peak)"
function basisReason(c: Claim): string {
  const b = c.confidence?.basis ?? "";
  const i = b.lastIndexOf(": ");
  return i === -1 ? "" : ` (${humanValue(b.slice(i + 2))})`;
}

const WATER_AGREEMENT_TEXT: Record<string, string> = {
  both_support: "optical and SAR both detect water",
  optical_only: "water seen in optical only — SAR does not confirm",
  sar_only: "water seen in SAR only — optical does not confirm",
  both_absent: "no water in either optical or SAR",
  disagree: "optical and SAR water maps disagree",
};

const WITHHELD_TEXT: Record<string, (c: Claim) => string> = {
  built_up_direction: () => "built-up direction not determined — no land-cover class tool",
  built_up_agreement: () => "built-up agreement withheld — no built-up tool",
  coreg_shift_px: (c) => `co-registration shift inconclusive${basisReason(c)}`,
  coreg_transform_offset_m: () => "co-registration offset withheld",
  water_agreement: (c) => `optical/SAR water agreement ${humanValue(c.value)}`,
  canonical_answer: () => "remote-sensing model answer unavailable",
  primary_transition: () => "main land-cover transition not identified",
  largest_change_class: () => "largest change class not identified",
  trend: () => "trend not determined",
  cdvqa_answer: () => "change-QA answer withheld",
};

// measured claims that are loud as "label: value" (model/semantic answers)
const LOUD_GENERIC = new Set([
  "primary_transition",
  "largest_change_class",
  "trend",
  "cdvqa_answer",
  "canonical_answer",
]);

export function composeVerified(b: RunBundle): VerifiedFacts {
  const claims = b.evidence_packet?.claims ?? [];
  const limits = (b.evidence_packet?.limitations ?? []) as string[];
  const to = b.tool_outputs ?? {};
  const area = rec(to.area_calc);
  const agree = rec(to.sar_agreement);
  const measured: Fact[] = [];
  const withheld: Fact[] = [];
  const used = new Set<string>();
  const firstOk = (p: string) =>
    claims.find((c) => c.predicate === p && !used.has(c.id) && !isWithheldClaim(c));
  const take = <T extends Claim | undefined>(c: T): T => {
    if (c) used.add(c.id);
    return c;
  };
  const ids = (...cs: (Claim | undefined)[]) =>
    cs.filter(Boolean).map((c) => c!.id).join(",");
  // area_calc's percent describes the mask whose pixel count it reports —
  // only attach it to a claim with the exact same count
  const pctFor = (px: number | null) => {
    const v = num(area?.percent_of_image);
    if (px == null || v == null || num(area?.changed_pixels) !== px) return null;
    return { text: fmtPct(v), exact: String(v) };
  };

  // change detection
  const cd = take(firstOk("change_detected"));
  if (cd) {
    const pxc = take(firstOk("changed_pixels"));
    const px = num(pxc?.value);
    if (cd.value === false) {
      measured.push({ key: "change", text: "no change detected", source: ids(cd, pxc) });
    } else {
      const p = pctFor(px);
      measured.push({
        key: "change",
        text: p
          ? `change detected across ${p.text}% of the image`
          : px != null
            ? `change detected in ${fmtInt(px)} pixels`
            : "change detected",
        exact: p && px != null ? `${p.exact}% · ${fmtInt(px)} px` : undefined,
        source: ids(cd, pxc),
      });
    }
  }

  // optical + SAR water agreement
  const wa = take(firstOk("water_agreement"));
  if (wa) {
    const of = num(agree?.optical_water_fraction);
    const sf = num(agree?.sar_water_fraction);
    const head =
      WATER_AGREEMENT_TEXT[String(wa.value)] ??
      `optical/SAR water agreement: ${humanValue(wa.value)}`;
    const both = of != null && sf != null;
    if (both)
      for (const c of claims) if (c.predicate === "water_pixels" && !isWithheldClaim(c)) take(c);
    measured.push({
      key: "water_agreement",
      text: both
        ? `${head} (optical ${fmtPct(of * 100)}% · SAR ${fmtPct(sf * 100)}% of the image)`
        : head,
      exact: both ? `optical fraction ${of} · SAR fraction ${sf}` : undefined,
      source: wa.id,
    });
    const iou = take(firstOk("agreement_iou"));
    const iv = num(iou?.value);
    if (iou && iv != null)
      measured.push({
        key: "agreement_iou",
        text: `the two water masks overlap at IoU ${fmtRatio(iv)}`,
        exact: String(iv),
        source: iou.id,
      });
  }

  // water (optical tool first)
  const wp = take(
    claims.find(
      (c) =>
        c.predicate === "water_pixels" &&
        !used.has(c.id) &&
        !isWithheldClaim(c) &&
        c.provenance?.tool === "water_highlight",
    ) ?? firstOk("water_pixels"),
  );
  const wv = num(wp?.value);
  if (wp && wv != null) {
    const p = pctFor(wv);
    measured.push({
      key: "water",
      text:
        wv === 0
          ? "no water detected"
          : p
            ? `water covers ${p.text}% of the image`
            : `water detected in ${fmtInt(wv)} pixels`,
      exact: p ? `${p.exact}% · ${fmtInt(wv)} px` : undefined,
      source: wp.id,
    });
  }

  // area — only from measured area claims; otherwise say it was withheld
  const km2 = take(firstOk("area_km2"));
  const m2 = take(firstOk("area_m2"));
  if (km2 || m2) {
    const k = num(km2?.value) ?? (m2 ? num(area?.area_km2) : null);
    const mv = num(m2?.value);
    const gsd = num(area?.gsd_m);
    const what = cd ? "changed area" : wp ? "water area" : "area";
    const val = k != null ? `${k} km²` : mv != null ? `${fmtInt(mv)} m²` : null;
    if (val)
      measured.push({
        key: "area",
        text: `${what}: ${val}${gsd != null ? ` (at ${gsd} m/px)` : ""}`,
        exact: mv != null ? `${mv} m²` : undefined,
        source: ids(km2, m2),
      });
  } else if (area || limits.some((l) => /square metres withheld/i.test(l))) {
    // upstream withhold ("water_highlight withheld (no_spectral_basis)") is
    // the real cause — surface it instead of guessing "no map scale"
    const reason = typeof area?.withheld_reason === "string" ? area.withheld_reason : "";
    const dep = reason.match(/^([a-z_]+) withheld/i);
    const why = dep ? `${humanTool(dep[1])}${reason.slice(dep[1].length)}` : reason;
    const noScale = area ? num(area.gsd_m) == null : true;
    withheld.push({
      key: "area",
      text: why
        ? `area in km² withheld — ${why}`
        : noScale
          ? "area in km² withheld — no map scale (GSD)"
          : "area in km² withheld",
      source: area ? "area_calc" : "limitations",
    });
  }

  const q = take(firstOk("dominant_quadrant"));
  if (q && typeof q.value === "string")
    measured.push({
      key: "quadrant",
      text: `mostly in the ${QUADRANT[q.value] ?? q.value} quadrant`,
      source: q.id,
    });

  const g = take(firstOk("iou_vs_gt"));
  const gv = num(g?.value);
  if (g && gv != null)
    measured.push({
      key: "iou_vs_gt",
      text: `matches the reference mask at IoU ${fmtRatio(gv)}`,
      exact: String(gv),
      source: g.id,
    });

  for (const c of claims) {
    if (used.has(c.id) || isWithheldClaim(c) || !LOUD_GENERIC.has(c.predicate)) continue;
    used.add(c.id);
    measured.push({
      key: `m:${c.id}`,
      text: `${humanPredicate(c.predicate)}: ${humanValue(c.value)}`,
      source: c.id,
    });
  }

  for (const c of claims) {
    if (used.has(c.id) || !isWithheldClaim(c)) continue;
    used.add(c.id);
    const fn =
      WITHHELD_TEXT[c.predicate] ??
      ((x: Claim) => `${humanPredicate(x.predicate)}: ${humanValue(x.value)}`);
    withheld.push({ key: `w:${c.id}`, text: fn(c), exact: c.confidence?.basis, source: c.id });
  }

  return { measured, withheld };
}

// --- display sanitizer ----------------------------------------------------

// Stored bundles and older pipeline copy carry internal-audit phrasing —
// scrub it at the display layer so the surface stays product-voiced even
// when the stored text predates the copy cleanup.
export function sanitizeCopy(s: string): string {
  let out = s;
  // 1. old unverified-interpretation flag line
  out = out.replace(/\n*UNVERIFIED INTERPRETATION — JSON card wins\.\n*/g, "\n\n");
  // 2. pipeline interpretation-note block (also stripped from shown text)
  out = out.replace(
    /\n*### Interpretation\n+Measured values are computed by the tools; the written summary may restate them approximately — cite the measurement card\.\n*/g,
    "\n\n",
  );
  // 3. "VLM did not compute" provenance phrasing
  out = out.replace(
    /deterministic; VLM did not compute this/g,
    "deterministic tool measurement",
  );
  out = out.replace(
    /\(VLM did not compute this\)/g,
    "(deterministic tool measurement)",
  );
  out = out.replace(
    /\(VLM did not compute these\)/g,
    "(deterministic tool measurement)",
  );
  out = out.replace(
    /; VLM did not compute this/g,
    "; deterministic tool measurement",
  );
  out = out.replace(
    /; VLM did not compute these/g,
    "; deterministic tool measurement",
  );
  out = out.replace(/VLM did not compute these/g, "deterministic tool measurement");
  out = out.replace(/VLM did not compute this/g, "deterministic tool measurement");
  // 4. honesty aside + heading
  out = out.replace(/ \(honesty, not a fake 0\.99\)/g, "");
  out = out.replace(/Confidence hierarchy/g, "Confidence");
  // 5. interpretation labels
  out = out.replace(
    /\*\*VLM interpretation \(not a measurement\)\*\*/g,
    "**Interpretation (written summary)**",
  );
  out = out.replace(
    /\(VLM interpretation — not a measurement\)/g,
    "(interpretation)",
  );
  out = out.replace(/ \(not a measurement\)/g, "");
  // 6–10 sentence rewrites
  out = out.replace(
    /The answer paragraph may mis-attach numbers\. Do not treat prose km² as independent evidence\./g,
    "The summary may restate measured values approximately; cite the tool measurements above.",
  );
  out = out.replace(/\*\*Do not conflate:\*\*/g, "**Reading these numbers:**");
  out = out.replace(
    /If the VLM paragraph invents km², \*\*the JSON card wins\.\*\*/g,
    "Area in km² is withheld; cite this card, not the written summary.",
  );
  out = out.replace(
    /If the VLM paragraph attaches NE pixels to whole-image percent, \*\*the JSON card wins\.\*\*/g,
    "The NE figure is quadrant-only; cite this card for whole-image values.",
  );
  out = out.replace(
    /These are the numbers a judge should cite\./g,
    "These are the values to cite.",
  );
  return out.trim();
}

// --- answer text split ---------------------------------------------------

// either flag form marks the written summary as needing a caution chip —
// detected for the report-check chip, but the marker itself never renders
export const INTERPRETATION_FLAG =
  /UNVERIFIED INTERPRETATION[^\n]*|### Interpretation\n+Measured values are computed by the tools[^\n]*/;

export interface AnswerParts {
  // deterministic "Findings (from tools)" block (report.findings)
  toolFindings: string;
  // everything after it, minus the flag line
  prose: string;
  // the pipeline's unverified-interpretation flag line, verbatim, if present
  flag: string | null;
  // true when a vision-language model ran (trace.vlm populated) — the prose
  // is model-written; false means it is deterministic tool/planner text
  proseByModel: boolean;
}

export function splitAnswer(b: RunBundle): AnswerParts {
  const vis = b.visible_answer ?? b.answer ?? "";
  const findings = (b.report?.findings ?? "").trim();
  let rest = findings && vis.startsWith(findings) ? vis.slice(findings.length) : (b.answer ?? vis);
  const m = rest.match(INTERPRETATION_FLAG);
  const flag = m ? m[0].trim() : null;
  if (m) rest = rest.replace(m[0], "");
  const vlm = rec(b.trace?.vlm);
  return {
    toolFindings: sanitizeCopy(findings),
    prose: sanitizeCopy(rest),
    flag,
    proseByModel: !!vlm && Object.keys(vlm).length > 0,
  };
}

export function reportCheck(b: RunBundle): { ok: boolean; issues: string[] } | null {
  const nc = rec(b.report?.narration_check) ?? rec(b.trace?.narration_check);
  if (!nc || typeof nc.ok !== "boolean") return null;
  return {
    ok: nc.ok,
    issues: Array.isArray(nc.issues) ? nc.issues.map(String) : [],
  };
}

// --- one-line image summary ---------------------------------------------

function formatOf(name: string, crs: string | null): string | null {
  const n = name.toLowerCase();
  if (/\.tiff?$/.test(n)) return crs ? "GeoTIFF" : "TIFF";
  if (n.endsWith(".png")) return "PNG";
  if (/\.jpe?g$/.test(n)) return "JPEG";
  if (n.endsWith(".npz")) return "NPZ";
  return null;
}

// e.g. "PNG · no map scale · area in km² withheld", "GeoTIFF · 10 m/px · EPSG:32645"
export function imageSummary(p: {
  names: string[];
  gsd_m: number | null;
  crs: string | null;
  calibrated?: boolean | null;
}): string {
  const fmts = [...new Set(p.names.map((n) => formatOf(n, p.crs)).filter(Boolean))];
  const bits: string[] = [];
  if (fmts.length) bits.push(fmts.join(" + "));
  if (p.gsd_m != null) {
    bits.push(`${p.gsd_m} m/px`);
    if (p.crs) bits.push(p.crs);
  } else {
    bits.push("no map scale", "area in km² withheld");
  }
  if (p.calibrated === true) bits.push("SAR calibrated");
  if (p.calibrated === false) bits.push("SAR not calibrated");
  return bits.join(" · ");
}
