// Verified-line composition against REAL stored run bundles (fixtures/
// bundle_*.json, captured from the API; only the repo path redacted).
import { describe, expect, it } from "vitest";
import type { RunBundle } from "../types";
import {
  composeVerified,
  humanValue,
  imageSummary,
  reportCheck,
  sanitizeCopy,
  splitAnswer,
} from "../verified";
import bitemporal from "./fixtures/bundle_bitemporal_flagged.json";
import single from "./fixtures/bundle_single_area.json";
import optsar from "./fixtures/bundle_optsar.json";
import nogsd from "./fixtures/bundle_single_nogsd.json";
import maskOnly from "./fixtures/bundle_mask_only.json";
import refused from "./fixtures/bundle_refused.json";

const B = (x: unknown) => x as RunBundle;
const texts = (fs: { text: string }[]) => fs.map((f) => f.text);

describe("composeVerified — real bundles", () => {
  it("bi-temporal: change % loud, area + built-up direction withheld loud", () => {
    const v = composeVerified(B(bitemporal));
    expect(texts(v.measured)).toEqual([
      "change detected across 25.8% of the image",
      "mostly in the north-east quadrant",
    ]);
    // rounded display never hides the exact bundle value
    expect(v.measured[0].exact).toBe("25.80738067626953% · 270,610 px");
    expect(v.measured[0].source).toBe("E1,E2");
    expect(texts(v.withheld)).toEqual([
      "area in km² withheld — no map scale (GSD)",
      "built-up direction not determined — no land-cover class tool",
    ]);
    // the narrator's contradicting prose never reaches the loud lines
    const all = [...v.measured, ...v.withheld].map((f) => f.text).join(" ");
    expect(all).not.toMatch(/residential|developed|30\.2/);
  });

  it("single GeoTIFF: water % and km² at the measured GSD, nothing withheld", () => {
    const v = composeVerified(B(single));
    expect(texts(v.measured)).toEqual([
      "water covers 53.3% of the image",
      "water area: 3.49 km² (at 10 m/px)",
    ]);
    expect(v.measured[1].exact).toBe("3490000 m²");
    expect(v.withheld).toEqual([]);
  });

  it("optical+SAR: agreement + IoU loud; built-up + coreg shift withheld", () => {
    const v = composeVerified(B(optsar));
    expect(texts(v.measured)).toEqual([
      "optical and SAR both detect water (optical 53.3% · SAR 54.7% of the image)",
      "the two water masks overlap at IoU 0.892",
    ]);
    expect(v.measured[1].exact).toBe("0.891927");
    expect(texts(v.withheld)).toEqual([
      "built-up agreement withheld — no built-up tool",
      "co-registration shift inconclusive (weak peak)",
    ]);
  });

  it("PNG without GSD: water % measured, area withheld", () => {
    const v = composeVerified(B(nogsd));
    expect(texts(v.measured)).toEqual(["water covers 5.8% of the image"]);
    expect(texts(v.withheld)).toEqual(["area in km² withheld — no map scale (GSD)"]);
  });

  it("mask-only run (no area tool): pixel count, reference IoU", () => {
    const v = composeVerified(B(maskOnly));
    expect(texts(v.measured)).toEqual([
      "change detected in 270,611 pixels",
      "matches the reference mask at IoU 0.852",
    ]);
    expect(texts(v.withheld)).toEqual([
      "built-up direction not determined — no land-cover class tool",
    ]);
  });

  it("refused run: no facts at all", () => {
    expect(composeVerified(B(refused))).toEqual({ measured: [], withheld: [] });
  });

  it("withheld-only packet: nothing measured, every withheld claim listed", () => {
    const b = B({
      ...optsar,
      tool_outputs: {},
      evidence_packet: {
        claims: [
          { id: "E1", predicate: "water_agreement", value: "withheld_uncalibrated", confidence: { level: "withheld", basis: "verdict withheld: withheld_uncalibrated" } },
          { id: "E2", predicate: "built_up_agreement", value: "withheld_no_tool", confidence: { level: "withheld" } },
          { id: "E3", predicate: "canonical_answer", value: null, confidence: { level: "withheld", basis: "canonical_vqa_unavailable" } },
        ],
        limitations: [],
      },
    });
    const v = composeVerified(b);
    expect(v.measured).toEqual([]);
    expect(texts(v.withheld)).toEqual([
      "optical/SAR water agreement withheld — SAR not calibrated",
      "built-up agreement withheld — no built-up tool",
      "remote-sensing model answer unavailable",
    ]);
  });
});

describe("splitAnswer", () => {
  it("separates tool findings, the verbatim flag, and model prose", () => {
    const p = splitAnswer(B(bitemporal));
    expect(p.toolFindings.startsWith("### Findings (from tools)")).toBe(true);
    expect(p.flag).toBe("UNVERIFIED INTERPRETATION — JSON card wins.");
    expect(p.prose.startsWith("Between the two images")).toBe(true);
    expect(p.prose).not.toContain("UNVERIFIED");
    expect(p.proseByModel).toBe(true);
  });

  it("no flag when the report check passed; prose still model-written", () => {
    const p = splitAnswer(B(single));
    expect(p.flag).toBeNull();
    expect(p.prose.startsWith("Yes, there is a large water body")).toBe(true);
    expect(p.proseByModel).toBe(true);
  });

  it("deterministic tool text is not labeled model prose", () => {
    expect(splitAnswer(B(maskOnly)).proseByModel).toBe(false);
    expect(splitAnswer(B(nogsd)).prose).toBe("");
  });
});

describe("sanitizeCopy — internal-audit phrasing scrubbed at the display layer", () => {
  it("strips the old UNVERIFIED flag line and tidies the gap", () => {
    expect(
      sanitizeCopy("### Findings\n\nUNVERIFIED INTERPRETATION — JSON card wins.\n\nThe answer."),
    ).toBe("### Findings\n\nThe answer.");
  });

  it("strips the new pipeline interpretation-note block", () => {
    const s =
      "findings\n\n### Interpretation\n\nMeasured values are computed by the tools; the written summary may restate them approximately — cite the measurement card.\n\nrest";
    expect(sanitizeCopy(s)).toBe("findings\n\nrest");
  });

  it("rewrites 'VLM did not compute' provenance phrasing", () => {
    expect(sanitizeCopy("(VLM did not compute this)")).toBe(
      "(deterministic tool measurement)",
    );
    expect(sanitizeCopy("(VLM did not compute these)")).toBe(
      "(deterministic tool measurement)",
    );
    expect(
      sanitizeCopy("tools.sar_read (preview DN; VLM did not compute this)."),
    ).toBe("tools.sar_read (preview DN; deterministic tool measurement).");
    // no duplicated "deterministic" when the source already had it
    expect(
      sanitizeCopy("tools.coreg_check (deterministic; VLM did not compute this)."),
    ).toBe("tools.coreg_check (deterministic tool measurement).");
  });

  it("drops the honesty aside and shortens the heading", () => {
    expect(sanitizeCopy("### Confidence hierarchy (honesty, not a fake 0.99)")).toBe(
      "### Confidence",
    );
  });

  it("relabels interpretation headings and drops '(not a measurement)'", () => {
    expect(
      sanitizeCopy("**VLM interpretation (not a measurement)** — role `x`."),
    ).toBe("**Interpretation (written summary)** — role `x`.");
    expect(sanitizeCopy("## Answer (VLM interpretation — not a measurement)")).toBe(
      "## Answer (interpretation)",
    );
    expect(sanitizeCopy("card (not a measurement) ok")).toBe("card ok");
  });

  it("rewrites the audit-voice sentences", () => {
    expect(
      sanitizeCopy(
        "The answer paragraph may mis-attach numbers. Do not treat prose km² as independent evidence.",
      ),
    ).toBe(
      "The summary may restate measured values approximately; cite the tool measurements above.",
    );
    expect(sanitizeCopy("**Do not conflate:**")).toBe("**Reading these numbers:**");
    expect(
      sanitizeCopy("If the VLM paragraph invents km², **the JSON card wins.**"),
    ).toBe("Area in km² is withheld; cite this card, not the written summary.");
    expect(
      sanitizeCopy(
        "If the VLM paragraph attaches NE pixels to whole-image percent, **the JSON card wins.**",
      ),
    ).toBe("The NE figure is quadrant-only; cite this card for whole-image values.");
    expect(sanitizeCopy("These are the numbers a judge should cite.")).toBe(
      "These are the values to cite.",
    );
  });

  it("passes the new demo copy through unchanged and is idempotent", () => {
    const clean =
      "### Measurement card (tool `area_calc` — deterministic tool measurement)\n\n" +
      "**Interpretation (written summary)** — model role `narrator`; tools selected: [vqa]. " +
      "The summary may restate measured values approximately; cite the tool measurements above.\n\n" +
      "**Reading these numbers:**\n- Area in km² is withheld; cite this card, not the written summary.\n" +
      "- The NE figure is quadrant-only; cite this card for whole-image values.\n" +
      "These are the values to cite.";
    expect(sanitizeCopy(clean)).toBe(clean);
    const messy =
      "x (VLM did not compute this) UNVERIFIED INTERPRETATION — JSON card wins. (not a measurement)";
    expect(sanitizeCopy(sanitizeCopy(messy))).toBe(sanitizeCopy(messy));
  });
});

describe("reportCheck / humanValue / imageSummary", () => {
  it("reads the narration audit", () => {
    expect(reportCheck(B(bitemporal))).toEqual({ ok: false, issues: ["invented number 30.2"] });
    expect(reportCheck(B(single))).toEqual({ ok: true, issues: [] });
    expect(reportCheck(B(refused))).toBeNull();
  });

  it("humanizes snake_case values", () => {
    expect(humanValue("not_determined")).toBe("not determined");
    expect(humanValue("low_vegetation->buildings")).toBe("low vegetation → buildings");
    expect(humanValue(null)).toBe("not identified");
    expect(humanValue(true)).toBe("yes");
    expect(humanValue("RGB Otsu on (B-R), threshold=4.0")).toBe("RGB Otsu on (B-R), threshold=4.0");
  });

  it("one-line image summary", () => {
    expect(imageSummary({ names: ["a.png", "b.png"], gsd_m: null, crs: null })).toBe(
      "PNG · no map scale · area in km² withheld",
    );
    expect(imageSummary({ names: ["s.tiff"], gsd_m: 10, crs: "EPSG:32645" })).toBe(
      "GeoTIFF · 10 m/px · EPSG:32645",
    );
    expect(
      imageSummary({ names: ["o.tiff", "s.tiff"], gsd_m: 10, crs: "EPSG:32645", calibrated: true }),
    ).toBe("GeoTIFF · 10 m/px · EPSG:32645 · SAR calibrated");
  });
});
