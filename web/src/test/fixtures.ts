// Test fixtures shaped like real run bundles (api/schemas.py contract).
import type { RunBundle, UploadResponse } from "../types";

export const fakeUpload: UploadResponse = {
  upload_id: "up_test1",
  workdir: "C:/tmp/wd",
  warnings: [],
  detected: {
    gsd_m: 10.0,
    gsd_source: "geotransform",
    crs: "EPSG:32645",
    crs_note: null,
    bands: 4,
    dtype: "uint16",
    calibrated: null,
  },
};

export const fakeSarUpload: UploadResponse = {
  ...fakeUpload,
  upload_id: "up_test2",
  detected: { ...fakeUpload.detected, calibrated: true, bands: 2, dtype: "float64" },
};

export function makeBundle(over: Partial<RunBundle> = {}): RunBundle {
  return {
    run_id: "run_test1",
    answer: "Water is confirmed in this scene.",
    visible_answer: "### Findings (from tools)\n\n- water_pixels=34900\n\nWater is confirmed.",
    plan: { supported: true, tools: ["water_highlight", "sar_read", "sar_agreement"] },
    tool_outputs: {
      water_highlight: { water_pixels: 34900 },
      sar_read: { water_pixels: 35877 },
      sar_agreement: { water: "both_support" },
    },
    evidence_packet: {
      claims: [
        {
          id: "E1",
          predicate: "water_pixels",
          value: 34900,
          confidence: { level: "measured", basis: "RGB Otsu mask" },
          provenance: { tool: "water_highlight" },
        },
        {
          id: "E2",
          predicate: "built_up_agreement",
          value: "withheld_no_tool",
          confidence: { level: "withheld", basis: "no optical built-up tool" },
          provenance: { tool: "sar_agreement" },
        },
        {
          id: "E3",
          predicate: "canonical_answer",
          value: "yes",
          confidence: { level: "measured", basis: "greedy decode" },
          provenance: {
            tool: "canonical_vqa",
            model: "canonical/Qwen3VL-8B-RSVQA-Q4_K_M.gguf",
            seat: "127.0.0.1:8091",
            gguf_sha256: "a8797686abc",
          },
        },
      ],
      limitations: ["measured, not assumed; pair not realigned."],
    },
    trace: {
      query: "Is there water?",
      input_mode: "optical+sar",
      input_source: "upload",
      live: true,
      gsd: { gsd_m: 10.0 },
      first_token_s: 2.1,
      narration_check: { ok: true },
      vlm: { url: "http://127.0.0.1:8080" },
    },
    report: {
      findings: "- sar_agreement: water=both_support",
      narration_check: { ok: true },
      limitations: ["measured, not assumed; pair not realigned."],
    },
    artifacts: [
      { type: "overlay", path: "C:/x/overlay.png", name: "overlay_live.png", url: "/artifacts/run_test1/overlay_live.png" },
      { type: "geo_export", path: "C:/x/water_sar.tif", name: "water_sar_mask.tif", url: "/artifacts/run_test1/water_sar_mask.tif" },
    ],
    ...over,
  } as RunBundle;
}

export const refusedBundle = makeBundle({
  plan: { supported: false, refusal: "needs two co-registered images" },
  answer: "This query needs a bi-temporal pair.",
  visible_answer: "This query needs a bi-temporal pair.",
  tool_outputs: {},
  evidence_packet: null,
});
