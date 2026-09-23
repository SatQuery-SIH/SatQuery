import { describe, expect, it } from "vitest";
import { parseSseText, SseParser, type SseFrame } from "../sse";
import refusal from "./fixtures/sse_refusal.txt?raw";
import supported from "./fixtures/sse_supported.txt?raw";
import seatDown from "./fixtures/sse_seat_down.txt?raw";

type Row = [string, string | null, string | null, string | null];

function seq(frames: SseFrame[]): Row[] {
  return frames.map((f) => {
    let stage: string | null = null;
    let status: string | null = null;
    let tool: string | null = null;
    try {
      const d = JSON.parse(f.data) as Record<string, unknown>;
      stage = (d.stage as string) ?? null;
      status = (d.status as string) ?? null;
      tool = (d.tool as string) ?? null;
    } catch {
      /* non-json payloads (done bundle / error body) keep nulls */
    }
    return [f.event, stage, status, tool];
  });
}

function chunked(text: string, n: number): SseFrame[] {
  const out: SseFrame[] = [];
  const p = new SseParser((f) => out.push(f));
  for (let i = 0; i < text.length; i += n) p.push(text.slice(i, i + n));
  p.end();
  return out;
}

describe("SseParser — recorded fixtures", () => {
  it("refusal: exact recorded sequence ending in done", () => {
    expect(seq(parseSseText(refusal))).toEqual([
      ["stage", "plan", "start", null],
      ["stage", "plan", "done", null],
      ["stage", "bind", "start", null],
      ["stage", "bind", "done", null],
      ["stage", "plan", "withheld", null],
      ["stage", "packet", "start", null],
      ["stage", "packet", "done", null],
      ["stage", "done", "done", null],
      ["done", null, null, null],
    ]);
  });

  it("supported: exact recorded sequence ending in done", () => {
    expect(seq(parseSseText(supported))).toEqual([
      ["stage", "plan", "start", null],
      ["stage", "plan", "done", null],
      ["stage", "bind", "start", null],
      ["stage", "bind", "done", null],
      ["stage", "tool", "start", "water_highlight"],
      ["stage", "tool", "done", "water_highlight"],
      ["stage", "tool", "start", "area_calc"],
      ["stage", "tool", "done", "area_calc"],
      ["stage", "narration", "withheld", null],
      ["stage", "packet", "start", null],
      ["stage", "packet", "done", null],
      ["stage", "done", "done", null],
      ["done", null, null, null],
    ]);
  });

  it("seat_down: exact recorded sequence ending in error", () => {
    expect(seq(parseSseText(seatDown))).toEqual([
      ["stage", "plan", "start", null],
      ["stage", "plan", "done", null],
      ["stage", "bind", "start", null],
      ["stage", "bind", "done", null],
      ["stage", "tool", "start", "water_highlight"],
      ["stage", "tool", "done", "water_highlight"],
      ["stage", "tool", "start", "area_calc"],
      ["stage", "tool", "done", "area_calc"],
      ["stage", "narration", "start", null],
      ["stage", "narration", "fail", null],
      ["error", null, null, null],
    ]);
  });
});

describe("SseParser — framing rules", () => {
  it("chunk sizes 1, 3, 17, 256 give identical frames", () => {
    const want = parseSseText(supported);
    for (const n of [1, 3, 17, 256]) {
      expect(chunked(supported, n)).toEqual(want);
    }
  });

  it("CRLF variant — including \\r|\\n split across chunks — is identical", () => {
    const want = parseSseText(supported);
    const crlf = supported.replace(/\n/g, "\r\n");
    expect(parseSseText(crlf)).toEqual(want);
    for (const n of [1, 3, 17]) expect(chunked(crlf, n)).toEqual(want);
    // bare-CR line endings too
    const cr = supported.replace(/\n/g, "\r");
    expect(parseSseText(cr)).toEqual(want);
  });

  it("comments and unknown fields ignored; multi-line data joins with \\n", () => {
    const frames = parseSseText(
      ": keepalive\n" +
        "event: x\n" +
        "data: a\n" +
        "data:b\n" +
        "foo: ignored\n" +
        "data:  c\n" +
        "\n" +
        "event: y\n" +
        "\n" +
        "data: hi\n" +
        "\n",
    );
    expect(frames).toEqual([
      { event: "x", data: "a\nb\n c" },
      { event: "message", data: "hi" },
    ]);
  });

  it("an unterminated event at EOF is discarded", () => {
    expect(parseSseText("event: x\ndata: never dispatched")).toEqual([]);
  });
});
