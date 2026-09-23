import { describe, expect, it, vi, afterEach } from "vitest";
import {
  ApiHttpError,
  queryStream,
  StreamPipelineError,
  StreamTransportError,
} from "../api";
import { parseSseText } from "../sse";
import type { StageEvent } from "../types";
import supported from "./fixtures/sse_supported.txt?raw";
import seatDown from "./fixtures/sse_seat_down.txt?raw";

const REQ = { query: "highlight the water", input_mode: "single" as const };

function sseRes(text: string): Response {
  const bytes = new TextEncoder().encode(text);
  return new Response(
    new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(bytes);
        c.close();
      },
    }),
    { status: 200, headers: { "content-type": "text/event-stream" } },
  );
}

function stubFetch(fn: (url: string, init?: RequestInit) => Promise<Response> | Response) {
  vi.stubGlobal("fetch", vi.fn(fn));
}

afterEach(() => vi.unstubAllGlobals());

describe("api.queryStream", () => {
  it("supported fixture resolves the recorded bundle; 12 ordered onStage calls", async () => {
    stubFetch(() => sseRes(supported));
    const stages: [string | undefined, string | undefined, string | undefined][] = [];
    const b = await queryStream(REQ, (e: StageEvent) => {
      stages.push([e.stage, e.status, e.tool]);
    });
    expect(b.run_id).toBe("cc3f49c89a234e31ad9f997a19eb9692");
    expect(stages).toEqual([
      ["plan", "start", undefined],
      ["plan", "done", undefined],
      ["bind", "start", undefined],
      ["bind", "done", undefined],
      ["tool", "start", "water_highlight"],
      ["tool", "done", "water_highlight"],
      ["tool", "start", "area_calc"],
      ["tool", "done", "area_calc"],
      ["narration", "withheld", undefined],
      ["packet", "start", undefined],
      ["packet", "done", undefined],
      ["done", "done", undefined],
    ]);
  });

  it("seat_down rejects StreamPipelineError after 10 onStage calls", async () => {
    stubFetch(() => sseRes(seatDown));
    const stages: StageEvent[] = [];
    const err = await queryStream(REQ, (e) => stages.push(e)).then(
      () => null,
      (e) => e,
    );
    expect(stages).toHaveLength(10);
    expect(err).toBeInstanceOf(StreamPipelineError);
    expect((err as StreamPipelineError).slug).toBe("pipeline_error");
    expect((err as StreamPipelineError).detail).toContain("WinError 10061");
  });

  it("404 {error:unknown_upload} rejects ApiHttpError with the slug", async () => {
    stubFetch(
      () =>
        new Response(
          JSON.stringify({ error: "unknown_upload", detail: "unknown upload_id" }),
          { status: 404 },
        ),
    );
    const err = await queryStream(REQ, () => {}).then(
      () => null,
      (e) => e,
    );
    expect(err).toBeInstanceOf(ApiHttpError);
    expect((err as ApiHttpError).status).toBe(404);
    expect((err as ApiHttpError).slug).toBe("unknown_upload");
  });

  it("fetch rejecting TypeError → StreamTransportError(eventsSeen 0)", async () => {
    stubFetch(() => {
      throw new TypeError("failed to fetch");
    });
    const err = await queryStream(REQ, () => {}).then(
      () => null,
      (e) => e,
    );
    expect(err).toBeInstanceOf(StreamTransportError);
    expect((err as StreamTransportError).eventsSeen).toBe(0);
  });

  it("empty body → StreamTransportError(0)", async () => {
    stubFetch(() => new Response(null, { status: 200 }));
    const err = await queryStream(REQ, () => {}).then(
      () => null,
      (e) => e,
    );
    expect(err).toBeInstanceOf(StreamTransportError);
    expect((err as StreamTransportError).eventsSeen).toBe(0);
  });

  it("truncated mid-stream → StreamTransportError(eventsSeen>0)", async () => {
    const prefix = supported.slice(0, 4000);
    stubFetch(() => sseRes(prefix));
    const stages: StageEvent[] = [];
    const err = await queryStream(REQ, (e) => stages.push(e)).then(
      () => null,
      (e) => e,
    );
    expect(err).toBeInstanceOf(StreamTransportError);
    const seen = (err as StreamTransportError).eventsSeen;
    expect(seen).toBeGreaterThan(0);
    expect(seen).toBe(parseSseText(prefix).length);
    expect(stages.length).toBeLessThanOrEqual(seen);
  });

  it("a multibyte char split across Uint8Array chunks decodes correctly", async () => {
    const text = 'event: done\ndata: {"run_id":"r—1","answer":"x—y"}\n\n';
    const bytes = new TextEncoder().encode(text);
    const idx = bytes.indexOf(0xe2); // first byte of "—" (0xE2 0x80 0x94)
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(bytes.slice(0, idx + 1));
        c.enqueue(bytes.slice(idx + 1, idx + 2));
        c.enqueue(bytes.slice(idx + 2));
        c.close();
      },
    });
    stubFetch(() => new Response(stream, { status: 200 }));
    const b = await queryStream(REQ, () => {});
    expect(b.run_id).toBe("r—1");
    expect(b.answer).toBe("x—y");
  });
});
