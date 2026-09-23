// Server-Sent Events frame parser per the WHATWG spec — pure, zero deps.
// Handles \n / \r\n / \r line breaks (a \r at a chunk boundary is held so a
// split \r\n is not two breaks), ":" comments, one leading space stripped
// from field values, multi-line `data` joined with "\n", and dispatch on a
// blank line only when at least one data line was collected.

export interface SseFrame {
  event: string;
  data: string;
}

export class SseParser {
  private buf = "";
  private eventName = "";
  private dataLines: string[] = [];
  private onFrame: (f: SseFrame) => void;

  constructor(onFrame: (f: SseFrame) => void) {
    this.onFrame = onFrame;
  }

  push(text: string): void {
    this.buf += text;
    let i = 0;
    let start = 0;
    while (i < this.buf.length) {
      const ch = this.buf[i];
      if (ch === "\n") {
        this.line(this.buf.slice(start, i));
        i++;
        start = i;
      } else if (ch === "\r") {
        if (i === this.buf.length - 1) break; // hold trailing \r — may be \r\n
        this.line(this.buf.slice(start, i));
        i++;
        if (this.buf[i] === "\n") i++;
        start = i;
      } else {
        i++;
      }
    }
    this.buf = this.buf.slice(start);
  }

  end(): void {
    // A \r held at the chunk boundary is a real line break at EOF — flush it
    // (a trailing blank line still dispatches). Anything else pending is an
    // unterminated event and is discarded, per spec.
    if (this.buf.endsWith("\r")) this.line(this.buf.slice(0, -1));
    this.buf = "";
    this.eventName = "";
    this.dataLines = [];
  }

  private line(raw: string): void {
    if (raw === "") {
      if (this.dataLines.length) {
        this.onFrame({
          event: this.eventName || "message",
          data: this.dataLines.join("\n"),
        });
      }
      this.eventName = "";
      this.dataLines = [];
      return;
    }
    if (raw.startsWith(":")) return; // comment line
    const ci = raw.indexOf(":");
    const field = ci === -1 ? raw : raw.slice(0, ci);
    let value = ci === -1 ? "" : raw.slice(ci + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") this.eventName = value;
    else if (field === "data") this.dataLines.push(value);
    // other fields (id, retry, unknown) are ignored
  }
}

export function parseSseText(text: string): SseFrame[] {
  const out: SseFrame[] = [];
  const p = new SseParser((f) => out.push(f));
  p.push(text);
  p.end();
  return out;
}
