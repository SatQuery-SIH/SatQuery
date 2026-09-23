import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { Markdown } from "../md";

describe("Markdown — JSON blocks", () => {
  it("a line ending in { collects through the closing } into one <pre>", () => {
    const text = [
      "measurement card",
      "{",
      '  "changed_pixels": 1284,',
      '  "area_m2": 51360',
      "}",
      "",
      "trailing paragraph",
    ].join("\n");
    const { container } = render(<Markdown text={text} />);
    const pre = container.querySelector("pre.md-pre");
    expect(pre).toBeInTheDocument();
    expect(pre!.textContent).toBe(
      '{\n  "changed_pixels": 1284,\n  "area_m2": 51360\n}',
    );
    expect(container.querySelectorAll("pre.md-pre").length).toBe(1);
    // surrounding text still renders as paragraphs
    const ps = [...container.querySelectorAll("p")].map((p) => p.textContent);
    expect(ps).toContain("measurement card");
    expect(ps).toContain("trailing paragraph");
  });

  it("text before the brace becomes a paragraph; unclosed block falls back", () => {
    const closed = render(<Markdown text={"summary {\n  1: 2\n}"} />);
    expect(closed.container.querySelector("p")!.textContent).toBe("summary");
    expect(
      closed.container.querySelector("pre.md-pre")!.textContent,
    ).toBe("{\n  1: 2\n}");

    const open = render(
      <Markdown text={"note {\n  never closed\nmore text"} />,
    );
    // no closing } → no <pre>; lines render as normal paragraphs
    expect(open.container.querySelector("pre.md-pre")).toBeNull();
    const ps = [...open.container.querySelectorAll("p")].map(
      (p) => p.textContent,
    );
    expect(ps).toContain("note {");
    expect(ps).toContain("  never closed");
  });
});

describe("Markdown — existing behavior unchanged", () => {
  it("bullets, headers, bold, code", () => {
    const { container } = render(
      <Markdown
        text={
          "### Title\n\n- item **one**\n- item `two`\n\npara with **bold** and `code`"
        }
      />,
    );
    expect(container.querySelector("h4")!.textContent).toBe("Title");
    const lis = container.querySelectorAll("li");
    expect(lis.length).toBe(2);
    expect(lis[0].querySelector("strong")!.textContent).toBe("one");
    expect(lis[1].querySelector("code")!.textContent).toBe("two");
    const p = [...container.querySelectorAll("p")].at(-1)!;
    expect(p.querySelector("strong")!.textContent).toBe("bold");
    expect(p.querySelector("code")!.textContent).toBe("code");
  });
});
