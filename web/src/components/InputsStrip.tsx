// Bound-input summary: what imagery the next run will actually use —
// server-rendered upload previews or the prepared scene's static real pixels.
export function InputsStrip({
  title,
  source,
  items,
  gsd,
}: {
  title: string;
  source: "upload" | "prepared";
  items: { role: string; label: string; src: string | null }[];
  gsd?: { gsd_m: number | null; gsd_source: string } | null;
}) {
  return (
    <div className="inputs-strip" data-testid="inputs-strip" data-source={source}>
      <div className="inputs-strip-title">{title}</div>
      <div className="inputs-strip-row">
        {items.map((it) => (
          <figure className="inputs-thumb" key={it.role}>
            {it.src ? (
              <img src={it.src} alt={it.label} />
            ) : (
              <div className="inputs-thumb-empty">no preview</div>
            )}
            <figcaption>{it.label}</figcaption>
          </figure>
        ))}
        {gsd &&
          (gsd.gsd_m != null ? (
            <span className="gsd-chip" data-testid="gsd-chip">
              GSD {gsd.gsd_m} m/px · {gsd.gsd_source}
            </span>
          ) : (
            <span className="gsd-chip muted" data-testid="gsd-chip">
              GSD not detected
            </span>
          ))}
      </div>
    </div>
  );
}
