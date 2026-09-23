// Small formatting helpers shared across components.

export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return "";
  if (n < 1024) return `${Math.round(n)} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function formatSecs(n: number): string {
  if (!Number.isFinite(n)) return "";
  if (n >= 10) return `${n.toFixed(1)}s`;
  if (n >= 1) return `${n.toFixed(2)}s`;
  if (n >= 0.001) return `${n.toFixed(3)}s`;
  return "<1 ms";
}
