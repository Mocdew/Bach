const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const usd0 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export const money = (v?: number | null) => (v == null ? "—" : usd.format(v));
export const money0 = (v?: number | null) => (v == null ? "—" : usd0.format(v));
export const signedMoney = (v?: number | null) =>
  v == null ? "—" : `${v >= 0 ? "+" : "−"}${usd.format(Math.abs(v))}`;
export const pct = (v?: number | null, digits = 0) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);
export const signedPct = (v?: number | null, digits = 1) =>
  v == null ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(digits)}%`;
export const num = (v?: number | null, digits = 0) =>
  v == null ? "—" : v.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });
export const prob = (v?: number | null) => (v == null ? "—" : v.toFixed(2));

/** Hours since failure, human-sized: 45m, 5h, 3.5d. */
export const hours = (h?: number | null) => {
  if (h == null) return "—";
  if (h < 1) return `${Math.round(h * 60)}m`;
  if (h < 48) return `${+h.toFixed(1)}h`;
  return `${+(h / 24).toFixed(1)}d`;
};

export const when = (iso?: string | null) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC" });
};
export const day = (iso?: string | null) =>
  iso ? new Date(iso).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric", timeZone: "UTC" }) : "—";

export const label = (s?: string | null) => (s ? s.replace(/_/g, " ") : "—");
