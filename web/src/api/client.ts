// Typed API access. Every type here is derived from src/api/schema.d.ts, which
// `npm run gen:api` generates from recoup/api/schemas.py -- never hand-edit it.
import createClient from "openapi-fetch";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { components, paths } from "./schema";

export type Schemas = components["schemas"];
export type Overview = Schemas["Overview"];
export type GateStatus = Schemas["GateStatus"];
export type GateHistoryPoint = Schemas["GateHistoryPoint"];
export type ModelSummary = Schemas["ModelSummary"];
export type DecisionRow = Schemas["DecisionRow"];
export type InvoiceRow = Schemas["InvoiceRow"];
export type InvoiceDetail = Schemas["InvoiceDetail"];
export type Explanation = Schemas["Explanation"];
export type CurvePoint = Schemas["CurvePoint"];
export type SimStatus = Schemas["SimStatus"];
export type Score = Schemas["Score"];
export type Coverage = Schemas["Coverage"];

export const api = createClient<paths>({ baseUrl: "" });

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

/** Unwrap an openapi-fetch result: data, or a thrown ApiError carrying the server's detail. */
async function unwrap<T>(p: Promise<{ data?: T; error?: unknown; response: Response }>): Promise<T> {
  const { data, error, response } = await p;
  if (error !== undefined || data === undefined) {
    const detail = (error as { detail?: unknown } | undefined)?.detail;
    throw new ApiError(response.status, typeof detail === "string" ? detail : response.statusText);
  }
  return data;
}

// The batch jobs run hourly; a minute of staleness is invisible to an operator.
const LIVE = { refetchInterval: 60_000, staleTime: 30_000 } as const;

export const useOverview = () =>
  useQuery({ queryKey: ["overview"], queryFn: () => unwrap(api.GET("/api/overview")), ...LIVE });

export const useGateHistory = () =>
  useQuery({ queryKey: ["gate-history"], queryFn: () => unwrap(api.GET("/api/gate/history")), ...LIVE });

export const useModel = () =>
  useQuery({ queryKey: ["model"], queryFn: () => unwrap(api.GET("/api/model")), ...LIVE });

export const useModelVersions = () =>
  useQuery({ queryKey: ["model-versions"], queryFn: () => unwrap(api.GET("/api/model/versions")), ...LIVE });

export const useCoefficients = () =>
  useQuery({ queryKey: ["coefficients"], queryFn: () => unwrap(api.GET("/api/model/coefficients")) });

export type DecisionQuery = NonNullable<paths["/api/decisions"]["get"]["parameters"]["query"]>;
export const useDecisions = (q: DecisionQuery) =>
  useQuery({
    queryKey: ["decisions", q],
    queryFn: () => unwrap(api.GET("/api/decisions", { params: { query: q } })),
    placeholderData: keepPreviousData,
    ...LIVE,
  });

export const useDaily = (days = 60) =>
  useQuery({
    queryKey: ["daily", days],
    queryFn: () => unwrap(api.GET("/api/decisions/daily", { params: { query: { days } } })),
    ...LIVE,
  });

export const useCoverage = () =>
  useQuery({ queryKey: ["coverage"], queryFn: () => unwrap(api.GET("/api/decisions/coverage")), ...LIVE });

export type InvoiceQuery = NonNullable<paths["/api/invoices"]["get"]["parameters"]["query"]>;
export const useInvoices = (q: InvoiceQuery) =>
  useQuery({
    queryKey: ["invoices", q],
    queryFn: () => unwrap(api.GET("/api/invoices", { params: { query: q } })),
    placeholderData: keepPreviousData,
    ...LIVE,
  });

export const useInvoice = (id: string | null) =>
  useQuery({
    queryKey: ["invoice", id],
    queryFn: () => unwrap(api.GET("/api/invoices/{invoice_id}", { params: { path: { invoice_id: id! } } })),
    enabled: !!id,
  });

export const useSim = () =>
  useQuery({
    queryKey: ["sim"],
    queryFn: () => unwrap(api.GET("/api/sim")),
    // Poll fast only while a step is running.
    refetchInterval: (q) => (q.state.data?.busy ? 1_500 : 30_000),
  });

export const useScore = (enabled: boolean) =>
  useQuery({ queryKey: ["score"], queryFn: () => unwrap(api.GET("/api/sim/score")), enabled });

export const useStep = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Schemas["StepRequest"]) => unwrap(api.POST("/api/sim/step", { body })),
    onSuccess: (s) => qc.setQueryData(["sim"], s),
  });
};

/** Invalidate everything that reads job output -- after a simulation step lands. */
export const useRefreshAll = () => {
  const qc = useQueryClient();
  return () => qc.invalidateQueries({ predicate: (q) => q.queryKey[0] !== "sim" });
};
