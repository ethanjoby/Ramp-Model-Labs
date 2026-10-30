export type Metrics = {
  accuracy: number;
  macro_f1: number;
  per_label_f1: Record<string, number>;
  confusion: number[][];
  latency_p50_ms: number;
  latency_p95_ms: number;
  cost_per_1k: number;
  train_cost: number;
  train_seconds: number;
  size_kb: number | null;
  n_eval: number;
  notes: string;
  errors: { text: string; expected: string; predicted: string }[];
  search?: { C: number; dev_acc: number }[];
  instruction?: string;
  skip_reason?: string;
  error?: string;
};

export type CandidateRow = {
  id?: string;
  key: string;
  name: string;
  family: "classical" | "prompted" | "finetuned" | "reference";
  status: "queued" | "running" | "done" | "failed" | "skipped";
  metrics: Partial<Metrics>;
  is_winner?: boolean;
  blurb?: string;
  reason?: string;
};

export type Spec = {
  name: string;
  description: string;
  labels: string[];
  label_descriptions: Record<string, string>;
  input_description: string;
  template: string | null;
  target_accuracy: number;
  max_latency_ms: number;
  priority: "cost" | "accuracy" | "latency" | "balanced";
};

export type Savings = {
  frontier_cost_per_1k: number;
  frontier_measured: boolean;
  frontier_accuracy: number | null;
  winner_cost_per_1k: number;
  multiple: number;
  monthly_at_1m: { winner: number; frontier: number };
};

export type Dataset = {
  train: number;
  dev: number;
  test: number;
  sources: Record<string, number>;
  label_counts: Record<string, number>;
  eval_source: string;
  samples: { text: string; label: string; source: string }[];
  imbalance: number;
};

export type LabResult = {
  winner: string;
  winner_name: string;
  winner_metrics: Metrics;
  pareto: string[];
  rationale: string;
  savings: Savings;
  deployment: { slug: string; id: string };
  agent_spend: number;
};

export type Lab = {
  id: string;
  name: string;
  prompt: string;
  status: "queued" | "running" | "completed" | "failed";
  error: string | null;
  created_at: string;
  finished_at: string | null;
  spec: Spec | Record<string, never>;
  summary: { winner_name?: string; savings?: Savings; deployment?: { slug: string } };
  winner_metrics: Partial<Metrics>;
  result?: LabResult | Record<string, never>;
  dataset?: Dataset | Record<string, never>;
  candidates?: CandidateRow[];
  seed_count?: number;
};

export type LabEvent = {
  seq: number;
  agent: string;
  kind: "log" | "stage" | "candidate" | "metric" | "result" | "error";
  message: string;
  data: Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  ts: string;
};

export type Deployment = {
  id: string;
  slug: string;
  name: string;
  status: string;
  lab_id: string;
  created_at: string;
  requests: number;
  avg_latency_ms: number | null;
  model: string;
  family: string;
  metrics: Partial<Metrics>;
  labels: string[];
  savings: Savings | null;
  feedback_count: number;
};

export type Prediction = { label: string; confidence: number | null; top: { label: string; p: number }[] | null; route?: "rule" | "model" | "llm" | "review" };

export type ChatAction =
  | { type: "lab_started"; lab_id: string; prompt: string }
  | { type: "lab_ref"; lab_id: string }
  | { type: "prediction"; slug: string; name: string; rows: ({ input: string } & Prediction)[]; latency_ms: number }
  | { type: "models"; models: { slug: string; name: string; model: string; accuracy: number; cost_per_1k: number }[] };

export type ChatMsg = { role: "user" | "assistant"; content: string; actions?: ChatAction[]; ts?: string; pending?: boolean };

export type User = { id: string; email: string; name: string; avatar_url?: string | null; provider?: string };
