from __future__ import annotations

import argparse
from pathlib import Path

from driftscope.eval.longmemeval.mem0_baseline import Mem0BaselineConfig, Mem0BaselineRunner
from driftscope.eval.longmemeval.runner import LongMemEvalRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DriftScope on a LongMemEval-style dataset.")
    parser.add_argument("dataset_path", help="Path to the LongMemEval JSON dataset.")
    parser.add_argument("output_dir", help="Directory where predictions/logs/summary will be written.")
    parser.add_argument(
        "--baseline",
        choices=["driftscope", "mem0"],
        default="driftscope",
        help="System to evaluate. 'driftscope' runs the native pipeline; 'mem0' runs Mem0 as an external memory baseline.",
    )
    parser.add_argument(
        "--backend",
        choices=["env", "mock", "openrouter", "deepseek"],
        default="env",
        help="LLM backend override for DriftScope agents and Mem0 baseline answer generation. 'env' uses DRIFTSCOPE_CONFLICT_LLM from .env.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override, e.g. deepseekv4flash or gpt4omini.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of instances to run for smoke testing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent instance workers (thread pool). Default 1 = sequential.",
    )
    parser.add_argument(
        "--no-persist-memory",
        action="store_true",
        help="Disable per-question SQLite persistence. Default: each question's MemoryBase is saved to <output_dir>/db/<question_id>.db for post-hoc inspection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Max replay turns packed into one DriftScope UpdateAgent or Mem0 add batch. Default: 16.",
    )
    parser.add_argument(
        "--driftscope-config",
        default=None,
        help="Optional path to a DriftScope config YAML. Replaces the bundled default.yaml. Ignored when --baseline mem0.",
    )
    parser.add_argument(
        "--mem0-mode",
        choices=["oss", "cloud"],
        default="oss",
        help="Mem0 baseline mode. 'oss' uses mem0.Memory; 'cloud' uses mem0.MemoryClient.",
    )
    parser.add_argument(
        "--mem0-config",
        default=None,
        help="Optional JSON/YAML config file passed to mem0.Memory.from_config in --mem0-mode oss.",
    )
    parser.add_argument(
        "--mem0-api-key",
        default=None,
        help="Optional Mem0 API key for --mem0-mode cloud. Defaults to MEM0_API_KEY.",
    )
    parser.add_argument(
        "--mem0-top-k",
        type=int,
        default=20,
        help="Number of Mem0 search results used for answering. Default: 20, matching retrieval.top_k.",
    )
    parser.add_argument(
        "--mem0-threshold",
        type=float,
        default=0.1,
        help="Mem0 search score threshold. Default: 0.1.",
    )
    parser.add_argument(
        "--mem0-rerank",
        action="store_true",
        help="Enable Mem0 reranking during search when supported.",
    )
    parser.add_argument(
        "--mem0-no-infer",
        action="store_true",
        help="Store raw Mem0 memories with infer=False instead of using Mem0 extraction.",
    )
    parser.add_argument(
        "--mem0-answer-mode",
        choices=["llm", "concat"],
        default="llm",
        help="How Mem0 baseline turns search results into the LongMemEval hypothesis. Default: llm.",
    )
    parser.add_argument(
        "--mem0-user-prefix",
        default="driftscope_lme_mem0",
        help="Prefix for per-question Mem0 user_id isolation.",
    )
    parser.add_argument(
        "--mem0-run-id",
        default="longmemeval",
        help="Mem0 run_id used with each per-question user_id. Set to empty string to omit.",
    )
    parser.add_argument(
        "--mem0-no-cleanup",
        action="store_true",
        help="Do not delete existing Mem0 memories for each per-question scope before replay.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    llm_backend = None if args.backend == "env" else args.backend
    if args.baseline == "mem0":
        runner = Mem0BaselineRunner(
            config=Mem0BaselineConfig(
                mode=args.mem0_mode,
                config_path=Path(args.mem0_config) if args.mem0_config else None,
                api_key=args.mem0_api_key,
                top_k=args.mem0_top_k,
                threshold=args.mem0_threshold,
                rerank=args.mem0_rerank,
                infer=not args.mem0_no_infer,
                answer_mode=args.mem0_answer_mode,
                batch_size=args.batch_size,
                user_prefix=args.mem0_user_prefix,
                run_id=args.mem0_run_id or None,
                clean_start=not args.mem0_no_cleanup,
                llm_backend=llm_backend,
                llm_model=args.model,
            ),
            workers=args.workers,
        )
    else:
        runner = LongMemEvalRunner(
            llm_backend=llm_backend,
            llm_model=args.model,
            workers=args.workers,
            persist_memory=not args.no_persist_memory,
            replay_batch_size=args.batch_size,
            config_path=args.driftscope_config,
        )
    summary = runner.run(
        Path(args.dataset_path),
        Path(args.output_dir),
        limit=args.limit,
    )
    print(f"Completed {summary.num_predictions}/{summary.num_instances} instances.")
    print(f"Predictions: {summary.predictions_path}")
    print(f"Turns: {summary.turns_path}")
    print(f"Summary: {summary.summary_path}")
    return 0
