import json
from pathlib import Path

from driftscope.eval.longmemeval.cli import main


def test_longmemeval_cli_runs_with_mock_backend(tmp_path: Path, capsys) -> None:
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "我住在哪？",
                    "haystack_sessions": ["我现在住在上海"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    exit_code = main(
        [
            str(dataset_path),
            str(output_dir),
            "--backend",
            "mock",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Completed 1/1 instances." in captured.out
    assert (output_dir / "predictions.jsonl").exists()
    assert (output_dir / "turns.jsonl").exists()
    assert (output_dir / "summary.json").exists()
