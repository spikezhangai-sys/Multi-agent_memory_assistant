import json
from pathlib import Path

from driftscope.eval.longmemeval.adapter import LongMemEvalAdapter


def test_adapter_normalizes_instances(tmp_path: Path) -> None:
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "我住在哪？",
                    "haystack_sessions": [
                        {"text": "我现在住在上海", "timestamp": "2026-01-01T00:00:00Z"},
                        {"text": "我喜欢吃日料"},
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    instances = LongMemEvalAdapter().load(dataset_path)
    assert len(instances) == 1
    assert instances[0].question_id == "q1"
    assert len(instances[0].replay_turns) == 2
    assert instances[0].question_turn.query == "我住在哪？"


def test_adapter_flattens_nested_sessions_and_dates(tmp_path: Path) -> None:
    dataset_path = tmp_path / "longmemeval_nested.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q2",
                    "question": "我喜欢吃什么？",
                    "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/20 (Sat) 02:57"],
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "我最近喜欢吃日料"},
                            {"role": "assistant", "content": "好的，我记住了"},
                        ],
                        [
                            {"role": "user", "content": "我不能吃花生"},
                        ],
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    instances = LongMemEvalAdapter().load(dataset_path)
    assert len(instances[0].replay_turns) == 3
    assert instances[0].replay_turns[0].user_input == "我最近喜欢吃日料"
    assert instances[0].replay_turns[0].origin_role == "user"
    assert instances[0].replay_turns[1].user_input == "好的，我记住了"
    assert instances[0].replay_turns[1].origin_role == "assistant"
    assert instances[0].replay_turns[0].timestamp.isoformat().startswith("2023-05-20T02:21")


def test_adapter_can_filter_to_user_turns_only(tmp_path: Path) -> None:
    dataset_path = tmp_path / "longmemeval_user_only.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q3",
                    "question": "我喜欢什么？",
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "我喜欢日料"},
                            {"role": "assistant", "content": "好的，我记住了"},
                        ]
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    instances = LongMemEvalAdapter(user_turns_only=True).load(dataset_path)
    assert len(instances[0].replay_turns) == 1
    assert instances[0].replay_turns[0].origin_role == "user"
