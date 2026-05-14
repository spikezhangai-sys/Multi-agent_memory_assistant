from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import re

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from driftscope.core.schema import TopicID

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[一-鿿]", re.IGNORECASE)
_MIN_MATCH_SCORE = 3.0
_MIN_MATCH_MARGIN = 1.0
_LEAF_SUFFIX_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,62}$")


class TopicCategory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: TopicID
    description: str
    default_type: str

    @field_validator("id", "description", "default_type")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("topic category fields must be non-empty")
        return value


class TopicLeaf(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: TopicID
    description: str
    examples: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    @field_validator("path", "description")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("topic leaf fields must be non-empty")
        return value


@dataclass(frozen=True)
class TopicMatch:
    path: TopicID
    score: float


class TopicTree:
    def __init__(
        self,
        *,
        categories: list[TopicCategory],
        seed_leaves: list[TopicLeaf],
    ) -> None:
        if not categories:
            raise ValueError("topic tree requires at least one category")
        if len({c.id for c in categories}) != len(categories):
            dupes = sorted({c.id for c in categories if sum(1 for x in categories if x.id == c.id) > 1})
            raise ValueError(f"duplicate category ids: {dupes}")
        self._categories: dict[str, TopicCategory] = {c.id: c for c in sorted(categories, key=lambda x: x.id)}

        for leaf in seed_leaves:
            category = category_of_path(leaf.path)
            if category not in self._categories:
                raise ValueError(f"seed leaf '{leaf.path}' has no registered category '{category}'")

        if len({leaf.path for leaf in seed_leaves}) != len(seed_leaves):
            dupes = sorted({leaf.path for leaf in seed_leaves if sum(1 for x in seed_leaves if x.path == leaf.path) > 1})
            raise ValueError(f"duplicate seed leaf paths: {dupes}")
        self._seeds: dict[str, TopicLeaf] = {
            leaf.path: leaf for leaf in sorted(seed_leaves, key=lambda x: x.path)
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TopicTree":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if isinstance(payload, list):
            # Legacy format: flat list of leaves without explicit categories.
            raise ValueError(
                "topic_tree.yaml uses legacy flat-list format; migrate to "
                "'categories' + 'seed_leaves' schema"
            )
        categories = [TopicCategory.model_validate(item) for item in payload.get("categories", [])]
        seed_leaves = [TopicLeaf.model_validate(item) for item in payload.get("seed_leaves", [])]
        return cls(categories=categories, seed_leaves=seed_leaves)

    @classmethod
    def load_default(cls) -> "TopicTree":
        resource = resources.files("driftscope.config").joinpath("topic_tree.yaml")
        return cls.from_yaml(resource)

    # --- category-level API -------------------------------------------------

    def category_ids(self) -> list[str]:
        return list(self._categories.keys())

    def categories(self) -> list[TopicCategory]:
        return list(self._categories.values())

    def has_category(self, category_id: str) -> bool:
        return category_id in self._categories

    def get_category(self, category_id: str) -> TopicCategory:
        return self._categories[category_id]

    def category_for(self, topic_id: str) -> str | None:
        category = category_of_path(topic_id)
        return category if category in self._categories else None

    def default_type_for_category(self, category_id: str) -> str:
        return self._categories[category_id].default_type

    def default_type_for_topic(self, topic_id: str) -> str | None:
        category = self.category_for(topic_id)
        if category is None:
            return None
        return self.default_type_for_category(category)

    def compose_leaf_path(self, category_id: str, leaf_suffix: str) -> str | None:
        if category_id not in self._categories:
            return None
        normalized = normalize_leaf_suffix(leaf_suffix)
        if normalized is None:
            return None
        return f"{category_id}.{normalized}"

    # --- seed-leaf API ------------------------------------------------------

    def topic_ids(self) -> list[TopicID]:
        """Return seed leaf paths (static, YAML-defined topics)."""
        return list(self._seeds.keys())

    def has_topic(self, topic_id: TopicID) -> bool:
        """True iff topic_id is a seed leaf known at YAML-load time.

        Runtime-registered leaves (via ``MemoryBase.canonicalize_topic``) are
        NOT visible here; call ``MemoryBase.is_known_topic`` for the combined
        check.
        """
        return topic_id in self._seeds

    def get(self, topic_id: TopicID) -> TopicLeaf:
        return self._seeds[topic_id]

    def seeds_in_category(self, category_id: str) -> list[TopicLeaf]:
        return [leaf for leaf in self._seeds.values() if category_of_path(leaf.path) == category_id]

    def match(self, text: str) -> TopicID | None:
        best = self.match_with_score(text)
        if best is None or best.score <= 0:
            return None
        return best.path

    def match_with_score(self, text: str) -> TopicMatch | None:
        normalized = _normalize_text(text)
        if not normalized:
            return None

        best_path: TopicID | None = None
        best_score = 0.0
        second_best_score = 0.0
        for path, leaf in self._seeds.items():
            score = self._score_leaf(normalized, leaf)
            if score > best_score or (score == best_score and score > 0 and best_path is not None and path < best_path):
                if path != best_path:
                    second_best_score = best_score
                best_path = path
                best_score = score
            elif score > second_best_score and path != best_path:
                second_best_score = score

        if best_path is None or best_score < _MIN_MATCH_SCORE:
            return None
        if second_best_score > 0 and (best_score - second_best_score) < _MIN_MATCH_MARGIN:
            return None
        return TopicMatch(path=best_path, score=best_score)

    def _score_leaf(self, normalized_text: str, leaf: TopicLeaf) -> float:
        score = 0.0
        token_set = set(_tokenize(normalized_text))
        for keyword in leaf.keywords:
            normalized_keyword = _normalize_text(keyword)
            if normalized_keyword and normalized_keyword in normalized_text:
                score += 3.0
            score += 0.5 * len(token_set.intersection(_tokenize(normalized_keyword)))

        if _normalize_text(leaf.description) in normalized_text:
            score += 2.0
        score += 0.3 * len(token_set.intersection(_tokenize(leaf.description)))

        for example in leaf.examples:
            normalized_example = _normalize_text(example)
            if normalized_example and normalized_example in normalized_text:
                score += 2.0
            score += 0.4 * len(token_set.intersection(_tokenize(normalized_example)))

        leaf_parts = leaf.path.split(".")
        score += 0.2 * len(token_set.intersection({part.lower() for part in leaf_parts}))
        return score


def category_of_path(path: str) -> str:
    parts = path.rsplit(".", 1)
    return parts[0] if len(parts) == 2 else path


def normalize_leaf_suffix(suffix: str) -> str | None:
    if not isinstance(suffix, str):
        return None
    cleaned = suffix.strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not _LEAF_SUFFIX_RE.match(cleaned):
        return None
    return cleaned


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return _TOKEN_RE.findall(normalized)
