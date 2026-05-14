from __future__ import annotations

from datetime import UTC, datetime
import math
import re

from driftscope.agents.base import Agent
from driftscope.agents.topic_predictor import TopicPredictor
from driftscope.agents.types import CandidateMatch, RetrievalInput, RetrievalResult
from driftscope.config.loader import DriftScopeConfig, load_default_config
from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import MemoryEntry
from driftscope.core.topic_tree import TopicTree
from driftscope.retrieval.query_time_parser import QueryTimeHint, QueryTimeParser

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)
_TOPIC_HINT_BONUS = 0.1

_QUOTED_PHRASE_RES: tuple[re.Pattern, ...] = (
    re.compile(r"'([^']{3,60})'"),
    re.compile(r'"([^"]{3,60})"'),
)
_PERSON_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,15}\b")
_NOT_NAMES: frozenset[str] = frozenset({
    "What", "When", "Where", "Who", "How", "Which",
    "Did", "Do", "Was", "Were", "Have", "Has", "Had", "Is", "Are",
    "The", "My", "Our", "Their",
    "Can", "Could", "Would", "Should", "Will", "Shall", "May", "Might",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "June", "July",
    "August", "September", "October", "November", "December",
    "In", "On", "At", "For", "To", "Of", "With", "By", "From", "And", "But",
    "I", "It", "Its", "This", "That", "These", "Those",
    "Previously", "Recently", "Also", "Just", "Very", "More",
})

_STOPWORDS: set[str] = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "done",
    "i", "me", "my", "mine", "we", "our", "us", "you", "your", "yours", "he", "she", "it", "they", "them", "their",
    "this", "that", "these", "those",
    "of", "to", "from", "in", "on", "at", "for", "by", "with", "about", "as", "into", "over", "after", "before",
    "and", "or", "but", "if", "then", "than", "so", "because", "while",
    "what", "when", "where", "which", "who", "whom", "whose", "how", "why",
    "have", "has", "had", "having",
    "can", "could", "should", "would", "will", "may", "might", "must", "shall",
    "not", "no", "yes",
    "s", "t", "d", "ll", "m", "re", "ve",
}


class HeuristicRetrieverAgent(Agent):
    name = "retriever"

    def __init__(
        self,
        *,
        memory_base: MemoryBase,
        topic_tree: TopicTree | None = None,
        config: DriftScopeConfig | None = None,
    ) -> None:
        self.memory_base = memory_base
        self.topic_tree = topic_tree or memory_base.topic_tree
        self.config = config or load_default_config()

    def run(self, input_obj: RetrievalInput) -> RetrievalResult:
        visible = self.memory_base.query_visible(input_obj.scope, input_obj.timestamp)
        predicted_topic = self.topic_tree.match(input_obj.query)
        query_tokens = _tokenize(input_obj.query)
        query_content_tokens = query_tokens - _STOPWORDS
        gating_stats = {
            "visible_before_topic": len(visible),
        }

        idf = _build_idf(visible, input_obj.allow_sensitive_raw)

        candidates = [
            memory
            for memory in visible
            if (
                predicted_topic is not None and memory.topic_id == predicted_topic
            )
            or _overlap(query_content_tokens, _memory_tokens(memory, input_obj.allow_sensitive_raw)) > 0
        ]
        if not candidates:
            candidates = visible

        gating_stats["after_topic_filter"] = len(candidates)

        scored: list[CandidateMatch] = []
        for memory in candidates:
            score_breakdown = self._score_memory(
                memory,
                query_content_tokens,
                input_obj.timestamp,
                input_obj.allow_sensitive_raw,
                predicted_topic=predicted_topic,
                idf=idf,
            )
            score = (
                self.config.retrieval.lambda_r * score_breakdown["rel"]
                + self.config.retrieval.lambda_f * score_breakdown["fresh"]
                + self.config.retrieval.lambda_c * score_breakdown["conf"]
                + _TOPIC_HINT_BONUS * score_breakdown["topic_hint"]
            ) * score_breakdown["source_weight"]
            scored.append(
                CandidateMatch(
                    memory=memory,
                    score=score,
                    score_breakdown=score_breakdown,
                    matched_by=_matched_by(memory, predicted_topic, query_content_tokens, input_obj.allow_sensitive_raw),
                )
            )

        scored.sort(key=lambda item: (-item.score, -item.memory.ingest_time.timestamp(), item.memory.id))
        ranked = [match for match in scored if match.memory.type != "constraint"][: self.config.retrieval.top_k]

        injected_constraints: list[MemoryEntry] = []
        seen_ids = {match.memory.id for match in ranked}
        for memory in scored:
            item = memory.memory
            if item.type != "constraint" or item.id in seen_ids:
                continue
            if predicted_topic is not None and item.topic_id == predicted_topic:
                injected_constraints.append(item)
                seen_ids.add(item.id)
                continue
            if predicted_topic is None and _overlap(query_content_tokens, _memory_tokens(item, input_obj.allow_sensitive_raw)) > 0:
                injected_constraints.append(item)
                seen_ids.add(item.id)

        gating_stats["after_scoring"] = len(ranked)
        gating_stats["injected_constraints"] = len(injected_constraints)
        return RetrievalResult(
            ranked_memories=ranked,
            injected_constraints=injected_constraints,
            gating_stats=gating_stats,
            predicted_topic=predicted_topic,
        )

    def _score_memory(
        self,
        memory: MemoryEntry,
        query_tokens: set[str],
        timestamp: datetime,
        allow_sensitive_raw: bool,
        *,
        predicted_topic: str | None,
        idf: dict[str, float],
    ) -> dict[str, float]:
        rel = _idf_coverage(query_tokens, _memory_tokens(memory, allow_sensitive_raw), idf)
        anchor_time = _ensure_aware(
            memory.event_time if memory.type in ("episodic", "raw_session") and memory.event_time is not None else memory.ingest_time
        )
        query_time = _ensure_aware(timestamp)
        age_seconds = max((query_time - anchor_time).total_seconds(), 0.0)
        age_days = age_seconds / 86400.0
        if memory.type == "preference":
            fresh = math.exp(-age_days / max(self.config.retrieval.tau_pref, 1e-6))
        elif memory.type in ("episodic", "raw_session"):
            fresh = math.exp(-age_days / max(self.config.retrieval.tau_episodic, 1e-6))
        elif memory.type == "fact":
            fresh = math.exp(-age_days / max(self.config.retrieval.tau_fact, 1e-6))
        else:
            fresh = 1.0
        return {
            "rel": rel,
            "fresh": fresh,
            "conf": memory.conf.combined,
            "topic_hint": 1.0 if predicted_topic is not None and memory.topic_id == predicted_topic else 0.0,
            "source_weight": _source_weight(memory),
        }


def _memory_tokens(memory: MemoryEntry, allow_sensitive_raw: bool) -> set[str]:
    if memory.sensitive and not allow_sensitive_raw:
        text = memory.summary_for_retrieval or ""
    else:
        text = memory.content
    return _tokenize(text)


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "")}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    denom = len(left.union(right))
    if denom == 0:
        return 0.0
    return len(left.intersection(right)) / denom


def _overlap(left: set[str], right: set[str]) -> int:
    if not left or not right:
        return 0
    return len(left & right)


def _build_idf(memories, allow_sensitive_raw: bool) -> dict[str, float]:
    n = len(memories) or 1
    df: dict[str, int] = {}
    for memory in memories:
        for token in _memory_tokens(memory, allow_sensitive_raw):
            df[token] = df.get(token, 0) + 1
    return {
        token: math.log((n + 1) / (count + 1)) + 1.0
        for token, count in df.items()
    }


def _idf_coverage(query_tokens: set[str], memory_tokens: set[str], idf: dict[str, float]) -> float:
    if not query_tokens:
        return 0.0
    total = sum(idf.get(tok, 1.0) for tok in query_tokens)
    if total <= 0:
        return 0.0
    matched = sum(idf.get(tok, 1.0) for tok in query_tokens & memory_tokens)
    return matched / total


def _source_weight(memory: MemoryEntry) -> float:
    if memory.type == "raw_session":
        return 0.4 if memory.origin_role == "assistant" else 0.7
    if memory.origin_role == "assistant":
        return 0.65
    if memory.source_kind != "explicit":
        return 0.8
    return 1.0


def _matched_by(
    memory: MemoryEntry,
    predicted_topic: str | None,
    query_tokens: set[str],
    allow_sensitive_raw: bool,
) -> list[str]:
    matched: list[str] = []
    if predicted_topic is not None and memory.topic_id == predicted_topic:
        matched.append("topic_hint")
    if _overlap(query_tokens, _memory_tokens(memory, allow_sensitive_raw)) > 0:
        matched.append("lexical_overlap")
    return matched or ["fallback_visible"]


class HybridRetrieverAgent(Agent):
    """Dense + sparse retrieval fused with RRF, plus time-proximity scoring.

    Falls back to sparse-only scoring when no embedder is wired or when the
    memory store has no dense vectors. A sibling of HeuristicRetrieverAgent —
    same inputs/outputs, different ranker.
    """

    name = "retriever"

    def __init__(
        self,
        *,
        memory_base: MemoryBase,
        topic_tree: TopicTree | None = None,
        config: DriftScopeConfig | None = None,
        embedder=None,
        query_time_parser: QueryTimeParser | None = None,
        topic_predictor: TopicPredictor | None = None,
    ) -> None:
        self.memory_base = memory_base
        self.topic_tree = topic_tree or memory_base.topic_tree
        self.config = config or load_default_config()
        self.embedder = embedder if embedder is not None else getattr(memory_base, "embedder", None)
        self.query_time_parser = query_time_parser
        self.topic_predictor = topic_predictor

    def run(self, input_obj: RetrievalInput) -> RetrievalResult:
        paired = self.memory_base.query_visible_with_vectors(input_obj.scope, input_obj.timestamp)
        query_tokens = _tokenize(input_obj.query)
        query_content_tokens = query_tokens - _STOPWORDS
        time_hint = self._parse_time_hint(input_obj.query, input_obj.timestamp)
        quoted_phrases = _extract_quoted_phrases(input_obj.query)
        person_names = _extract_person_names(input_obj.query)

        query_vector = None
        if self.embedder is not None and input_obj.query.strip():
            try:
                query_vector = self.embedder.embed([input_obj.query])[0]
            except Exception:
                query_vector = None

        legacy_topic = self.topic_tree.match(input_obj.query)
        predicted_via_embedding = False
        if legacy_topic is None and query_vector is not None:
            legacy_topic = self._predict_topic_from_embedding(query_vector)
            predicted_via_embedding = legacy_topic is not None

        candidate_topics: list[str | None]
        predictor_failed = False
        if self.topic_predictor is not None:
            all_paths = [path for path, _, _ in self.memory_base.all_known_leaves()]
            try:
                predicted_paths = self.topic_predictor.predict(input_obj.query, all_paths)
            except Exception:
                predicted_paths = []
                predictor_failed = True
            candidate_topics = list(predicted_paths) if predicted_paths else [legacy_topic]
        else:
            candidate_topics = [legacy_topic]

        leaf_vectors_by_path: dict[str, object] = {}
        seen_categories: set[str] = set()
        for topic in candidate_topics:
            if topic is None:
                continue
            cat = self.topic_tree.category_for(topic)
            if cat is None or cat in seen_categories:
                continue
            seen_categories.add(cat)
            for path, vec in self.memory_base.known_leaves_in_category(cat):
                if vec is not None:
                    leaf_vectors_by_path[path] = vec

        gating_stats: dict[str, int] = {
            "visible_before_topic": len(paired),
            "predicted_via_embedding": int(predicted_via_embedding),
            "candidate_topics": len(candidate_topics),
            "predictor_failed": int(predictor_failed),
        }

        visible = [memory for memory, _ in paired]
        idf = _build_idf(visible, input_obj.allow_sensitive_raw)

        sparse_scores: dict[str, float] = {}
        for memory in visible:
            sparse_scores[memory.id] = _idf_coverage(
                query_content_tokens,
                _memory_tokens(memory, input_obj.allow_sensitive_raw),
                idf,
            )

        dense_scores: dict[str, float] = {}
        if query_vector is not None:
            import numpy as np

            for memory, vector in paired:
                if vector is None:
                    continue
                denom = float(np.linalg.norm(vector)) * float(np.linalg.norm(query_vector))
                if denom <= 0.0:
                    continue
                dense_scores[memory.id] = float(np.dot(vector, query_vector) / denom)

        sparse_rank = _ranks_from_scores(sparse_scores, self.config.retrieval.sparse_top_n)
        dense_rank = _ranks_from_scores(dense_scores, self.config.retrieval.dense_top_n)

        rrf_k = max(1, int(self.config.retrieval.rrf_k))
        rrf_scores: dict[str, float] = {}
        for mid, rank in sparse_rank.items():
            rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (rrf_k + rank)
        for mid, rank in dense_rank.items():
            rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (rrf_k + rank)

        if not rrf_scores:
            rrf_scores = {memory.id: 0.0 for memory in visible}

        gating_stats["after_sparse_topn"] = len(sparse_rank)
        gating_stats["after_dense_topn"] = len(dense_rank)
        gating_stats["hybrid_candidates"] = len(rrf_scores)

        max_rrf = max(rrf_scores.values(), default=0.0) or 1.0

        pass_rankings: list[list[CandidateMatch]] = []
        for topic in candidate_topics:
            cat_for_pass = self.topic_tree.category_for(topic) if topic else None
            scored_pass = self._score_pass(
                visible=visible,
                predicted_topic=topic,
                predicted_category=cat_for_pass,
                leaf_vectors_by_path=leaf_vectors_by_path,
                timestamp=input_obj.timestamp,
                allow_sensitive_raw=input_obj.allow_sensitive_raw,
                time_hint=time_hint,
                sparse_scores=sparse_scores,
                dense_scores=dense_scores,
                rrf_scores=rrf_scores,
                max_rrf=max_rrf,
                quoted_phrases=quoted_phrases,
                person_names=person_names,
                query_content_tokens=query_content_tokens,
                sparse_rank=sparse_rank,
                dense_rank=dense_rank,
            )
            pass_rankings.append(scored_pass)

        if len(pass_rankings) == 1:
            scored = pass_rankings[0]
        else:
            scored = self._rrf_merge_passes(
                pass_rankings,
                k=rrf_k,
                top_k_per_pass=max(self.config.retrieval.top_k * 2, 30),
            )

        ranked = [match for match in scored if match.memory.type != "constraint"][: self.config.retrieval.top_k]

        candidate_topic_set = {t for t in candidate_topics if t is not None}
        injected_constraints: list[MemoryEntry] = []
        seen_ids = {match.memory.id for match in ranked}
        for match in scored:
            item = match.memory
            if item.type != "constraint" or item.id in seen_ids:
                continue
            if candidate_topic_set and item.topic_id in candidate_topic_set:
                injected_constraints.append(item)
                seen_ids.add(item.id)
                continue
            if not candidate_topic_set and _overlap(query_content_tokens, _memory_tokens(item, input_obj.allow_sensitive_raw)) > 0:
                injected_constraints.append(item)
                seen_ids.add(item.id)

        gating_stats["after_scoring"] = len(ranked)
        gating_stats["injected_constraints"] = len(injected_constraints)
        return RetrievalResult(
            ranked_memories=ranked,
            injected_constraints=injected_constraints,
            gating_stats=gating_stats,
            predicted_topic=candidate_topics[0] if candidate_topics else None,
        )

    def _score_pass(
        self,
        *,
        visible: list[MemoryEntry],
        predicted_topic: str | None,
        predicted_category: str | None,
        leaf_vectors_by_path: dict[str, object],
        timestamp: datetime,
        allow_sensitive_raw: bool,
        time_hint: QueryTimeHint | None,
        sparse_scores: dict[str, float],
        dense_scores: dict[str, float],
        rrf_scores: dict[str, float],
        max_rrf: float,
        quoted_phrases: list[str],
        person_names: list[str],
        query_content_tokens: set[str],
        sparse_rank: dict[str, int],
        dense_rank: dict[str, int],
    ) -> list[CandidateMatch]:
        """Score every visible memory under a single predicted_topic, sorted desc."""
        lambda_time = self.config.retrieval.lambda_time
        scored: list[CandidateMatch] = []
        for memory in visible:
            rel = rrf_scores.get(memory.id, 0.0) / max_rrf
            breakdown = self._score_memory(
                memory=memory,
                rel=rel,
                timestamp=timestamp,
                allow_sensitive_raw=allow_sensitive_raw,
                predicted_topic=predicted_topic,
                predicted_category=predicted_category,
                leaf_vectors_by_path=leaf_vectors_by_path,
                time_hint=time_hint,
                sparse_score=sparse_scores.get(memory.id, 0.0),
                dense_score=dense_scores.get(memory.id, 0.0),
                quoted_phrases=quoted_phrases,
                person_names=person_names,
            )
            score = (
                self.config.retrieval.lambda_r * breakdown["rel"]
                + self.config.retrieval.lambda_f * breakdown["fresh"]
                + self.config.retrieval.lambda_c * breakdown["conf"]
                + _TOPIC_HINT_BONUS * breakdown["topic_hint"]
                + lambda_time * breakdown["time_prox"]
                + self.config.retrieval.lambda_quoted * breakdown["quoted"]
                + self.config.retrieval.lambda_person * breakdown["person"]
            ) * breakdown["source_weight"]
            matched = _hybrid_matched_by(
                memory=memory,
                predicted_topic=predicted_topic,
                query_tokens=query_content_tokens,
                allow_sensitive_raw=allow_sensitive_raw,
                in_dense=memory.id in dense_rank,
                in_sparse=memory.id in sparse_rank,
                time_hit=breakdown["time_prox"] > 0.0 and time_hint is not None,
                quoted_hit=breakdown["quoted"] > 0.0,
                person_hit=breakdown["person"] > 0.0,
            )
            scored.append(
                CandidateMatch(
                    memory=memory,
                    score=score,
                    score_breakdown=breakdown,
                    matched_by=matched,
                )
            )
        scored.sort(key=lambda item: (-item.score, -item.memory.ingest_time.timestamp(), item.memory.id))
        return scored

    def _rrf_merge_passes(
        self,
        pass_rankings: list[list[CandidateMatch]],
        *,
        k: int,
        top_k_per_pass: int,
    ) -> list[CandidateMatch]:
        """Combine N independent rankings via Reciprocal Rank Fusion.

        Each pass contributes 1/(k + rank) for memories in its top
        ``top_k_per_pass``. Memories appearing in multiple passes accumulate.
        The CandidateMatch returned for each merged memory is taken from the
        pass where it scored highest, so matched_by/score_breakdown explain
        the strongest topic interpretation.
        """
        rrf: dict[str, float] = {}
        best_match_by_id: dict[str, CandidateMatch] = {}
        for ranking in pass_rankings:
            for rank, match in enumerate(ranking[:top_k_per_pass], start=1):
                mid = match.memory.id
                rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (k + rank)
                existing = best_match_by_id.get(mid)
                if existing is None or match.score > existing.score:
                    best_match_by_id[mid] = match

        merged: list[CandidateMatch] = []
        for mid in sorted(
            rrf.keys(),
            key=lambda m: (-rrf[m], -best_match_by_id[m].memory.ingest_time.timestamp(), m),
        ):
            base = best_match_by_id[mid]
            merged.append(
                CandidateMatch(
                    memory=base.memory,
                    score=rrf[mid],
                    score_breakdown=base.score_breakdown,
                    matched_by=base.matched_by,
                )
            )
        return merged

    def _parse_time_hint(self, query: str, timestamp: datetime) -> QueryTimeHint | None:
        if not self.config.retrieval.query_time_parse_enabled:
            return None
        if self.query_time_parser is None:
            return None
        try:
            return self.query_time_parser.parse(query=query, now=timestamp)
        except Exception:
            return None

    def _score_memory(
        self,
        *,
        memory: MemoryEntry,
        rel: float,
        timestamp: datetime,
        allow_sensitive_raw: bool,
        predicted_topic: str | None,
        predicted_category: str | None,
        leaf_vectors_by_path: dict[str, object],
        time_hint: QueryTimeHint | None,
        sparse_score: float,
        dense_score: float,
        quoted_phrases: list[str],
        person_names: list[str],
    ) -> dict[str, float]:
        anchor_time = _ensure_aware(
            memory.event_time if memory.type in ("episodic", "raw_session") and memory.event_time is not None else memory.ingest_time
        )
        query_time = _ensure_aware(timestamp)
        age_seconds = max((query_time - anchor_time).total_seconds(), 0.0)
        age_days = age_seconds / 86400.0
        if memory.type == "preference":
            fresh = math.exp(-age_days / max(self.config.retrieval.tau_pref, 1e-6))
        elif memory.type in ("episodic", "raw_session"):
            fresh = math.exp(-age_days / max(self.config.retrieval.tau_episodic, 1e-6))
        elif memory.type == "fact":
            fresh = math.exp(-age_days / max(self.config.retrieval.tau_fact, 1e-6))
        else:
            fresh = 1.0
        time_prox = 0.0
        if time_hint is not None:
            anchor = _ensure_aware(memory.event_time or memory.ingest_time)
            hint_center = _ensure_aware(time_hint.center)
            delta_seconds = abs((anchor - hint_center).total_seconds())
            delta_days = delta_seconds / 86400.0
            time_prox = math.exp(-delta_days / max(self.config.retrieval.tau_time, 1e-6))
        doc_text = _scorable_text(memory, allow_sensitive_raw)
        quoted = _phrase_hit_ratio(quoted_phrases, doc_text)
        person = _name_hit_ratio(person_names, doc_text)
        topic_hint = self._topic_hint_score(
            memory_topic_id=memory.topic_id,
            predicted_topic=predicted_topic,
            predicted_category=predicted_category,
            leaf_vectors_by_path=leaf_vectors_by_path,
        )
        return {
            "rel": rel,
            "fresh": fresh,
            "conf": memory.conf.combined,
            "topic_hint": topic_hint,
            "source_weight": _source_weight(memory),
            "time_prox": time_prox,
            "sparse": sparse_score,
            "dense": dense_score,
            "quoted": quoted,
            "person": person,
        }

    def _predict_topic_from_embedding(self, query_vector) -> str | None:
        """Nearest-leaf fallback when keyword match misses.

        Scans all known leaves (seed + runtime-registered) across every
        category and returns the leaf path whose suffix embedding has the
        highest cosine with the query — provided it passes
        ``topic_query_predict_threshold``. This is what lets queries targeting
        runtime-registered leaves (e.g. ``user.preference.music`` if ``music``
        was not a seed keyword) still activate the topic_hint signal.
        """
        if query_vector is None:
            return None

        import numpy as np

        q_norm = float(np.linalg.norm(query_vector))
        if q_norm <= 0.0:
            return None

        best_path: str | None = None
        best_cos = -1.0
        for path, _category, vec in self.memory_base.all_known_leaves():
            if vec is None:
                continue
            denom = float(np.linalg.norm(vec)) * q_norm
            if denom <= 0.0:
                continue
            cos = float(np.dot(vec, query_vector) / denom)
            if cos > best_cos:
                best_cos = cos
                best_path = path

        if best_path is None or best_cos < self.config.retrieval.topic_query_predict_threshold:
            return None
        return best_path

    def _topic_hint_score(
        self,
        *,
        memory_topic_id: str | None,
        predicted_topic: str | None,
        predicted_category: str | None,
        leaf_vectors_by_path: dict[str, object],
    ) -> float:
        """Topic hint scoring:

        - exact leaf match → 1.0
        - cross-category → 0.0
        - same-category, cosine ≥ threshold → soft_floor..1.0 (graded)
        - same-category, cosine < threshold OR vectors missing → sibling_floor (baseline)
        """
        if predicted_topic is None or memory_topic_id is None:
            return 0.0
        if memory_topic_id == predicted_topic:
            return 1.0
        if predicted_category is None:
            return 0.0
        memory_category = self.topic_tree.category_for(memory_topic_id)
        if memory_category != predicted_category:
            return 0.0

        sibling_floor = self.config.retrieval.topic_sibling_floor

        predicted_vec = leaf_vectors_by_path.get(predicted_topic)
        memory_vec = leaf_vectors_by_path.get(memory_topic_id)
        if predicted_vec is None or memory_vec is None:
            return sibling_floor

        import numpy as np

        denom = float(np.linalg.norm(predicted_vec)) * float(np.linalg.norm(memory_vec))
        if denom <= 0.0:
            return sibling_floor
        cosine = float(np.dot(predicted_vec, memory_vec) / denom)

        threshold = self.config.retrieval.topic_soft_hint_threshold
        floor = self.config.retrieval.topic_soft_hint_floor
        if cosine < threshold:
            return sibling_floor
        if threshold >= 1.0:
            return 1.0
        normalized = (cosine - threshold) / (1.0 - threshold)
        return floor + (1.0 - floor) * max(0.0, min(1.0, normalized))


def _ensure_aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _scorable_text(memory: MemoryEntry, allow_sensitive_raw: bool) -> str:
    if memory.sensitive and not allow_sensitive_raw:
        base = memory.summary_for_retrieval or ""
    else:
        base = memory.content or ""
    if memory.evidence:
        return f"{base}\n{memory.evidence}"
    return base


def _extract_quoted_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for pattern in _QUOTED_PHRASE_RES:
        phrases.extend(pattern.findall(text))
    return [p.strip() for p in phrases if len(p.strip()) >= 3]


def _extract_person_names(text: str) -> list[str]:
    names = {match for match in _PERSON_NAME_RE.findall(text) if match not in _NOT_NAMES}
    return sorted(names)


def _phrase_hit_ratio(phrases: list[str], doc: str) -> float:
    if not phrases:
        return 0.0
    doc_lower = doc.lower()
    hits = sum(1 for p in phrases if p.lower() in doc_lower)
    return min(hits / len(phrases), 1.0)


def _name_hit_ratio(names: list[str], doc: str) -> float:
    if not names:
        return 0.0
    doc_lower = doc.lower()
    hits = sum(1 for n in names if n.lower() in doc_lower)
    return min(hits / len(names), 1.0)


def _ranks_from_scores(scores: dict[str, float], top_n: int) -> dict[str, int]:
    if not scores:
        return {}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, int] = {}
    for index, (mid, score) in enumerate(ordered[: max(1, top_n)], start=1):
        if score <= 0.0:
            break
        ranks[mid] = index
    return ranks


def _hybrid_matched_by(
    *,
    memory: MemoryEntry,
    predicted_topic: str | None,
    query_tokens: set[str],
    allow_sensitive_raw: bool,
    in_dense: bool,
    in_sparse: bool,
    time_hit: bool,
    quoted_hit: bool = False,
    person_hit: bool = False,
) -> list[str]:
    matched: list[str] = []
    if predicted_topic is not None and memory.topic_id == predicted_topic:
        matched.append("topic_hint")
    if in_sparse and _overlap(query_tokens, _memory_tokens(memory, allow_sensitive_raw)) > 0:
        matched.append("lexical_overlap")
    if in_dense:
        matched.append("dense")
    if time_hit:
        matched.append("time_prox")
    if quoted_hit:
        matched.append("quoted_phrase")
    if person_hit:
        matched.append("person_name")
    return matched or ["fallback_visible"]
