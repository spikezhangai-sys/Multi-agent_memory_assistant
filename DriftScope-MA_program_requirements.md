# DriftScope-MA 系统实现需求文档（Program Requirements）

> 本文档是 DriftScope-MA 在本仓库中的唯一实现规格。
> 历史上的 v1–v5 修订稿只作为背景来源，不作为实现依赖。
> 若正文、配置文件示例、测试夹具之间冲突，以正文规则为准。
> 假设环境：Python 3.10+，单机离线评测，LLM 通过 API 或本地推理服务调用。

---

## 1. 文档定位与范围

### 1.1 目标

实现一个支持 LongMemEval 评测的 personal assistant memory 系统。P0 目标是把主流程跑通并形成可重复评测的工程实现；研究扩展能力保留在同一份文档中，但不作为 LongMemEval 主线的硬阻塞条件。

### 1.2 分阶段能力清单

| 能力 | 优先级 | 主验收路径 |
|---|---|---|
| 版本化 memory 写入与查询 | P0 | LongMemEval 主线 |
| 显式 `transition_type` 标记 | P0 | LongMemEval 主线 |
| `fact / preference / constraint` 三类 memory | P0 | LongMemEval 主线 |
| 4 agent 数据流（Update / Conflict / Retriever / Response）| P0 | LongMemEval 主线 |
| 按需激活机制 | P0 | LongMemEval 主线 |
| 两阶段检索（Gating + Scoring）| P0 | LongMemEval 主线 |
| Constraint Injection | P0 | LongMemEval 主线 |
| Scope 隔离与 `ScopeCompat` 检查 | P1 | 扩展集成测试 |
| 用户撤销与 rollback | P1 | 扩展集成测试 |
| `sensitive` 字段与 `summary_for_retrieval` | P1 | 扩展集成测试 |
| Scope Promotion / Demotion | P2 | 后续研究扩展 |
| Partial supersede（字段级）| P2 | 后续研究扩展 |
| 关系索引（`same_topic` / `potential_tension`）| P3 | 后续研究扩展 |

P0 的实现和验收必须自洽且独立完成。P1 可以与 P0 共用接口和数据结构，但不得改变 P0 的主流程语义。P2/P3 不进入当前实现承诺。

### 1.3 非目标

- Web UI
- 多用户管理
- 在线高并发服务
- DriftScopeBench 数据集本身
- 依赖外部论文草稿才能理解的隐式规则

---

## 2. 模块与职责

### 2.1 建议目录结构

```text
driftscope/
├── core/
│   ├── schema.py                # Scope, Confidence, MemoryEntry, SupersedeLink 等
│   ├── memory_base.py           # MemoryBase，封装 CRUD、索引、dump/load
│   ├── topic_tree.py            # 受控 topic 树，加载、匹配、查询
│   └── scope_compat.py          # ScopeVisible / ScopeCompat 规则
│
├── agents/
│   ├── base.py                  # Agent 抽象基类与统一 LLM 调用
│   ├── update_agent.py          # 抽取写入意图与候选 memory
│   ├── conflict_agent.py        # 目标定位、兼容性检查、状态迁移决策
│   ├── retriever_agent.py       # Gating、Scoring、Constraint Injection
│   └── response_agent.py        # 回答生成与引用校验
│
├── pipeline/
│   ├── orchestrator.py          # Turn 编排与 agent 按需激活
│   └── transitions.py           # 状态迁移与 rollback 算子
│
├── eval/
│   ├── longmemeval/
│   │   ├── adapter.py           # LongMemEval 数据格式 -> TurnInput
│   │   ├── runner.py            # 评测主循环
│   │   └── metrics.py           # 离线指标计算
│   ├── baselines/
│   │   ├── flat_rag.py          # 研究目标对比基线
│   │   └── single_agent.py      # 研究目标对比基线
│   └── instrumentation.py       # JSONL turn log、违反检测、汇总
│
├── llm/
│   ├── client.py                # 统一 LLM API 封装
│   ├── prompts/                 # 各 agent prompt 模板
│   └── parsers.py               # 结构化输出解析与校验
│
├── config/
│   ├── topic_tree.yaml          # 固定受控 topic 叶子集合
│   ├── scope_compat.yaml        # 与本文 §4 镜像一致的规则文件
│   └── default.yaml             # 阈值、权重、窗口等超参
│
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

### 2.2 依赖原则

```text
eval -> pipeline -> agents -> llm
                |          |
                v          v
               core       core
```

- `core` 不调用 LLM。
- `pipeline` 负责流程编排，不持有业务规则真源；规则真源来自本文和 `core`。
- P0 为离线评测优先，主流程默认顺序执行，保证结果可复现。并发读写不是 P0 硬要求。

---

## 3. 公共数据结构

### 3.1 TopicID

```python
TopicID = str
```

约束：

- `TopicID` 必须等于 `topic_tree.yaml` 中某个叶子节点的完整路径，例如 `user.preference.food`
- P0 不要求固定 30 个叶子，但要求 topic 树在代码仓库中固化，运行时不可动态改写
- `topic_tree.yaml` 是叶子集合的实现载体；本文不再引用外部草稿中的叶子清单

### 3.2 Scope

```python
@dataclass(frozen=True)
class Scope:
    kind: Literal["global", "personal", "project", "session"]
    ref: str | None = None
```

约束：

- `kind in {"global", "personal"}` 时，`ref` 必须为 `None`
- `kind in {"project", "session"}` 时，`ref` 必须为非空字符串
- 示例：`Scope(kind="project", ref="alpha")`
- 两个 `Scope` 只有在 `kind` 和 `ref` 都一致时才视为同一作用域

### 3.3 Confidence

```python
@dataclass
class Confidence:
    prior: float
    llm_self: float | None
    combined: float
```

约束：

- 所有数值范围均为 `[0.0, 1.0]`
- `combined = w_p * prior + w_l * llm_self`，权重来自 `default.yaml`
- 当 `llm_self is None` 时，`combined = prior`

### 3.4 TimeRange

```python
@dataclass(frozen=True)
class TimeRange:
    start: datetime
    end: datetime | None = None
```

约束：

- `end is None` 表示无穷远
- 若 `end is not None`，则必须满足 `start <= end`
- `valid_time` 用于“事实在什么时候为真”，`ingest_time` 用于“系统在什么时候写入”

### 3.5 MemoryEntry

```python
@dataclass
class MemoryEntry:
    id: str
    content: str
    type: Literal["fact", "preference", "constraint"]
    topic_id: TopicID
    scope: Scope
    src: Literal["user_explicit", "user_implicit", "inferred", "external"]
    conf: Confidence
    valid_time: TimeRange
    ingest_time: datetime
    state: Literal["active", "superseded", "revoked"]
    revoked_at: datetime | None = None
    supersedes: list["SupersedeLink"] = field(default_factory=list)
    sensitive: bool = False
    summary_for_retrieval: str | None = None
```

约束：

- `id` 全系统唯一，由 `uuid4()` 生成
- P0 运行时只允许 `type in {"fact", "preference", "constraint"}`
- `summary_for_retrieval` 仅在 `sensitive == True` 时必填
- `state == "revoked"` 时，`revoked_at` 必填
- `state != "revoked"` 时，`revoked_at` 必须为 `None`
- 序列化格式为 JSON，反序列化必须做 schema 校验

说明：

- `episodic / task / procedural` 不在当前版本规格内；若未来支持，需新增独立规则表与检索语义
- P0 中“最新显式偏好覆盖旧偏好”；偏好漂移通过 `transition_type="preference_shifted"` 记录，而不是通过多条 active preference 并存实现

### 3.6 SupersedeLink

```python
@dataclass
class SupersedeLink:
    target: str
    mode: Literal["full"]
    transition_type: Literal["corrected", "preference_shifted", "user_revoked"]
```

约束：

- 当前版本只支持 `mode == "full"`
- `target` 必须指向已存在的 `MemoryEntry.id`
- `transition_type` 的合法性由本文 §4.3 定义

说明：

- Partial supersede 已延期到 P2，不进入当前实现承诺

### 3.7 TopicQuery、ScoredMemory、TurnInput、TurnResult

```python
@dataclass
class TopicQuery:
    topic_id: TopicID | None
    keywords: list[str]

@dataclass
class ScoredMemory:
    memory: MemoryEntry
    score: float
    score_breakdown: dict[str, float]
    gating_trace: dict[str, str]

@dataclass
class TurnInput:
    scope: Scope
    timestamp: datetime
    user_input: str | None = None
    query: str | None = None

@dataclass
class TurnResult:
    answer: str | None
    cited_memory_ids: list[str]
    agents_called: list[str]
    write_applied: bool
    write_only: bool
    query_only: bool
    errors: list[str]
```

约束：

- `TurnInput.user_input` 与 `TurnInput.query` 至少有一个非空
- LongMemEval replay turn 使用 `user_input != None, query is None`
- LongMemEval final question turn 使用 `user_input is None, query != None`
- 交互式 mixed turn 允许二者同时非空，但不属于 LongMemEval 主线

### 3.8 MemoryBase

```python
class MemoryBase:
    def add(self, m: MemoryEntry) -> None: ...
    def get(self, id: str) -> MemoryEntry: ...
    def update_state(
        self,
        id: str,
        new_state: Literal["active", "superseded", "revoked"],
        revoked_at: datetime | None = None,
    ) -> None: ...
    def add_supersede_link(self, new_id: str, link: SupersedeLink) -> None: ...
    def query_by_topic(self, topic_id: TopicID, scope: Scope, time: datetime) -> list[MemoryEntry]: ...
    def query_visible(self, scope: Scope, time: datetime) -> list[MemoryEntry]: ...
    def query_revoked_within(
        self,
        window_days: int,
        scope: Scope | None = None,
        time: datetime | None = None,
    ) -> list[MemoryEntry]: ...
    def get_supersede_chain(self, id: str, direction: Literal["forward", "backward"]) -> list[MemoryEntry]: ...
    def rollback(self, id: str) -> bool: ...
    def dump_json(self, path: str) -> None: ...
    def load_json(self, path: str) -> None: ...
```

性能要求：

- 千条 memory 规模下，`query_by_topic` 与 `query_visible` 单次调用小于 50ms
- `add`、`update_state`、`rollback` 单次调用小于 10ms
- 全量 dump/load 小于 5s

实现建议：

- SQLite 作为主存储，按 `topic_id`、`scope.kind`、`scope.ref`、`state` 建索引
- 额外提供 JSON dump/load，便于调试与实验复现

---

## 4. 规则与状态迁移

### 4.1 ScopeVisible：检索可见性规则

检索时，当前 turn 的 `scope` 决定可见 memory 的上界。规则如下：

| 当前 scope | 可见 memory scope |
|---|---|
| `global` | `global` |
| `personal` | `global`, `personal` |
| `project::<x>` | `global`, `personal`, `project::<x>` |
| `session::<x>` | `global`, `personal`, `session::<x>` |

补充约束：

- `project::<alpha>` 不可见 `project::<beta>`
- `session::<a>` 不可见 `session::<b>`
- `session` 与 `project` 之间不自动互通
- LongMemEval 适配器固定使用 `personal`，因此 LongMemEval 主线只依赖 `global + personal` 的可见性

### 4.2 ScopeCompat：覆盖与撤销兼容性规则

发生 `supersede`、`revoke`、`rollback` 时，目标 memory 必须与当前写入 scope 兼容。当前版本采用“同 scope 精确匹配”规则：

| 当前写入 scope | 允许作用的目标 scope |
|---|---|
| `global` | `global` |
| `personal` | `personal` |
| `project::<x>` | `project::<x>` |
| `session::<x>` | `session::<x>` |

处理要求：

- 不允许跨 scope 自动覆盖
- 不允许 `session` 自动提升为 `personal`
- `Scope Promotion / Demotion` 不在当前版本内；兼容性检查失败时，`Conflict Agent` 必须返回 `request_clarification`

### 4.3 `type × transition_type` 合法组合表

| 目标 memory.type | `corrected` | `preference_shifted` | `user_revoked` |
|---|---|---|---|
| `fact` | 允许 | 不允许 | 允许 |
| `preference` | 允许 | 允许 | 允许 |
| `constraint` | 允许 | 不允许 | 允许 |

处理要求：

- `supersede_full` 必须显式给出 `transition_type`
- `revoke` 固定使用 `transition_type="user_revoked"`
- 当 `UpdateProposal.candidate is None` 时，合法性在 `Conflict Agent` 解析出目标 memory 后校验
- 非法组合必须拒绝写入并记录错误，不得静默降级为其他 `transition_type`

### 4.4 状态迁移语义

#### 4.4.1 `add`

- 写入一条新的 `active` memory
- 不创建 `SupersedeLink`
- 适用于此前没有同义历史、或找不到明确目标的 standalone 事实

#### 4.4.2 `supersede_full`

- 新 memory 写入为 `active`
- 旧 memory 状态更新为 `superseded`
- 在新 memory 上附加 `SupersedeLink(target=old_id, mode="full", transition_type=...)`
- `transition_type="corrected"` 与 `transition_type="preference_shifted"` 都遵循上述状态变化；两者差异只体现在语义标签与分析指标

#### 4.4.3 `revoke`

- 不创建新的 `MemoryEntry`
- 目标 memory 从 `active` 更新为 `revoked`
- 设置 `revoked_at = 当前 turn timestamp`
- `instrumentation` 必须记录 revoke 事件、执行人、目标 memory id

#### 4.4.4 `rollback`

- 仅对 `state == "revoked"` 的 memory 生效
- 成功时将目标 memory 由 `revoked` 变回 `active`
- 成功时清空 `revoked_at`
- rollback 本身通过 turn log 记录，不要求新建 tombstone memory

### 4.5 Rollback 合法条件

`handle_rollback()` 在执行前必须满足以下三个条件：

1. 目标 memory 当前处于 `revoked` 状态。
2. `当前时间 - revoked_at <= rollback_window_days`。
3. 在同一 `scope`、同一 `topic_id` 下，不存在创建时间晚于 `revoked_at` 的 `active` memory。

处理要求：

- 任一条件不满足时返回 `False`，不修改任何状态
- `query_revoked_within()` 的候选集只包含窗口内且未归档的 revoked memory
- 若 `locate_by_hint()` 定位到多个得分相同的目标，视为歧义，返回 `False` 并要求澄清

---

## 5. Agent 接口需求

### 5.1 通用 Agent 协议

```python
class Agent(ABC):
    name: str

    @abstractmethod
    def run(self, input_obj) -> object: ...

    def _call_llm(self, prompt: str, schema: dict) -> dict:
        """统一 LLM 调用，自动重试并做结构化校验。"""
```

统一要求：

- 所有 agent 输出必须是结构化对象，不返回裸 LLM 文本
- 所有 LLM 调用必须记录 prompt 摘要、模型名、token 数、延迟、失败原因
- `Retriever Agent` 的 Stage 1 Gating 不允许调用 LLM
- P0 主流程下，agent 默认顺序执行，保证确定性和可复现性

### 5.2 Update Agent

输入：

```python
@dataclass
class UpdateInput:
    user_input: str
    scope: Scope
    timestamp: datetime
    nearby_memories: list[MemoryEntry]
```

输出：

```python
@dataclass
class UpdateProposal:
    intent: Literal["add", "supersede_full", "revoke", "rollback", "ignore"]
    candidate: MemoryEntry | None
    target_hint: TopicQuery | None
    transition_type: Literal["corrected", "preference_shifted", "user_revoked"] | None
```

约束：

- `intent == "ignore"` 时，其他字段必须全为 `None`
- `intent == "add"` 时，`candidate` 必填，`target_hint` 和 `transition_type` 必须为 `None`
- `intent == "supersede_full"` 时，`candidate`、`target_hint`、`transition_type` 必填
- `intent == "revoke"` 时，`candidate is None`，`target_hint` 必填，`transition_type == "user_revoked"`
- `intent == "rollback"` 时，`candidate is None`，`target_hint` 必填，`transition_type is None`

说明：

- P0 中 query-only turn 不调用 `Update Agent`
- 对 LongMemEval replay turn，`Update Agent` 是唯一必调的 LLM agent
- `nearby_memories` 由当前 `scope` 可见且 `state == "active"` 的 memory 中，按 BM25 或词项重叠预取前 `nearby_k` 条组成，`nearby_k` 从配置读取

### 5.3 Conflict Agent

输入：

- `UpdateProposal`
- 当前 `scope` 下与 `target_hint` 最相关的候选 memory
- `ScopeCompat` 与 `transition_type` 规则

输出：

```python
@dataclass
class ConflictResolution:
    action: Literal[
        "apply_add",
        "confirm_supersede",
        "confirm_revoke",
        "downgrade_to_add",
        "request_clarification",
        "reject",
    ]
    target_id: str | None
    state_transition: list[dict]
    supersede_link: SupersedeLink | None
    reason: str | None
```

约束：

- `action == "apply_add"` 时，`target_id is None`
- `action == "confirm_supersede"` 时，`target_id` 与 `supersede_link` 必填
- `action == "confirm_revoke"` 时，`target_id` 必填，`supersede_link is None`
- `action == "downgrade_to_add"` 只允许从 `supersede_full` 降级而来
- `ScopeCompat` 失败时，必须返回 `request_clarification`

决策规则：

- `supersede_full` 且定位到唯一兼容目标时，返回 `confirm_supersede`
- `supersede_full` 且未定位到兼容目标时，返回 `downgrade_to_add`
- `revoke` 且未定位到兼容目标时，返回 `request_clarification`
- 多个候选目标分数并列时，返回 `request_clarification`
- `transition_type` 不合法时，返回 `reject`

### 5.4 Retriever Agent

输入：

```python
@dataclass
class RetrievalInput:
    query: str
    scope: Scope
    timestamp: datetime
    memory_base: MemoryBase
    allow_sensitive_raw: bool = False
```

输出：

```python
@dataclass
class RetrievalResult:
    ranked_memories: list[ScoredMemory]
    injected_constraints: list[MemoryEntry]
    gating_stats: dict[str, int]
    predicted_topic: TopicID | None
```

实现要求：

- Stage 1 Gating 必须使用纯 Python，不调用 LLM
- Stage 2 Scoring 中 `rel` 使用 embedding 相似度；embedding 不可用时回退到 BM25
- Constraint Injection 在 Scoring 之后执行，注入项不参与 `top_k` 截断
- 当 `memory.sensitive == True` 且 `allow_sensitive_raw=False` 时，检索和打分必须使用 `summary_for_retrieval`，不得直接使用原始 `content`

Stage 1 Gating 规则：

1. 只读取 `query_visible(scope, timestamp)` 返回的 memory。
2. 只保留 `state == "active"` 的 memory。
3. 只保留 `valid_time` 覆盖 `timestamp` 的 memory。
4. 用 `topic_tree.match(query)` 预测 `predicted_topic`。
5. 若 `predicted_topic is not None`，优先保留 `topic_id == predicted_topic` 的 memory；若为空集，则回退到词项重叠预筛。

Stage 2 Scoring 规则：

```python
score = lambda_r * rel + lambda_f * fresh + lambda_c * conf
```

其中：

- `rel`：embedding cosine，相似度归一化到 `[0, 1]`
- `fresh`：对于 `preference`，使用 `exp(-age_days / tau_pref)`；对于 `fact` 与 `constraint` 固定为 `1.0`
- `conf`：`memory.conf.combined`

Constraint Injection 规则：

- 从同一候选池中额外选出 `type == "constraint"` 的 memory
- 若 `constraint.topic_id == predicted_topic`，直接注入
- 若 `predicted_topic is None`，则要求 query 与 `constraint.content` 至少有一个去停用词后的关键词重叠
- 注入项从最终回答角度等价于“硬上下文”，不得因 `top_k` 截断被丢弃

### 5.5 Response Agent

输入：

```python
@dataclass
class ResponseInput:
    query: str
    retrieval: RetrievalResult
    allow_sensitive_raw: bool = False
```

输出：

```python
@dataclass
class ResponseOutput:
    answer: str
    cited_memory_ids: list[str]
    context_only_ids: list[str]
    abstained: bool
    abstain_reason: str | None
```

约束：

- `cited_memory_ids` 必须全部来自 `ranked_memories` 或 `injected_constraints`
- 若无足够证据回答，必须 `abstained=True` 并输出澄清问题或拒答说明
- `allow_sensitive_raw=False` 时，只允许使用 `summary_for_retrieval`，不得直接暴露敏感内容

---

## 6. 流程编排需求

### 6.1 Turn 模式

系统支持三种 turn：

- `write_only`：只有 `user_input`，用于 LongMemEval replay turn
- `query_only`：只有 `query`，用于 LongMemEval final question turn
- `mixed`：`user_input` 和 `query` 同时存在，只用于交互式场景

### 6.2 主流程

```python
def process_turn(turn: TurnInput) -> TurnResult:
    errors: list[str] = []
    agents_called: list[str] = []
    write_applied = False
    answer = None
    cited_memory_ids: list[str] = []

    if turn.user_input is not None:
        agents_called.append("update")
        update_out = update_agent.run(UpdateInput(...))

        if update_out.intent == "rollback":
            ok = handle_rollback(update_out.target_hint, turn.scope, turn.timestamp)
            write_applied = ok
        elif update_out.intent != "ignore":
            agents_called.append("conflict")
            conflict_out = conflict_agent.run(...)
            write_applied = apply_state_transition(conflict_out, update_out.candidate)

    if turn.query is not None:
        agents_called.append("retriever")
        retrieval = retriever_agent.run(...)
        agents_called.append("response")
        response = response_agent.run(...)
        answer = response.answer
        cited_memory_ids = response.cited_memory_ids

    return TurnResult(
        answer=answer,
        cited_memory_ids=cited_memory_ids,
        agents_called=agents_called,
        write_applied=write_applied,
        write_only=(turn.user_input is not None and turn.query is None),
        query_only=(turn.user_input is None and turn.query is not None),
        errors=errors,
    )
```

实现要求：

- P0 主流程默认顺序执行
- 任一子步骤失败时必须记录错误，但不得导致整轮日志丢失
- `apply_state_transition()` 只能执行 `ConflictResolution` 明确批准的状态变更

### 6.3 Rollback 处理

```python
def handle_rollback(target_hint: TopicQuery, scope: Scope, now: datetime) -> bool:
    candidates = memory_base.query_revoked_within(
        window_days=config.retention.rollback_window_days,
        scope=scope,
        time=now,
    )
    target = locate_by_hint(candidates, target_hint)
    if target is None:
        return False
    if not is_rollback_legal(target, scope, now):
        return False
    return memory_base.rollback(target.id)
```

`locate_by_hint()` 的排序规则：

1. `topic_id` 完全匹配优先于关键词匹配。
2. 关键词重叠数高者优先。
3. `revoked_at` 更近者优先。
4. 若前 3 项全部并列，返回歧义，要求澄清。

---

## 7. 错误处理与降级

| 错误情况 | 降级行为 |
|---|---|
| LLM API 超时 | 重试最多 3 次，指数退避；仍失败则走该 agent 的 fallback |
| Update Agent JSON 解析失败 | fallback 为 `intent="ignore"`，并记录告警 |
| Conflict Agent JSON 解析失败 | fallback 为 `action="reject"`，跳过本轮写入 |
| Retriever embedding 服务不可用 | fallback 为 BM25 keyword retrieval |
| Response Agent 生成空回答 | fallback 为 `abstained=True` + 通用澄清模板 |
| Response Agent 引用了未检索到的 ID | 后处理剔除非法引用；若剔除后无证据，则改为 abstain |
| MemoryBase 写入冲突 | 重试一次；仍失败则跳过写入并记录错误 |
| `topic_id` 不在受控树中 | 用待写入文本重新执行 `topic_tree.match(...)`；若并列则按字典序选第一项并记录告警 |

要求：

- 所有 fallback 都必须写入 turn log
- 任一 fallback 的发生率都必须能在 `summary.json` 中汇总

---

## 8. 配置需求

### 8.1 `topic_tree.yaml`

`topic_tree.yaml` 必须是仓库内固化的受控叶子树。每个叶子至少包含以下字段：

```yaml
- path: user.preference.food
  description: 用户的食物口味偏好
  default_type: preference
  examples:
    - 我最近喜欢吃日料
    - 我不吃香菜
  keywords:
    - 食物
    - 口味
    - 饮食
```

约束：

- 所有 `path` 全局唯一
- `default_type` 只能取 `fact / preference / constraint`
- `topic_tree.match()` 必须基于 `description + examples + keywords` 做纯 Python 匹配

### 8.2 `scope_compat.yaml`

`scope_compat.yaml` 是本文 §4.1 与 §4.2 的镜像配置，用于单测和运行时加载。语义仍以正文为准。

### 8.3 `default.yaml` 关键超参

```yaml
retrieval:
  top_k: 10
  embedding_model: "text-embedding-3-small"
  lambda_r: 0.5
  lambda_f: 0.3
  lambda_c: 0.2
  tau_pref: 30
  tau_direct: 0.7

update:
  nearby_k: 10

confidence:
  w_p: 0.7
  w_l: 0.3
  prior_table:
    user_explicit: 0.9
    user_implicit: 0.6
    inferred: 0.4
    external: 0.5

retention:
  corrected_archive_days: 90
  revoked_archive_days: 30
  rollback_window_days: 30

llm:
  default_model: "gpt-4o-mini"
  timeout_sec: 30
  max_retries: 3
```

要求：

- 所有阈值和权重必须从 YAML 读取，不得硬编码
- 任何变更都必须能通过 config 覆盖，而不是改业务代码

---

## 9. Instrumentation 与日志

### 9.1 每轮必须记录的字段

```json
{
  "turn_id": "...",
  "timestamp": "...",
  "scope": {"kind": "personal", "ref": null},
  "user_input": "...",
  "query": "...",
  "agents_called": ["update", "conflict", "retriever", "response"],
  "write_applied": true,
  "update_proposal": {...},
  "conflict_resolution": {...},
  "retrieval": {
    "predicted_topic": "user.preference.food",
    "candidate_ids_after_gating": [...],
    "candidate_ids_after_scoring": [...],
    "injected_constraint_ids": [...],
    "gating_stats": {...}
  },
  "response": {
    "answer": "...",
    "cited_memory_ids": [...],
    "abstained": false
  },
  "tokens": {"update": 312, "conflict": 0, "response": 221},
  "latency_ms": {"update": 1240, "conflict": 0, "retriever": 15, "response": 840},
  "fallbacks": [...],
  "errors": [...]
}
```

### 9.2 输出形式

- JSONL，每轮一行
- 单次评测输出到 `runs/<run_id>/turns.jsonl`
- 汇总输出到 `runs/<run_id>/summary.json`

### 9.3 离线汇总要求

`summary.json` 至少包含以下聚合字段：

- `num_turns`
- `num_questions`
- `agent_call_counts`
- `fallback_rates`
- `avg_latency_ms`
- `avg_tokens`
- `write_apply_rate`
- `abstain_rate`

---

## 10. LongMemEval 适配需求

### 10.1 数据流

1. 加载 LongMemEval 数据。
2. 对每个 question instance：
   - 按 timestamp 顺序回放 `haystack_sessions`，每个历史 turn 映射为 `write_only` turn。
   - 历史回放结束后，将 question 映射为 `query_only` turn。
   - 收集 `Response Agent` 的 `answer`。
3. 输出 `<question_id, hypothesis>` JSONL。
4. 用官方评测脚本计算端到端 QA 指标。

### 10.2 输入映射

- LongMemEval replay turn：`TurnInput(user_input=<turn_text>, query=None, scope=personal)`
- LongMemEval final question：`TurnInput(user_input=None, query=<question>, scope=personal)`
- LongMemEval 无 `sensitive` 标记，默认 `False`
- LongMemEval 无 `topic_id` 标注，由 `Update Agent` 写入时分配

### 10.3 输出格式

输出 JSONL 每行至少包含：

```json
{"question_id": "...", "hypothesis": "..."}
```

### 10.4 预算与成本估算

设：

- question 数量约 500
- 平均 replay turns 约 30
- replay turn 总数约 15,000

P0 主线下的调用预算按以下方式估算：

- `Update Agent`：每个 replay turn 1 次，共约 15,000 次
- `Conflict Agent`：只在 `intent != "ignore"` 时调用；若写入率为 30% 到 40%，约 4,500 到 6,000 次
- `Retriever Agent`：默认 0 次 LLM 调用
- `Response Agent`：每个 final question 1 次，共约 500 次

因此，全量评测的 LLM 调用量通常在 20,000 到 21,500 次之间，而不是把每个 replay turn 都按 4 个 agent 计费。

要求：

- 真实成本按运行时 provider 定价计算，不在本文写死
- embedding 结果必须缓存，避免同一文本重复嵌入

---

## 11. 测试需求

### 11.1 单元测试

P0 必须覆盖以下测试点：

- `Scope`、`Confidence`、`TimeRange`、`MemoryEntry` 的序列化与校验
- `MemoryBase` 的 `add / get / update_state / query_visible / rollback`
- `ScopeVisible` 与 `ScopeCompat` 规则
- `type × transition_type` 合法组合表
- `topic_tree.match()` 的命中与回退逻辑
- Retriever Stage 1 Gating 的边界条件

### 11.2 集成测试

至少构造以下场景：

- 事实更正：旧 fact 进入 `superseded`
- 偏好变化：旧 preference 进入 `superseded`，新 memory 的 `transition_type="preference_shifted"`
- 约束冲突：Constraint 不被静默覆盖，而是触发澄清或显式 revoke
- Scope 隔离：`session` 信息不污染 `personal`
- Rollback：30 天窗口内恢复成功，窗口外恢复失败
- Query-only turn：只触发 Retriever + Response，不触发 Update

### 11.3 端到端测试

- LongMemEval 50 样本 smoke run：主流程不崩、输出格式正确、日志字段齐全
- LongMemEval 500 样本 full run：全量可运行、可重复生成官方评测输入

---

## 12. 验收标准

### 12.1 P0 工程验收

- [ ] 50 样本 smoke run 完整跑通
- [ ] 500 样本 full run 完整跑通
- [ ] 输出 JSONL 可被官方评测脚本读取
- [ ] 单元测试通过率 100%，P0 模块覆盖率不低于 70%
- [ ] `turns.jsonl` 与 `summary.json` 字段完整，满足 §9 要求
- [ ] replay turn 不触发 Retriever / Response
- [ ] final question turn 不触发 Update

### 12.2 P1 扩展验收

- [ ] `ScopeVisible` 与 `ScopeCompat` 在集成测试中验证通过
- [ ] rollback 在集成测试中验证通过
- [ ] `sensitive` 字段与 `summary_for_retrieval` 在集成测试中验证通过

### 12.3 研究目标

以下条目是实验目标，不是当前实现 blocker：

- LongMemEval 端到端准确率不低于 Flat RAG baseline
- Knowledge Update 子任务相对 Flat RAG 有明显提升
- 与 Flat RAG、Single-Agent Versioned 等基线形成可复现实验对比

---

## 13. 里程碑建议

| 周次 | 交付物 |
|---|---|
| W1 | `core` 完成，单元测试通过 |
| W2 | Update + Conflict 完成，写入链路跑通 |
| W3 | Retriever + Response 完成，query-only 链路跑通 |
| W4 | LongMemEval 50 样本 smoke run 跑通 |
| W5 | LongMemEval 500 样本 full run 跑通 |
| W6 | P1 扩展测试与研究基线对比 |

---

## 14. 已知风险与缓解

| 风险 | 缓解 |
|---|---|
| LLM 结构化输出不稳定 | 强制 schema 校验，失败时走显式 fallback |
| LongMemEval 全量耗时较长 | 先用 50 样本子集迭代，再做 500 样本全量 |
| replay turn 调用量高 | 只让 Update 必调，Conflict 按需触发，Retriever 不走 LLM |
| topic 匹配误差影响写入质量 | 固化 topic 树、缓存匹配结果、对回退命中做告警统计 |
| scope 扩展规则与 LongMemEval 主线脱节 | 将 scope 能力放到 P1 扩展验收，不阻塞 P0 |

---

## 附：与实现模块的映射

| 规格部分 | 主要实现位置 |
|---|---|
| 数据结构 | `core/schema.py` |
| 可见性与兼容性规则 | `core/scope_compat.py` |
| 版本化写入 | `core/memory_base.py` + `pipeline/transitions.py` |
| Agent 协议与提示词 | `agents/*` + `llm/*` |
| 主流程编排 | `pipeline/orchestrator.py` |
| LongMemEval 适配 | `eval/longmemeval/*` |
| 日志与离线指标 | `eval/instrumentation.py` + `eval/longmemeval/metrics.py` |
