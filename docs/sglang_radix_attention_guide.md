# SGLang RadixAttention & Radix Cache: Complete Guide

**Author's Note**: This document synthesizes insights from the SGLang codebase, the [LMSYS blog on SGLang/RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/), the [SGLang paper (arXiv:2312.07104)](https://arxiv.org/pdf/2312.07104), and the [vLLM automatic prefix caching design doc](https://docs.vllm.ai/en/v0.9.0/design/automatic_prefix_caching.html).

---

## 1) CEO Explanation (1–2 pages)

### What's Happening: The Story of LLM Inference

Imagine you have a team of brilliant analysts who can answer any question about your company's documents. Each time you ask a question, they must:

1. **Read the entire document** (expensive, slow) - this is called "prefill"
2. **Think and answer one word at a time** (faster per word, but sequential) - this is "decode"

Now here's the key insight: **many questions start the same way**. If 100 customers all ask about your return policy, the analysts must read the same "Return Policy" section 100 times. That's wasteful.

### What is KV Cache?

When the model reads text, it creates "notes" about what it read - these are Key-Value (KV) pairs. These notes are **expensive to create** (require GPU compute) but **cheap to store** (just memory). The KV cache stores these notes so the model doesn't have to re-read the same text.

**Business Impact**:
- Without caching: Every request pays full compute cost for reading
- With caching: Shared "readings" computed once, reused many times

### The Radix Tree: Your Company's Filing System

Think of how you'd organize documents in a filing cabinet:

```
ROOT
├── "Dear Customer,"
│   ├── "Thank you for your inquiry about..."
│   │   ├── "...our return policy" → [Answer A]
│   │   └── "...shipping times" → [Answer B]
│   └── "We regret to inform you..."
│       └── "...your order was cancelled" → [Answer C]
└── "Hi Team,"
    └── "Please review the attached..." → [Answer D]
```

Instead of storing each complete message separately, you **share the common beginnings**. SGLang's radix tree does exactly this for LLM "notes" (KV cache).

**Tiny Example**:
- Request 1: "What is your return policy for electronics?"
- Request 2: "What is your return policy for clothing?"
- Request 3: "What is your shipping policy?"

```
Root
└── "What is your "
    ├── "return policy for "
    │   ├── "electronics?" → [Cached KV for this path]
    │   └── "clothing?" → [Cached KV for this path]
    └── "shipping policy?" → [Cached KV for this path]
```

Request 2 **reuses 100% of the work** from the shared prefix with Request 1. Request 3 reuses the "What is your " portion.

### Business Terms: What This Means for Cost & Speed

| Metric | Without RadixAttention | With RadixAttention | Impact |
|--------|----------------------|---------------------|---------|
| **Latency (TTFT)** | Full prefill every time | Skip cached prefixes | **Up to 5x faster first token** |
| **Throughput** | Limited by GPU compute | Compute only new tokens | **3-5x more requests/second** |
| **Cost** | Pay for all compute | Pay only for new work | **Significant GPU cost reduction** |

*Source: SGLang blog reports "up to 5x faster" on workloads with shared prefixes.*

### When to Choose SGLang vs. vLLM

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| **Multi-turn chat with context** | SGLang | Radix tree excels at progressive prefix matching |
| **Agent/tool-calling workflows** | SGLang | System prompts + tool definitions heavily reused |
| **Few-shot prompting at scale** | SGLang | Same examples repeated = massive reuse |
| **Batch processing, diverse prompts** | Either | Less prefix sharing = similar performance |
| **Very short prompts (<100 tokens)** | Either | Caching overhead may exceed benefit |
| **Need APC block-hash eviction** | vLLM | Simpler eviction semantics for some workloads |
| **Multi-LoRA production** | Both (with caveats) | Both support LoRA isolation in cache |

**CEO Decision Framework**:
1. **Do your requests share prefixes?** (system prompts, chat history, examples) → SGLang
2. **Is latency critical?** (interactive chat, agents) → SGLang's radix matching is faster
3. **Is your workload highly variable?** (random prompts) → Either works; benchmark your specific case

---

## 2) Intern Onboarding Explanation (Deep but Friendly)

### Step-by-Step: How LLM Inference Works

#### Phase 1: Prefill
When a user sends a prompt like "Explain quantum computing in simple terms", the model must:

1. **Tokenize**: Convert text to token IDs (e.g., `[1, 2389, 17823, 942, ...]`)
2. **Process all tokens at once**: Run the full transformer for every token
3. **Generate KV cache**: For each layer, create Key and Value tensors

```
For a Llama-70B model with 80 layers, 8 KV heads, 128 head_dim:
- Per token: 80 layers × 2 (K+V) × 8 heads × 128 dims × 2 bytes = ~328KB
- For 4K context: 4096 × 328KB ≈ 1.3GB of KV cache!
```

**This is expensive**. Prefill is compute-bound and often takes 50-90% of total latency.

#### Phase 2: Decode
After prefill, the model generates tokens one at a time:
1. Use the cached KV from prefill (no recomputation!)
2. Only process the *new* token through all layers
3. Append new K, V to the cache
4. Repeat until done

**Decode is memory-bound** - the bottleneck is reading the KV cache, not computing.

### What is Stored in KV Cache?

For each transformer layer `i` and each token position `t`:

```python
# Conceptually:
K[layer_i][token_t] = Linear_K(hidden_state[t])  # Query-able "what I represent"
V[layer_i][token_t] = Linear_V(hidden_state[t])  # "what to return if queried"
```

The KV cache lets the model "remember" previous tokens without recomputing them.

**Why it's expensive to recompute**:
- Recomputing means running the token through all layers again
- For a 70B model, that's ~70 billion multiply-adds PER TOKEN
- Caching trades memory (cheap) for compute (expensive)

### What "Shared Prefix" Means in Real Workloads

#### Example 1: Multi-turn Chat
```
Turn 1: "You are a helpful assistant. User: What is Python? Assistant: Python is..."
Turn 2: "You are a helpful assistant. User: What is Python? Assistant: Python is... User: How do I install it?"
Turn 3: "You are a helpful assistant. User: What is Python? Assistant: Python is... User: How do I install it? Assistant: To install Python... User: What about pip?"
```

Each turn shares the entire previous conversation as prefix!

#### Example 2: Agent/Tool Calling
```
System: You are an agent with access to: [calculator, search, code_exec]. 
        Rules: 1. Think step by step... 2. Always verify... [500 tokens of instructions]
        
User: What is 15% of 340?
```

Every agent request shares the massive system prompt.

#### Example 3: Few-shot Prompting
```
Examples:
Q: What is the capital of France? A: Paris
Q: What is the capital of Germany? A: Berlin
Q: What is the capital of Italy? A: Rome

Now answer:
Q: What is the capital of Spain?
```

Same examples repeated across thousands of classification requests.

### ASCII Diagram (a): Radix Tree of Token Prefixes

```
                            ┌─────────┐
                            │  ROOT   │
                            │ (empty) │
                            └────┬────┘
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              ┌─────────┐ ┌─────────┐ ┌─────────┐
              │ "Hello" │ │ "What"  │ │ "The"   │
              │ KV[0:5] │ │ KV[0:4] │ │ KV[0:3] │
              └────┬────┘ └────┬────┘ └─────────┘
                   │           │
          ┌────────┴────┐      │
          ▼             ▼      ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ ", how"  │ │ ", world"│ │ " is the"│
    │ KV[5:10] │ │ KV[5:12] │ │ KV[4:11] │
    └────┬─────┘ └──────────┘ └────┬─────┘
         │                         │
         ▼                         ▼
   ┌───────────┐             ┌───────────┐
   │ " are you"│             │ " weather"│
   │ KV[10:18] │             │ KV[11:18] │
   └───────────┘             └───────────┘

Legend:
- Each node stores: key tokens + pointers to KV cache indices
- KV[a:b] means "KV cache for tokens a through b"
- Shared prefixes = shared nodes = no recomputation
```

### ASCII Diagram (b): KV Cache Reuse Flow

```
Existing Cache State:
═══════════════════════════════════════════════════════════════
RADIX TREE                          KV CACHE (GPU Memory)
                                    ┌─────────────────────────┐
     ROOT                           │ Slot 0: [K0,V0] token 0 │
      │                             │ Slot 1: [K1,V1] token 1 │
      ▼                             │ Slot 2: [K2,V2] token 2 │
 "What is the"                      │ Slot 3: [K3,V3] token 3 │
  │ value=[0,1,2,3]                 │ Slot 4: [K4,V4] token 4 │
  │                                 │ Slot 5: [K5,V5] token 5 │
  └─► " capital of France?"         │ Slot 6: (empty)         │
       value=[4,5,6,7,8,9]          │ ...                     │
                                    └─────────────────────────┘

New Request: "What is the capital of Spain?"
═══════════════════════════════════════════════════════════════

Step 1: Match Prefix
─────────────────────
match_prefix("What is the capital of Spain?")
  → Traverse tree: ROOT → "What is the" → " capital of" (partial)
  → Match length: 6 tokens ("What is the capital of")
  → Return: indices=[0,1,2,3,4,5], last_node

Step 2: Skip Prefill for Matched Tokens  
─────────────────────
- DO NOT recompute KV for "What is the capital of"
- These 6 tokens' KV already in slots [0-5]

Step 3: Prefill Only New Tokens
─────────────────────
- Only compute: " Spain?" (2 tokens)
- Allocate new slots: [10, 11]
- Run prefill ONLY for 2 tokens (not 8!)

Step 4: Insert New Branch
─────────────────────
     ROOT
      │
      ▼
 "What is the"
  │ value=[0,1,2,3]
  │
  └─► " capital of"
       │ value=[4,5]
       │
       ├─► " France?" ←── (existing)
       │    value=[6,7,8,9]
       │
       └─► " Spain?"  ←── (NEW!)
            value=[10,11]

SAVINGS: 6 tokens skipped = 75% less prefill compute!
```

### ASCII Diagram (c): Block-Based (vLLM) vs. Token/Prefix Tree (SGLang)

```
═══════════════════════════════════════════════════════════════
              vLLM: Block-Based Prefix Caching
═══════════════════════════════════════════════════════════════

Tokens:    [ T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | ... ]
           └──── Block 0 ────┘ └──── Block 1 ────┘ └── Block 2

Block Hash Table:
┌──────────────────┬────────────────────────────────────┐
│ hash(T0,T1,T2,T3)│ → KV Block 0 pointer               │
│ hash(T4,T5,T6,T7)│ → KV Block 1 pointer               │
│ hash(T0,T1,T2,T3,│                                    │
│      T4,T5,T6,T7)│ → (uses prefix-aware hash chain)   │
└──────────────────┴────────────────────────────────────┘

Matching: 
- Hash-based lookup at BLOCK granularity
- Block size typically 16 or 32 tokens
- If prompt = 100 tokens with block_size=16:
  - Check hash(block0), hash(block0+block1), ...
  - Must match COMPLETE blocks only

═══════════════════════════════════════════════════════════════
           SGLang: Token-Level Radix Tree
═══════════════════════════════════════════════════════════════

Tokens:    [ T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | ... ]

Radix Tree (can split at ANY token boundary):
           ROOT
            │
            ├─► [T0,T1,T2] ──► [T3,T4] ──► [T5] ──► [T6,T7,T8]
            │                    │
            │                    └─► [T3',T4',T5']  (different suffix)
            │
            └─► [T0',T1'] ──► ...

Matching:
- Tree traversal at TOKEN granularity (or configurable page_size)
- Finds longest matching prefix via tree walk
- Can match partial blocks (no hash boundary constraints)
- Splits nodes when match ends mid-segment

═══════════════════════════════════════════════════════════════
                    Key Differences
═══════════════════════════════════════════════════════════════

                        vLLM (Block)       SGLang (Radix)
Granularity:            Block (16-32)      Token (or page_size)
Lookup:                 Hash table O(1)    Tree walk O(prefix_len)
Partial Match:          No (block-aligned) Yes (any position)
Memory Overhead:        Lower (fewer keys) Higher (tree nodes)
Best for:               Long exact blocks  Variable-length prefixes
Eviction:               Block-level LRU    Leaf-level LRU
Cache-aware scheduling: Supported          Native (LPM policy)
```

---

## 3) Engineering Deep Dive

### Matching Granularity: Token-Level vs. Block-Hash

**SGLang Radix Cache** (from `python/sglang/srt/mem_cache/radix_cache.py`):

```python
# Key matching at token level (page_size=1) or page level
def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i

def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    # Match at page granularity for memory efficiency
    i = 0
    while i < min_len:
        if key0.token_ids[i:i+page_size] != key1.token_ids[i:i+page_size]:
            break
        i += page_size
    return i
```

**vLLM Automatic Prefix Caching** (from vLLM design doc):
- Uses content-based hashing: `block_hash = hash(parent_block_hash, tokens_in_block)`
- Fixed block boundaries (typically 16 or 32 tokens)
- Hash chain ensures position-aware matching

**Trade-offs**:

| Aspect | SGLang Radix Tree | vLLM Block Hash |
|--------|------------------|-----------------|
| **Match precision** | Token-exact | Block-aligned |
| **Lookup cost** | O(prefix_len) tree walk | O(1) hash + O(blocks) chain |
| **Memory overhead** | Tree node per segment | Hash table + block metadata |
| **Partial match** | Natural (splits nodes) | No (wastes partial blocks) |
| **CPU overhead** | Higher (tree ops) | Lower (hash ops) |

### Insert / Lookup / Eviction

**Insert** (`radix_cache.py:_insert_helper`):
```python
def _insert_helper(self, node: TreeNode, key: RadixKey, value, priority: int = 0):
    while len(key) > 0 and child_key in node.children.keys():
        node = node.children[child_key]
        prefix_len = self.key_match_fn(node.key, key)
        
        if prefix_len < len(node.key):
            # Split node at mismatch point
            new_node = self._split_node(node.key, node, prefix_len)
            
        key = key[prefix_len:]
        value = value[prefix_len:]
        
    if len(key):
        # Create new leaf node
        new_node = TreeNode(priority=priority)
        new_node.key = key
        new_node.value = value  # KV cache indices
        node.children[child_key] = new_node
        self.evictable_size_ += len(key)
```

**Lookup** (`match_prefix`):
```python
def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
    # Returns: (matched_kv_indices, last_matched_node)
    # O(prefix_length) traversal down the tree
    value, last_node = self._match_prefix_helper(self.root_node, key)
    return MatchResult(device_indices=torch.cat(value), last_device_node=last_node)
```

**Eviction** (`evict_policy.py` + `radix_cache.py:evict`):

SGLang supports multiple eviction strategies:

```python
class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time  # Evict oldest first

class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)

class PriorityStrategy(EvictionStrategy):
    # Priority-aware: respects request priority levels
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.priority, node.last_access_time)
```

Eviction process:
1. Collect all **leaf nodes** with `lock_ref == 0`
2. Build min-heap based on eviction strategy priority
3. Pop leaves, free their KV cache slots, remove from tree
4. If parent becomes childless and unlocked, add to heap

**Key insight**: Eviction is **leaf-only**. Internal nodes represent shared prefixes and cannot be evicted while children exist.

### Memory Layout: Paged KV with Radix

From `memory_pool.py`:

```python
class MHATokenToKVPool(KVCache):
    def __init__(self, size, page_size, ...):
        # [num_tokens, num_heads, head_dim] per layer
        self.k_buffer = [torch.zeros((size + page_size, head_num, head_dim), ...) 
                        for _ in range(layer_num)]
        self.v_buffer = [torch.zeros((size + page_size, head_num, head_dim), ...) 
                        for _ in range(layer_num)]
```

The radix tree stores **indices into these pools**:
```
TreeNode.value = torch.Tensor([5, 6, 7, 8])  # Indices into k_buffer/v_buffer
```

**Page size considerations** (from `server_args.py`):
- `page_size=1`: Maximum cache hit granularity (token-level)
- `page_size=16/64`: Better memory efficiency, less tree overhead
- Trade-off: larger pages = coarser matching = potential wasted compute

### Scheduler Interactions

**Cache-Aware Scheduling** (`schedule_policy.py`):

```python
class CacheAwarePolicy(Enum):
    LPM = "lpm"      # Longest Prefix Match - prioritize requests with most cache hits
    DFS_WEIGHT = "dfs-weight"  # DFS-based weighting on tree

class SchedulePolicy:
    def calc_priority(self, waiting_queue: List[Req]):
        if self.policy == CacheAwarePolicy.LPM:
            # For each request, compute match_prefix()
            for r in waiting_queue:
                match_result = self.tree_cache.match_prefix(...)
                r.prefix_indices = match_result.device_indices
            # Sort by longest prefix (most reuse = schedule first)
            waiting_queue.sort(key=lambda r: -len(r.prefix_indices))
```

**In-batch prefix caching** (from `schedule_policy.py`):
```python
# If multiple waiting requests share a prefix not yet cached,
# only schedule one to avoid redundant prefill
if len(in_batch_matching_prefixes) >= DEPRIORITIZE_THRESHOLD:
    temporary_deprioritized.add(r.rid)
```

**Continuous batching interaction**:
- Requests join/leave batches dynamically
- Lock refs prevent eviction of in-use prefixes
- `inc_lock_ref` / `dec_lock_ref` manage reference counting

### Strengths / Weaknesses

#### Best-Case Speedups
- **Multi-turn chat**: ~5x speedup on long conversations (per SGLang blog)
- **Few-shot prompting**: Nearly linear speedup with example count
- **Agent systems**: System prompts cached across all tool calls

#### Worst-Case Overheads
- **Random prompts**: Tree traversal overhead with no reuse
- **Short prompts**: Caching overhead exceeds compute savings
- **High churn**: Frequent eviction/insertion degrades tree quality

#### Fragmentation / Churn Scenarios
```
Symptom: Cache hit rate drops despite high memory usage
Cause: Many small, disjoint prefixes creating tree fragmentation
Fix: 
  - Increase page_size to coarsen granularity
  - Use priority-based eviction for important prefixes
  - Consider workload partitioning
```

#### Multi-Tenant Pitfalls
- **Problem**: Tenant A's cache evicts Tenant B's hot prefixes
- **Solutions**:
  - Per-tenant cache isolation via `extra_key` (see `RadixKey.extra_key`)
  - Priority-based eviction (`--enable-priority-scheduling`)
  - Separate server instances

#### LoRA Compatibility
From `radix_cache.py`:
```python
class RadixKey:
    def __init__(self, token_ids, extra_key=None, ...):
        self.extra_key = extra_key  # Can include LoRA ID
```
- Different LoRA adapters get separate cache namespaces via `extra_key`
- Prevents cross-contamination between adapter-specific KV caches

### Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Low cache hit rate | Short/diverse prompts | Redesign prompts for shared prefixes |
| OOM during prefill | Cache not evicting fast enough | Reduce `--mem-fraction-static`, increase eviction |
| High TTFT variance | Cache thrashing | Increase memory, use priority eviction |
| Wrong outputs with LoRA | Missing `extra_key` isolation | Ensure LoRA ID in cache key |
| Slow scheduling | Large waiting queue + LPM | Falls back to FCFS at 128+ queue size |

---

## 4) Practical Playbook: "How to Get Wins Fast"

### Checklist for Maximizing Cache Hit Rate

**Prompt Design**:
- [ ] Put shared content (system prompts, examples) at the **beginning**
- [ ] Use consistent formatting across requests (same whitespace, punctuation)
- [ ] For chat: include full conversation history (not just last turn)
- [ ] For agents: standardize tool descriptions, put them first

**Serving Patterns**:
- [ ] Batch similar requests together (same system prompt)
- [ ] Use sticky sessions for multi-turn chat (same prefix on same server)
- [ ] Pre-warm cache with common prefixes at startup

**Configuration**:
- [ ] Set appropriate `page_size` (1 for max reuse, 16-64 for efficiency)
- [ ] Enable cache-aware scheduling (`--schedule-policy lpm`)
- [ ] Tune memory allocation (`--mem-fraction-static`)

### Metrics to Instrument

```python
# Key metrics to track:
metrics = {
    # Cache performance
    "cache_hit_rate": "matched_tokens / total_input_tokens",
    "cache_hit_tokens": "total tokens skipped via cache",
    "eviction_rate": "tokens_evicted / time",
    "tree_size": "total_cached_tokens",
    
    # Latency
    "ttft_p50": "time to first token (median)",
    "ttft_p99": "time to first token (99th percentile)",
    "prefill_time": "time spent in prefill phase",
    
    # Throughput
    "tps": "tokens per second (output)",
    "requests_per_second": "completed requests / time",
    
    # Memory
    "kv_cache_utilization": "used_kv_slots / total_kv_slots",
    "evictable_size": "tokens available for eviction",
    "protected_size": "tokens locked by active requests",
}
```

SGLang exposes these via Prometheus metrics (`RadixCacheMetricsCollector`).

### Benchmark Plan

#### Microbenchmarks

**1. Prefix Match Latency**
```bash
# Measure tree traversal time for various prefix lengths
python -c "
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
import time

cache = RadixCache.create_simulated()
# Insert a long prefix
tokens = list(range(10000))
cache.insert(RadixKey(tokens))

# Benchmark match
for prefix_len in [100, 1000, 5000, 10000]:
    start = time.perf_counter()
    for _ in range(1000):
        cache.match_prefix(RadixKey(tokens[:prefix_len]))
    elapsed = (time.perf_counter() - start) / 1000
    print(f'Prefix {prefix_len}: {elapsed*1e6:.2f} µs')
"
```

**2. Eviction Throughput**
```bash
# Measure eviction speed under memory pressure
# Track: evictions/sec, tokens freed, tree rebalancing overhead
```

**3. Cache Hit Rate Sensitivity**
```bash
# Vary page_size, measure hit rate on fixed workload
for page_size in 1 4 16 64; do
    python benchmark.py --page-size $page_size --measure-hit-rate
done
```

#### Realistic Workloads

**1. Multi-Turn Chat**
```bash
# ShareGPT dataset with conversation threading
python -m sglang.bench_serving \
    --backend sglang \
    --dataset-name sharegpt \
    --num-prompts 500 \
    --request-rate 10
```

**2. Agent/Tool Loops**
```bash
# Simulated ReAct-style agent workload
# - Fixed system prompt (1000+ tokens)
# - Variable user queries
# - Multiple tool call rounds per request
python benchmark_agent.py \
    --system-prompt-length 1500 \
    --avg-tool-rounds 5 \
    --concurrent-agents 20
```

### Tuning Knobs

| Parameter | What It Does | Recommended |
|-----------|--------------|-------------|
| `--page-size` | Granularity of cache matching | 1 (max hit rate) or 16 (balanced) |
| `--schedule-policy` | Request scheduling strategy | `lpm` for cache-aware |
| `--mem-fraction-static` | GPU memory for KV cache | 0.8-0.9 |
| `--disable-radix-cache` | Turn off caching entirely | Only for debugging |
| `--eviction-policy` | LRU, LFU, priority | `lru` (default) |
| `--chunked-prefill-size` | Max tokens per prefill chunk | 8192 (balance latency/throughput) |

---

## 5) Minimal "Most-Optimized" Starter Snippets

### SGLang Server with Radix Cache (Enabled by Default)

```bash
# Basic launch - radix cache is ON by default
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000

# Optimized for multi-turn chat / agents
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --mem-fraction-static 0.85 \
    --schedule-policy lpm \
    --page-size 1 \
    --chunked-prefill-size 8192

# With hierarchical cache (CPU offload) for large context
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --hicache-write-policy write_through

# Disable cache (for comparison benchmarks)
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --disable-radix-cache
```

### vLLM Server with Prefix Caching

```bash
# Enable automatic prefix caching
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --enable-prefix-caching

# With tuned block size
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --enable-prefix-caching \
    --block-size 16 \
    --gpu-memory-utilization 0.85
```

### Request Flow Pseudo-Code

```python
# === SGLang Internal Request Flow ===

def process_request(prompt: str):
    # 1. Tokenize
    token_ids = tokenizer.encode(prompt)
    
    # 2. Match prefix in radix cache
    radix_key = RadixKey(token_ids=token_ids, extra_key=lora_id)
    match_result = radix_cache.match_prefix(radix_key)
    
    cached_kv_indices = match_result.device_indices  # Already computed KV
    last_node = match_result.last_device_node
    prefix_len = len(cached_kv_indices)
    
    # 3. Lock matched nodes (prevent eviction during use)
    radix_cache.inc_lock_ref(last_node)
    
    # 4. Prefill ONLY new tokens
    new_token_ids = token_ids[prefix_len:]
    if len(new_token_ids) > 0:
        # Allocate KV slots for new tokens
        new_kv_indices = kv_allocator.alloc(len(new_token_ids))
        
        # Run transformer prefill (only for new tokens!)
        new_kv_cache = model.prefill(
            input_ids=new_token_ids,
            past_kv_indices=cached_kv_indices  # Attend to cached KV
        )
        
        # Store new KV in allocated slots
        kv_pool.store(new_kv_indices, new_kv_cache)
    
    # 5. Decode loop
    full_kv_indices = torch.cat([cached_kv_indices, new_kv_indices])
    while not done:
        next_token = model.decode_one(full_kv_indices)
        # Extend KV cache with new token...
        yield next_token
    
    # 6. Insert full sequence into cache
    radix_cache.insert(radix_key, full_kv_indices)
    
    # 7. Release lock
    radix_cache.dec_lock_ref(last_node)
```

---

## What I Would Do Next Week (Engineer Onboarding Plan)

### Day 1-2: Understand the Basics
- [ ] Read this document thoroughly
- [ ] Run SGLang server locally with a small model (Llama-3-8B)
- [ ] Send 10 requests with shared prefixes, observe logs
- [ ] Toggle `--disable-radix-cache` and compare latency

### Day 3: Explore the Code
- [ ] Read `radix_cache.py` - understand TreeNode, insert, match_prefix
- [ ] Read `schedule_policy.py` - understand LPM scheduling
- [ ] Run the radix cache unit tests: `pytest test/registered/attention/test_radix_cache_unit.py`

### Day 4: Benchmark
- [ ] Run `bench_serving.py` with ShareGPT dataset
- [ ] Compare SGLang vs vLLM on same workload
- [ ] Identify cache hit rate from metrics

### Day 5: Optimize a Real Workload
- [ ] Pick a production-like workload (chat, agents, or RAG)
- [ ] Profile with different `page_size` values
- [ ] Tune `--schedule-policy` and `--chunked-prefill-size`
- [ ] Document findings and share with team

### Week 2: Deep Dive
- [ ] Explore HiCache for hierarchical caching
- [ ] Understand multi-LoRA cache isolation
- [ ] Study `schedule_batch.py` for continuous batching interaction
- [ ] Contribute a small fix or doc improvement to SGLang

---

## References

1. **SGLang Paper**: [Efficiently Programming Large Language Models using SGLang](https://arxiv.org/pdf/2312.07104) (arXiv:2312.07104)
2. **LMSYS Blog**: [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/)
3. **vLLM APC Design Doc**: [Automatic Prefix Caching](https://docs.vllm.ai/en/v0.9.0/design/automatic_prefix_caching.html)
4. **SGLang GitHub**: [sgl-project/sglang](https://github.com/sgl-project/sglang)
5. **HiCache Blog**: [HiCache: Hierarchical KV Cache for LLM Inference](https://lmsys.org/blog/2025-09-10-sglang-hicache/)
