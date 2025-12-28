# SGLang Radix Cache: Why It Exists and How It Works

**A code-grounded deep dive for CEOs deciding on adoption and engineers onboarding in week one.**

---

## Core Questions Answered First

### 1) What inference problem does SGLang solve that vLLM/TensorRT do not?

**The Repeated Prefill Problem.**

When you serve LLMs, the "prefill" phase (processing all input tokens) is the compute bottleneck. Here's the waste:

```
Request 1: "You are a helpful assistant. User: What is Python?"
Request 2: "You are a helpful assistant. User: What is Java?"
Request 3: "You are a helpful assistant. User: What is Rust?"
```

**Without SGLang**: Each request recomputes the KV cache for "You are a helpful assistant. User: What is " — that's 15+ tokens × 70+ billion parameters × 3 times.

**With SGLang**: Compute once, reuse the KV cache for the shared prefix.

vLLM's paging helps with **memory efficiency** (avoiding fragmentation), but it does NOT eliminate **repeated compute** for shared prefixes. TensorRT optimizes kernel speed but doesn't know your requests share content.

SGLang's radix cache stores KV at **token granularity** and matches **any partial prefix**, not just exact sequences.

### 2) Why are shared prompts and agent-style workloads fundamentally different?

Traditional serving assumes: "Each request is independent."

Agent/chat workloads violate this assumption:

| Workload | Prefix Sharing Pattern |
|----------|----------------------|
| **Multi-turn chat** | Each turn shares entire conversation history |
| **Agents/ReAct** | Every tool call shares system prompt + tool definitions (often 1000+ tokens) |
| **Few-shot learning** | Same examples repeated across thousands of requests |
| **RAG** | Same documents retrieved for similar queries |

For a chatbot with 10-turn conversations, turn 10 shares **90% of tokens** with turn 9. Without radix caching, you recompute 90% of prefill every turn.

### 3) Why is recomputing prefill wasteful, and why does paging alone not solve it?

**Prefill is compute-bound**:
```
Cost = num_tokens × num_layers × (attention_cost + mlp_cost)
     = 4096 tokens × 80 layers × O(hidden_dim²)
     ≈ 70 billion FLOPs for a 70B model
```

**Paging (vLLM) solves memory, not compute**:
- Paging breaks KV cache into blocks to avoid fragmentation
- Each request still computes its own KV — paging just stores it efficiently
- Two requests with identical prefixes still both compute prefill

**Radix caching solves compute**:
- First request computes KV for prefix, stores indices in tree
- Second request **looks up** the prefix, **skips prefill** for matched tokens
- Only computes KV for the suffix (new tokens)

```
         Paging                    Radix Cache
         ------                    -----------
Request 1: Compute [ABCDE]         Request 1: Compute [ABCDE] → store
Request 2: Compute [ABCFG]         Request 2: Match [ABC] → skip
           └─ waste!                         Compute [FG] only → 60% savings
```

### 4) What is a radix-tree KV cache, and why is token-level prefix reuse powerful?

A **radix tree** (also called trie or prefix tree) stores strings as paths from root to leaf, where shared prefixes share nodes.

**Applied to KV caching**:
- Each path from root = a sequence of tokens
- Each node stores = pointers to KV cache memory slots
- Shared prefixes = shared nodes = reused KV compute

**Why token-level matters**:
```
"The capital of France is Paris"
"The capital of Germany is Berlin"
```

- **Sequence-level**: Only exact matches. No reuse here.
- **Block-level (vLLM APC)**: Match "The capital of" only if it aligns to block boundaries
- **Token-level (SGLang)**: Match exactly "The capital of " (4 tokens) regardless of alignment

Token-level finds the **maximum possible reuse** for any workload.

### 5) When does SGLang win vs vLLM, and when does it lose?

| Scenario | Winner | Why |
|----------|--------|-----|
| Multi-turn chat (long history) | **SGLang** | Progressive prefix growth = massive reuse |
| Agent systems | **SGLang** | Fixed system prompt + tool defs = 1000+ tokens shared |
| Few-shot prompting | **SGLang** | Same examples = near-perfect reuse |
| Batch inference (diverse) | **Tie** | No shared prefixes = no advantage |
| Very short prompts (<50 tokens) | **vLLM** | Tree overhead exceeds compute savings |
| High request churn, low overlap | **vLLM** | Simpler eviction, less memory overhead |
| Need block-level isolation | **vLLM** | APC's hash-based approach is simpler |

**SGLang wins when prefixes are shared. vLLM wins when they're not.**

### 6) What trade-offs does radix caching introduce?

| Trade-off | Impact | Mitigation |
|-----------|--------|------------|
| **Memory overhead** | Tree nodes + metadata per cached segment | Use larger `page_size` (16+) |
| **CPU bookkeeping** | Tree traversal on every request | Falls back to FCFS at 128+ queue |
| **Scheduling constraints** | Cache-aware scheduling adds complexity | LPM policy handles this |
| **Eviction complexity** | Leaf-only eviction, no block-level control | LRU/LFU/Priority strategies |
| **Correctness risk** | LoRA/adapter isolation required | `extra_key` namespace separation |

---

## A) WHY SGLang Exists (CEO-Level, Concrete)

### The Problem: Repeated Work

Every LLM request has two phases:

1. **Prefill**: Read the entire prompt, generate "notes" (KV cache) — **expensive**
2. **Decode**: Generate tokens one by one using the notes — **fast per token**

The insight: **Most production workloads repeat the same beginnings**.

| Workload | Repeated Prefix | Waste Without Caching |
|----------|----------------|----------------------|
| Customer support bot | "You are a helpful customer service agent for Acme Corp..." | 500 tokens × every request |
| Code assistant | "You have access to: `search()`, `edit()`, `run()`..." | 800 tokens × every tool call |
| Document QA | Same documents embedded in context | 4000+ tokens per user session |

### What Radix Caching Changes

| Metric | Without Radix | With Radix | Improvement |
|--------|--------------|------------|-------------|
| **Time to First Token** | Full prefill | Skip cached prefix | **Up to 5x faster** |
| **GPU Utilization** | Wasted on repeated compute | Only new work | **3-5x more throughput** |
| **Cost per request** | Pay for all tokens | Pay for unique tokens | **Significant reduction** |

*Source: [SGLang blog](https://lmsys.org/blog/2024-01-17-sglang/) reports "up to 5x faster" on workloads with shared prefixes.*

### Decision Table: When to Use What

| Your Situation | Recommendation |
|---------------|----------------|
| Building a chatbot with multi-turn context | **SGLang** |
| Building an agent with tool calling | **SGLang** |
| Batch processing diverse, unrelated prompts | **Either** (benchmark) |
| Need simplest possible deployment | **vLLM** |
| Already using vLLM, want to add prefix reuse | **vLLM APC** or **migrate to SGLang** |
| Maximum latency optimization, shared prefixes | **SGLang** |
| Need to customize caching logic deeply | **SGLang** (more modular code) |

---

## B) Foundational Concepts (Intern-Friendly)

### Prefill vs Decode: The 30-Second Version

```
INPUT:  "What is the capital of France?"
         ↓
┌────────────────────────────────────────────────────────────┐
│ PREFILL (compute-bound, parallel across tokens)            │
│ - Process ALL input tokens at once through the model       │
│ - Generate K, V tensors for each token × each layer       │
│ - This is the expensive part: O(tokens × layers × hidden²) │
└────────────────────────────────────────────────────────────┘
         ↓
     [KV Cache stored in GPU memory]
         ↓
┌────────────────────────────────────────────────────────────┐
│ DECODE (memory-bound, sequential tokens)                   │
│ - Generate ONE token at a time                             │
│ - Read entire KV cache, compute attention, produce token   │
│ - Add new token's K, V to cache                            │
│ - Repeat until done                                        │
└────────────────────────────────────────────────────────────┘
         ↓
OUTPUT: " Paris"
```

**Key insight**: Prefill is where compute is spent. Decode is where memory bandwidth is the bottleneck.

### What "Shared Prefix" Really Means at the Token Level

Tokens are integers. Prompts are lists of integers:

```python
tokenizer.encode("You are a helpful assistant.")
# → [1, 887, 526, 263, 8444, 20255, 29889]

tokenizer.encode("You are a helpful chatbot.")
# → [1, 887, 526, 263, 8444, 13563, 7451, 29889]
#    ↑──────── shared ────────↑ ↑── different ──↑
```

**Shared prefix** = longest common prefix of token ID lists.

Here: `[1, 887, 526, 263, 8444]` — 5 tokens shared, then divergence.

### Why Token-Level Reuse Beats Sequence-Level

**Sequence-level** (naive approach):
- Cache full prompts as keys
- Only hit if EXACT match
- "You are a helpful assistant. User: X" ≠ "You are a helpful assistant. User: Y"
- **Result**: Almost no reuse in practice

**Block-level** (vLLM APC):
- Divide tokens into fixed blocks (e.g., 16 tokens each)
- Hash each block, cache by hash
- Can reuse blocks, but only at block boundaries
- Prefix of 15 tokens? No reuse. 17 tokens? Reuse 16.

**Token-level** (SGLang):
- Find longest matching prefix token-by-token
- Reuse ANY length prefix, at ANY boundary
- **Result**: Maximum possible reuse

### What a Radix Tree Is (Conceptually)

Imagine organizing a dictionary by shared prefixes:

```
Words: "cat", "car", "card", "care", "dog"

                ROOT
                 │
        ┌────────┼────────┐
        ↓                 ↓
       "ca"              "dog"
        │
   ┌────┼────┐
   ↓         ↓
  "t"       "r"
             │
       ┌─────┼─────┐
       ↓           ↓
      "d"         "e"
```

- "cat" and "car" share "ca"
- "card" and "care" share "car"
- Each node stores the **shared segment**
- Lookup = walk down matching path

**For KV cache**: Instead of storing strings, nodes store:
- `key`: Token ID segments
- `value`: Pointers to KV cache memory slots

---

## C) Repo Map (Engineer-Level, Code-Driven)

### Directory Structure for Radix Cache

```
python/sglang/srt/
├── mem_cache/
│   ├── radix_cache.py          # Core radix tree implementation
│   ├── base_prefix_cache.py    # Abstract interface
│   ├── evict_policy.py         # LRU/LFU/Priority eviction
│   ├── memory_pool.py          # KV cache memory management
│   ├── allocator.py            # Token-to-KV slot allocation
│   ├── hiradix_cache.py        # Hierarchical (GPU→CPU→Storage)
│   └── cpp_radix_tree/         # C++ optimized tree (optional)
├── managers/
│   ├── scheduler.py            # Main scheduler loop
│   ├── schedule_policy.py      # LPM/cache-aware scheduling
│   ├── schedule_batch.py       # Batch/request data structures
│   └── session_controller.py   # Multi-request sessions
├── model_executor/
│   ├── model_runner.py         # Forward pass execution
│   └── forward_batch_info.py   # Tensor batch metadata
└── layers/
    └── radix_attention.py      # Attention layer using cache
```

### 1) Radix Tree / Prefix Cache

**File**: `python/sglang/srt/mem_cache/radix_cache.py`

**Core Classes**:

```python
class RadixKey:
    """Key for radix tree lookups."""
    def __init__(self, token_ids: List[int], extra_key: Optional[str] = None):
        self.token_ids = token_ids      # The actual token sequence
        self.extra_key = extra_key      # Namespace (e.g., LoRA ID)
```

**Why `extra_key` exists**: Different LoRA adapters produce different KV values for the same tokens. `extra_key` isolates them.

```python
class TreeNode:
    """A node in the radix tree."""
    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)   # Child nodes by first token
        self.parent: TreeNode = None
        self.key: RadixKey = None               # Token segment this node represents
        self.value: Optional[torch.Tensor] = None  # KV cache indices (GPU)
        self.lock_ref = 0                       # Reference count (prevent eviction)
        self.last_access_time = time.monotonic()
        self.priority = priority                # For priority-aware eviction
```

**Why `lock_ref` exists**: Active requests must not have their cached KV evicted mid-computation.

```python
class RadixCache(BasePrefixCache):
    """The main radix tree for KV cache management."""
    def __init__(self, params: CacheInitParams):
        self.root_node = TreeNode()
        self.page_size = params.page_size  # 1 = token-level, 16+ = page-level
        self.eviction_strategy = LRUStrategy()  # Or LFU, Priority, etc.
```

**Why `page_size` exists**: Token-level (1) maximizes reuse but increases tree size. Page-level (16+) reduces overhead but may miss partial matches.

### 2) Cache Lookup + Insertion

**File**: `python/sglang/srt/mem_cache/radix_cache.py`

**Lookup** — `match_prefix()`:

```python
def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
    """Find longest cached prefix of key in the radix tree."""
    value, last_node = self._match_prefix_helper(self.root_node, key)
    return MatchResult(
        device_indices=torch.cat(value) if value else torch.empty(...),
        last_device_node=last_node,  # For lock reference
    )

def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
    """Walk the tree, collecting KV indices for matched segments."""
    value = []
    while len(key) > 0 and child_key in node.children:
        child = node.children[child_key]
        prefix_len = self.key_match_fn(child.key, key)
        
        if prefix_len < len(child.key):
            # Partial match within node → split node at mismatch
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            return value, new_node
        else:
            # Full node match → continue to children
            value.append(child.value)
            node = child
            key = key[prefix_len:]
    
    return value, node
```

**Why `_split_node` exists**: If we match "ABC" but the node stores "ABCDE", we split into "ABC" and "DE" to reuse "ABC".

**Insertion** — `insert()`:

```python
def insert(self, key: RadixKey, value=None, priority: int = 0):
    """Insert a token sequence and its KV indices into the tree."""
    return self._insert_helper(self.root_node, key, value, priority)

def _insert_helper(self, node: TreeNode, key: RadixKey, value, priority: int = 0):
    while len(key) > 0 and child_key in node.children:
        node = node.children[child_key]
        prefix_len = self.key_match_fn(node.key, key)
        
        if prefix_len < len(node.key):
            # Partial match → split, then insert remainder
            self._split_node(node.key, node, prefix_len)
        
        key = key[prefix_len:]
        value = value[prefix_len:]
    
    if len(key):
        # Create new node for remaining tokens
        new_node = TreeNode(priority=priority)
        new_node.key = key
        new_node.value = value  # KV cache indices
        node.children[child_key] = new_node
```

**When insertion happens**: After prefill completes (both finished and unfinished requests).

```python
def cache_finished_req(self, req: Req, is_insert: bool = True):
    """Cache a completed request's KV into the tree."""
    token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
    kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :]
    
    radix_key = RadixKey(token_ids, req.extra_key)
    new_prefix_len = self.insert(radix_key, kv_indices)
    
    # Free duplicate indices (already in tree)
    self.token_to_kv_pool_allocator.free(
        kv_indices[req.cache_protected_len : new_prefix_len]
    )
```

### 3) Execution / Scheduling Logic

**File**: `python/sglang/srt/managers/scheduler.py`

**Scheduler class** (simplified):

```python
class Scheduler:
    def __init__(self, ...):
        self.waiting_queue: List[Req] = []       # Requests waiting for prefill
        self.running_batch: ScheduleBatch = None # Currently executing
        self.tree_cache: RadixCache = ...        # The radix cache
        self.schedule_policy = SchedulePolicy(policy="lpm", tree_cache=...)
```

**Main loop** (conceptual):

```python
def event_loop_normal(self):
    while True:
        # 1. Receive new requests
        recv_requests()
        
        # 2. Process completed outputs
        process_batch_result(self.last_batch)
        
        # 3. Get next batch (this is where cache lookup happens)
        batch = self.get_next_batch_to_run()
        
        # 4. Run model forward
        self.run_batch(batch)
```

**File**: `python/sglang/srt/managers/schedule_policy.py`

**Cache-aware scheduling** — `SchedulePolicy.calc_priority()`:

```python
def _compute_prefix_matches(self, waiting_queue: List[Req]):
    """For each waiting request, find its cache hit length."""
    for r in waiting_queue:
        prefix_ids = r.origin_input_ids + r.output_ids
        radix_key = RadixKey(token_ids=prefix_ids, extra_key=r.extra_key)
        
        # THIS IS THE KEY CALL: lookup in radix tree
        match_result = self.tree_cache.match_prefix(radix_key)
        
        r.prefix_indices = match_result.device_indices  # Cached KV slots
        r.last_node = match_result.last_device_node     # For locking
    
def _sort_by_longest_prefix(self, waiting_queue: List[Req]):
    """LPM policy: schedule requests with most cache hits first."""
    waiting_queue.sort(key=lambda r: -len(r.prefix_indices))
```

**Why LPM (Longest Prefix Match)**: Requests with longer prefix hits complete faster → better throughput.

**PrefillAdder** — decides what fits in next batch:

```python
class PrefillAdder:
    def add_one_req(self, req: Req, ...):
        """Try to add a request to the next prefill batch."""
        # req.extend_input_len = tokens needing prefill (total - cached)
        # req.prefix_indices = KV slots already cached
        
        real_input_tokens = req.extend_input_len - req.host_hit_length
        
        if total_tokens >= self.rem_total_tokens:
            return AddReqResult.NO_TOKEN  # Out of memory
        
        # Lock the matched cache nodes
        self.tree_cache.inc_lock_ref(req.last_node)
        
        self.can_run_list.append(req)
```

### 4) Model Execution + Kernel Calls

**File**: `python/sglang/srt/model_executor/model_runner.py`

**ModelRunner** executes the actual forward pass:

```python
class ModelRunner:
    def forward(self, forward_batch: ForwardBatch, ...):
        """Run model forward pass."""
        # forward_batch contains:
        # - input_ids: tokens to process
        # - req_to_token_pool: mapping from request → KV slots
        # - out_cache_loc: where to write new KV
        
        logits = self.model.forward(
            input_ids=forward_batch.input_ids,
            positions=forward_batch.positions,
            forward_batch=forward_batch,  # Contains KV cache pointers
        )
        return logits
```

**File**: `python/sglang/srt/layers/radix_attention.py`

**RadixAttention layer**:

```python
class RadixAttention(nn.Module):
    def forward(self, q, k, v, forward_batch: ForwardBatch, save_kv_cache: bool = True):
        # k, v are computed for NEW tokens only
        # forward_batch.attn_backend handles reusing cached KV
        
        return forward_batch.attn_backend.forward(
            q, k, v, self, forward_batch, save_kv_cache
        )
```

The attention backend (FlashInfer, Triton, etc.) knows:
- Which KV slots contain cached data
- Which slots are being written with new data
- How to compute attention over the combined sequence

---

## D) One Request Walkthrough

**Scenario**: Request with partial prefix match.

```
Existing cache contains: "You are a helpful assistant. User: What is Python?"
New request:             "You are a helpful assistant. User: What is Java?"
                          └──────────── 10 tokens match ────────────┘
```

### Step-by-Step Trace

```
1. REQUEST ARRIVES
   ─────────────────────────────────────────────────────────────────
   TokenizedGenerateReqInput:
     token_ids = [1, 887, 526, 263, 8444, 20255, 29889, 4911, 29901, ...]
                 └─────────────── "You are a helpful assistant." ───────┘
   
   → Scheduler.handle_generate_request()
   → Creates Req object, adds to waiting_queue

2. SCHEDULING DECISION  
   ─────────────────────────────────────────────────────────────────
   Scheduler.get_next_batch_to_run():
     → SchedulePolicy.calc_priority(waiting_queue)
       → For each request:
           radix_key = RadixKey(token_ids, extra_key)
           match_result = tree_cache.match_prefix(radix_key)   ← KEY STEP
   
   RadixCache.match_prefix():
     - Start at ROOT
     - Walk: ROOT → "You are a helpful assistant. User: What is " 
     - Match 10 tokens: [1, 887, 526, 263, 8444, 20255, 29889, 4911, 29901, 1724]
     - Divergence at "Python" vs "Java"
     - Return: MatchResult(
         device_indices=[45, 46, 47, 48, 49, 50, 51, 52, 53, 54],  # Cached KV slots
         last_device_node=<node for "What is ">
       )
   
   Request now has:
     req.prefix_indices = [45, 46, 47, ..., 54]  # 10 cached slots
     req.extend_input_len = 3                    # Only "Java?" needs prefill
     req.last_node = <tree node>

3. BATCH PREPARATION
   ─────────────────────────────────────────────────────────────────
   PrefillAdder.add_one_req(req):
     - req.extend_input_len = 3 (only new tokens)
     - Allocate 3 new KV slots: [100, 101, 102]
     - tree_cache.inc_lock_ref(req.last_node)  # Prevent eviction
     - can_run_list.append(req)
   
   ScheduleBatch created with:
     - req_to_token_pool: [45,46,...,54, 100,101,102]
                          └── cached ──┘ └── new ──┘

4. MODEL FORWARD (PREFILL)
   ─────────────────────────────────────────────────────────────────
   ModelRunner.forward(forward_batch):
     forward_batch:
       input_ids = [14603, 29973, ...]  # Only "Java?" tokens
       positions = [10, 11, 12]          # Continue from position 10
       out_cache_loc = [100, 101, 102]   # Where to write new KV
       prefix_lens = [10]                # 10 tokens from cache
   
   Attention computation:
     - Read cached KV from slots [45-54]
     - Compute new KV for tokens at positions [10,11,12]
     - Write new KV to slots [100,101,102]
     - Attend over all 13 positions

5. DECODE LOOP
   ─────────────────────────────────────────────────────────────────
   ForwardMode.DECODE:
     - Generate tokens one by one
     - Each new token: allocate 1 slot, compute K/V, append to cache
     - Attend over growing sequence

6. REQUEST COMPLETION
   ─────────────────────────────────────────────────────────────────
   tree_cache.cache_finished_req(req):
     - Full sequence: "You are a helpful assistant. User: What is Java? ..."
     - Insert into tree:
       
       BEFORE:                          AFTER:
       "...What is "                    "...What is "
            │                                │
            └─► "Python?"                    ├─► "Python?"
                                             └─► "Java? ..." (NEW)
     
     - Free duplicate slots (already in tree)
     - tree_cache.dec_lock_ref(req.last_node)  # Allow eviction
```

---

## E) ASCII Diagrams (Mandatory)

### 1) Inference WITHOUT Radix Cache (Repeated Prefill Waste)

```
Timeline: 3 requests with shared prefix
═══════════════════════════════════════════════════════════════════════════

Request 1: "You are helpful. What is Python?"
┌─────────────────────────────────────┬─────────────────┐
│         PREFILL (expensive)         │     DECODE      │
│   [You are helpful. What is Python?]│  → "Python is"  │
│         16 tokens computed          │                 │
└─────────────────────────────────────┴─────────────────┘
                                                          Time ──►

Request 2: "You are helpful. What is Java?"
┌─────────────────────────────────────┬─────────────────┐
│         PREFILL (expensive)         │     DECODE      │
│   [You are helpful. What is Java?]  │  → "Java is"    │
│         16 tokens computed          │                 │
│         ↑↑↑ WASTE: 14 tokens same   │                 │
└─────────────────────────────────────┴─────────────────┘

Request 3: "You are helpful. What is Rust?"
┌─────────────────────────────────────┬─────────────────┐
│         PREFILL (expensive)         │     DECODE      │
│   [You are helpful. What is Rust?]  │  → "Rust is"    │
│         16 tokens computed          │                 │
│         ↑↑↑ WASTE: 14 tokens same   │                 │
└─────────────────────────────────────┴─────────────────┘

TOTAL: 48 tokens prefilled
WASTE: 28 tokens redundantly computed (58%)
```

### 2) Radix Tree Structure with Shared Token Prefixes

```
Token IDs (simplified):
  "You are helpful" = [1, 2, 3]
  "What is"         = [4, 5]
  "Python"          = [6]
  "Java"            = [7]
  "Rust"            = [8]

Radix Tree After 3 Requests:
══════════════════════════════════════════════════════════════

                              ┌─────────────────────┐
                              │       ROOT          │
                              │ key: []             │
                              │ value: []           │
                              │ lock_ref: 1         │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │ "You are helpful"   │
                              │ key: [1, 2, 3]      │
                              │ value: [0, 1, 2]    │◄─── KV slots in GPU
                              │ lock_ref: 0         │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │ "What is"           │
                              │ key: [4, 5]         │
                              │ value: [3, 4]       │◄─── Shared by all 3
                              │ lock_ref: 0         │
                              └──────────┬──────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
   ┌──────────▼──────────┐   ┌───────────▼───────────┐   ┌──────────▼──────────┐
   │ "Python?"           │   │ "Java?"               │   │ "Rust?"             │
   │ key: [6, ...]       │   │ key: [7, ...]         │   │ key: [8, ...]       │
   │ value: [5, 6, ...]  │   │ value: [10, 11, ...]  │   │ value: [15, 16, ...]│
   └─────────────────────┘   └───────────────────────┘   └─────────────────────┘

Memory Layout:
KV Cache Slots: [0][1][2][3][4][5][6]...[10][11]...[15][16]...
                │  │  │  │  │  │  │      │   │      │   │
                └──┴──┴──┴──┴──┘  │      │   │      │   │
                   SHARED         └──────┴───┘      └───┘
                   (5 tokens)     Request 2         Request 3
```

### 3) Partial Prefix Match and Reuse Path

```
Existing tree has: "You are helpful. What is Python?" (cached)
New request:       "You are helpful. What is Java?"

STEP 1: Lookup (match_prefix)
════════════════════════════════════════════════════════════════════

                  ┌──────────┐
                  │   ROOT   │
                  └────┬─────┘
                       │ Match!
                  ┌────▼─────┐
                  │ [1,2,3]  │ "You are helpful"
                  │ val:[0-2]│ ← REUSE these KV slots
                  └────┬─────┘
                       │ Match!
                  ┌────▼─────┐
                  │ [4, 5]   │ "What is"
                  │ val:[3-4]│ ← REUSE these too
                  └────┬─────┘
                       │ MISMATCH! "Python" ≠ "Java"
                  ┌────▼─────┐
                  │ [6, ...]  │ "Python?" 
                  │ val:[5-8] │ ← NOT reused
                  └──────────┘

Result: 
  prefix_indices = [0, 1, 2, 3, 4]  ← 5 tokens cached
  extend_input_len = 2              ← Only "Java?" needs prefill


STEP 2: Prefill (only new tokens)
════════════════════════════════════════════════════════════════════

  Tokens to prefill: [7, ...] ("Java?")
  Positions: [5, 6, ...]
  
  ┌──────────────────────────────────────────────────────────────┐
  │ Attention Computation:                                        │
  │                                                                │
  │  Q = [q5, q6]  ← From new tokens only                         │
  │                                                                │
  │  K = [k0, k1, k2, k3, k4, k5, k6]                              │
  │       └── from cache ──┘  └ new ┘                              │
  │                                                                │
  │  V = [v0, v1, v2, v3, v4, v5, v6]                              │
  │       └── from cache ──┘  └ new ┘                              │
  │                                                                │
  │  Output = softmax(Q @ K.T) @ V                                 │
  └──────────────────────────────────────────────────────────────┘


STEP 3: Insert (after completion)
════════════════════════════════════════════════════════════════════

                  ┌──────────┐
                  │   ROOT   │
                  └────┬─────┘
                  ┌────▼─────┐
                  │ [1,2,3]  │
                  └────┬─────┘
                  ┌────▼─────┐
                  │ [4, 5]   │
                  └────┬─────┘
                       │
         ┌─────────────┼─────────────┐
         │                           │
   ┌─────▼─────┐               ┌─────▼─────┐
   │ [6, ...]  │               │ [7, ...]  │ ← NEW NODE
   │ "Python?" │               │ "Java?"   │
   └───────────┘               └───────────┘
```

### 4) Execution Timeline: Cache Hit vs Cache Miss

```
CACHE HIT (Request 2 after Request 1 established prefix)
═══════════════════════════════════════════════════════════════════════

        │ Receive │ Lookup │    Prefill      │     Decode      │ Done
        ├─────────┼────────┼─────────────────┼─────────────────┤
Time 0  │    ▓    │        │                 │                 │
     1  │         │   ▓    │                 │                 │ ← Tree walk
     2  │         │        │  ▓▓             │                 │ ← Only 2 new tokens!
     3  │         │        │                 │   ▓▓▓▓▓         │
     4  │         │        │                 │                 │  ✓
        └─────────┴────────┴─────────────────┴─────────────────┘
        
        TTFT (Time to First Token) = ~3 time units


CACHE MISS (Request 1 with no prior cache)
═══════════════════════════════════════════════════════════════════════

        │ Receive │ Lookup │    Prefill      │     Decode      │ Done
        ├─────────┼────────┼─────────────────┼─────────────────┤
Time 0  │    ▓    │        │                 │                 │
     1  │         │   ▓    │                 │                 │ ← Quick (no match)
     2  │         │        │  ▓▓▓▓▓▓▓▓       │                 │ ← ALL 16 tokens
     3  │         │        │  ▓▓▓▓▓▓▓▓       │                 │   (expensive!)
     4  │         │        │                 │   ▓▓▓▓▓         │
     5  │         │        │                 │                 │  ✓
        └─────────┴────────┴─────────────────┴─────────────────┘
        
        TTFT (Time to First Token) = ~5 time units  (67% slower!)
```

### 5) Comparison: vLLM (Paged KV) vs SGLang (Radix KV)

```
                    vLLM Paged Attention           SGLang Radix Cache
═══════════════════════════════════════════════════════════════════════════

DATA STRUCTURE:
                    
  ┌─────────────────────────────┐    ┌─────────────────────────────────┐
  │     Hash Table              │    │     Radix Tree                  │
  │ ┌────────────────────────┐  │    │              ROOT               │
  │ │ hash(block0) → slot 0  │  │    │               │                 │
  │ │ hash(blk0+1) → slot 1  │  │    │        ┌──────┴──────┐          │
  │ │ hash(blk0+1+2) → ...   │  │    │     [tokens]      [tokens]      │
  │ └────────────────────────┘  │    │        │              │         │
  │                             │    │     [tokens]      [tokens]      │
  └─────────────────────────────┘    └─────────────────────────────────┘

MATCHING:

  Prompt: "ABCDEFGHIJ..." (10 tokens, block_size=4)
  
  vLLM:                             SGLang:
  ─────                             ──────
  Block 0: [A,B,C,D] → hash → hit?  Token-by-token walk:
  Block 1: [E,F,G,H] → hash → hit?  A→B→C→D→E→F→G→H→I→J
  Block 2: [I,J,?,?] → ❌ partial   Match stops at first mismatch
                                    
  If "ABCDEFGHIXYZ" exists:         If "ABCDEFGHIXYZ" exists:
  - Blocks 0,1 hit (8 tokens)       - Match ABCDEFGHI (9 tokens)
  - Block 2 miss (no partial)       - Reuse 9, compute 1
  - Reuse 8, compute 2              - Better!

WHEN vLLM WINS:
────────────────
- Low prefix overlap → simpler hash beats tree overhead
- Very diverse traffic → hash table more memory efficient
- Need block-level eviction control

WHEN SGLang WINS:
─────────────────
- High prefix overlap → token-level matching finds more reuse
- Chat/agents → progressive prefix growth
- Variable-length prefixes → no block alignment required

MEMORY OVERHEAD:
────────────────
vLLM:  O(num_blocks) hash entries + block metadata
SGLang: O(tree_nodes) where nodes = unique prefix segments
        → More overhead if many unique short prefixes
        → Less overhead if few long shared prefixes
```

---

## F) Practical Takeaways

### Strengths of Radix Caching

1. **Maximum prefix reuse**: Token-level matching finds longest possible overlap
2. **Progressive matching**: Multi-turn conversations get increasingly fast
3. **Namespace isolation**: `extra_key` supports multi-LoRA, multi-tenant
4. **Flexible eviction**: LRU/LFU/Priority policies, configurable
5. **Hierarchical extension**: HiCache adds CPU and distributed storage tiers

### Weaknesses of Radix Caching

1. **Tree overhead**: Memory and CPU cost per cached segment
2. **Worst case**: Random prompts = zero reuse + full tree overhead
3. **Eviction complexity**: Leaf-only eviction can leave internal fragmentation
4. **Scheduling overhead**: LPM policy is O(n × prefix_len) for n requests

### Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| **Cache hit rate ~0%** | Prompts don't share prefixes | Redesign prompts, put shared content first |
| **High memory, low hits** | Many unique short prefixes | Increase `page_size` to 16+ |
| **TTFT variance** | Cache thrashing under load | Increase memory, use priority eviction |
| **Wrong outputs with LoRA** | Shared cache across adapters | Set `extra_key` to LoRA ID |
| **Slow scheduling** | >128 requests waiting | Falls back to FCFS, or increase server capacity |
| **OOM during prefill** | Cache not evicting fast enough | Reduce `--mem-fraction-static` |

### Workload Pattern Decision Guide

```
IF workload has:
  - Multi-turn conversations    → Radix cache helps a LOT (5x+)
  - Agent/tool system prompts   → Radix cache helps a LOT (3x+)
  - Few-shot examples           → Radix cache helps SIGNIFICANTLY (2-3x)
  - RAG with similar docs       → Radix cache helps MODERATELY
  - Batch inference, diverse    → Radix cache helps MINIMALLY (maybe hurts)
  - Very short prompts (<50)    → Radix cache may HURT (tree overhead)
```

### Top Tuning Knobs

| Parameter | What It Controls | Recommended Value |
|-----------|-----------------|-------------------|
| `--page-size` | Cache matching granularity | 1 (max reuse) or 16 (balanced) |
| `--schedule-policy` | Request ordering | `lpm` for cache-aware |
| `--eviction-policy` | What to evict first | `lru` (default) |
| `--mem-fraction-static` | GPU memory for KV cache | 0.8-0.9 |
| `--disable-radix-cache` | Turn off caching | Only for debugging/benchmarks |
| `--enable-hierarchical-cache` | Add CPU/storage tiers | For very long contexts |

### When to Combine SGLang with vLLM

**Hybrid Architecture**:

```
                    ┌─────────────────────────────────────┐
                    │           Load Balancer             │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
       ┌──────▼──────┐      ┌──────▼──────┐     ┌──────▼──────┐
       │  SGLang     │      │  SGLang     │     │   vLLM      │
       │  (agents)   │      │  (chat)     │     │  (batch)    │
       │             │      │             │     │             │
       │ High prefix │      │ High prefix │     │ Low prefix  │
       │ overlap     │      │ overlap     │     │ overlap     │
       └─────────────┘      └─────────────┘     └─────────────┘
```

- Route chat/agent traffic to SGLang
- Route batch/diverse traffic to vLLM
- Use cache-aware load balancing (sticky sessions for multi-turn)

---

## Quick Reference: Code Paths

| Operation | File | Function/Class |
|-----------|------|----------------|
| **Radix tree definition** | `mem_cache/radix_cache.py` | `RadixCache`, `TreeNode`, `RadixKey` |
| **Prefix lookup** | `mem_cache/radix_cache.py` | `match_prefix()`, `_match_prefix_helper()` |
| **Cache insertion** | `mem_cache/radix_cache.py` | `insert()`, `cache_finished_req()` |
| **Eviction** | `mem_cache/radix_cache.py` | `evict()`, `_collect_leaves()` |
| **Eviction policies** | `mem_cache/evict_policy.py` | `LRUStrategy`, `LFUStrategy`, etc. |
| **KV memory pool** | `mem_cache/memory_pool.py` | `MHATokenToKVPool`, `MLATokenToKVPool` |
| **Scheduler main loop** | `managers/scheduler.py` | `Scheduler.event_loop_normal()` |
| **Cache-aware scheduling** | `managers/schedule_policy.py` | `SchedulePolicy`, `PrefillAdder` |
| **Request data structure** | `managers/schedule_batch.py` | `Req`, `ScheduleBatch` |
| **Model forward** | `model_executor/model_runner.py` | `ModelRunner.forward()` |
| **Attention layer** | `layers/radix_attention.py` | `RadixAttention.forward()` |

---

## Final Thoughts

SGLang's radix cache exists because **most LLM serving workloads waste compute on repeated prefixes**, and paging alone doesn't solve this.

The radix tree design feels inevitable once you realize:
1. Prefill is the compute bottleneck
2. Prompts share prefixes at the token level
3. Token-level matching maximizes reuse
4. A tree naturally represents prefix sharing

**If your workload has shared prefixes, use SGLang. If not, benchmark both.**

The code is modular enough that you can modify the radix cache in week one:
- Add a new eviction policy: implement `EvictionStrategy` in `evict_policy.py`
- Change matching granularity: modify `page_size` parameter
- Add namespace isolation: use `extra_key` in `RadixKey`
- Extend to multi-tier: study `hiradix_cache.py`

Start with `radix_cache.py:match_prefix()` and trace through a request. Everything else follows from there.
