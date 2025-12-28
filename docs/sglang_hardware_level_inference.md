# Hardware-Level View: LLM Inference on NVIDIA Blackwell B200

**Understanding what actually happens on the GPU when you run inference with SGLang**

---

## The Hardware: NVIDIA Blackwell B200 at a Glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NVIDIA BLACKWELL B200 GPU                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    192 GB HBM3e Memory                               │   │
│  │                    ~8 TB/s Memory Bandwidth                          │   │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                  │   │
│  │  │Stack│Stack│Stack│Stack│Stack│Stack│Stack│Stack│  8 HBM3e stacks  │   │
│  │  │ 0   │ 1   │ 2   │ 3   │ 4   │ 5   │ 6   │ 7   │  24GB each       │   │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                        │
│                                    │ 8 TB/s                                 │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    L2 Cache (96 MB unified)                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                        │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────── 208 Streaming Multiprocessors ─────────────────┐   │
│  │ ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐ ... ┌────┐┌────┐  │   │
│  │ │SM 0││SM 1││SM 2││SM 3││SM 4││SM 5││SM 6││SM 7│     │SM  ││SM  │  │   │
│  │ │    ││    ││    ││    ││    ││    ││    ││    │     │206 ││207 │  │   │
│  │ └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘     └────┘└────┘  │   │
│  │                                                                     │   │
│  │  Each SM contains:                                                  │   │
│  │  • 4th Gen Tensor Cores (FP4, FP8, FP16, BF16, TF32, FP64)         │   │
│  │  • 128 CUDA cores                                                   │   │
│  │  • 256 KB Register File                                             │   │
│  │  • 228 KB L1 Cache / Shared Memory                                  │   │
│  │  • 2nd Gen Transformer Engine                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Peak Performance:                                                          │
│  • FP4 Tensor: 18 PFLOPS (with sparsity)                                   │
│  • FP8 Tensor: 9 PFLOPS                                                    │
│  • BF16 Tensor: 4.5 PFLOPS                                                 │
│  • TDP: 1000W                                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## A Real Inference Request: End-to-End Hardware View

### The Request

```
User sends:
┌──────────────────────────────────────────────────────────────────────────┐
│ System: "You are a helpful AI assistant."                                │
│ User: "Explain quantum computing in simple terms."                       │
└──────────────────────────────────────────────────────────────────────────┘

Tokenized to: [1, 887, 526, 263, ..., 12345, 29973]  (50 tokens)
Expected output: ~100 tokens
```

---

## Stage 1: Request Arrival and Tokenization (CPU)

### What Happens

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CPU (Host)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. HTTP Request Received                                                   │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ POST /v1/chat/completions                                       │    │
│     │ {"messages": [{"role": "user", "content": "Explain..."}]}       │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  2. Tokenization (CPU-bound, ~0.1ms)                                       │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ tokenizer.encode("You are a helpful AI assistant...")          │    │
│     │ → [1, 887, 526, 263, 8444, 20255, 29889, ...]                   │    │
│     │                                                                 │    │
│     │ Memory: ~400 bytes (50 tokens × 8 bytes/token)                  │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  3. Radix Cache Lookup (CPU, ~0.01ms)                                      │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ radix_cache.match_prefix(RadixKey([1, 887, 526, ...]))          │    │
│     │                                                                 │    │
│     │ Scenario A: Cache HIT (30 tokens matched)                       │    │
│     │   → prefix_indices = [45, 46, 47, ..., 74]  (30 KV slots)       │    │
│     │   → extend_input_len = 20 tokens (need prefill)                 │    │
│     │                                                                 │    │
│     │ Scenario B: Cache MISS (0 tokens matched)                       │    │
│     │   → prefix_indices = []                                          │    │
│     │   → extend_input_len = 50 tokens (full prefill)                 │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  4. Schedule into Batch (CPU, ~0.05ms)                                     │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ PrefillAdder.add_one_req(req)                                   │    │
│     │ → Check memory budget                                           │    │
│     │ → Allocate KV slots for new tokens                              │    │
│     │ → Lock cached nodes (prevent eviction)                          │    │
│     │ → Add to can_run_list                                           │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Problems Solved:
• Converts human-readable text to model-consumable tokens
• Finds reusable cached computation (avoids redundant prefill)
• Groups requests efficiently for GPU utilization

Why This Approach:
• Tokenization is fast on CPU, no GPU needed
• Radix lookup is O(prefix_length), very efficient
• Batching amortizes GPU kernel launch overhead
```

---

## Stage 2: Data Transfer to GPU (PCIe/NVLink)

### What Happens

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Host-to-Device Transfer                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CPU RAM                              GPU HBM3e                             │
│  ┌──────────────────┐                 ┌──────────────────┐                  │
│  │                  │   PCIe Gen5     │                  │                  │
│  │  Input Token IDs │ ═══════════════►│  Input Token IDs │                  │
│  │  [1,887,526,...] │   64 GB/s       │  [1,887,526,...] │                  │
│  │  200 bytes       │                 │                  │                  │
│  │                  │                 │                  │                  │
│  │  Attention Mask  │ ═══════════════►│  Attention Mask  │                  │
│  │  (if needed)     │                 │                  │                  │
│  │                  │                 │                  │                  │
│  └──────────────────┘                 └──────────────────┘                  │
│                                                                             │
│  Transfer Size (per batch):                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ • Token IDs: batch_size × seq_len × 4 bytes = 8 reqs × 50 × 4 = 1.6KB │ │
│  │ • Positions: batch_size × seq_len × 4 bytes = 1.6 KB               │    │
│  │ • Metadata: ~1 KB                                                   │    │
│  │                                                                     │    │
│  │ Total: ~5 KB (negligible vs. model weights)                         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  What's NOT transferred (already on GPU):                                   │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ • Model weights: 70B params × 2 bytes = 140 GB (BF16)              │    │
│  │ • KV Cache: Already resident in HBM                                 │    │
│  │ • CUDA kernels: Already compiled and loaded                         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Problems Solved:
• Minimizes PCIe bottleneck by keeping weights/cache on GPU
• Only transfers small token IDs, not embeddings

Why This Approach:
• PCIe is 100x slower than HBM bandwidth
• Model weights (140GB) would take 2+ seconds to transfer
• Keeping everything GPU-resident enables sub-second inference
```

---

## Stage 3: Prefill Phase (Compute-Bound)

### High-Level View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREFILL PHASE                                       │
│                    (Process all input tokens in parallel)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: 20 new tokens (30 cached via radix)                                 │
│  Output: KV cache entries for 20 tokens × 80 layers                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      GPU EXECUTION TIMELINE                         │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  Time ──────────────────────────────────────────────────────────►   │   │
│  │                                                                     │   │
│  │  Layer 0:  [Embed][──Attn──][─MLP─]                                 │   │
│  │  Layer 1:        [──Attn──][─MLP─]                                  │   │
│  │  Layer 2:              [──Attn──][─MLP─]                            │   │
│  │    ...                       ...                                    │   │
│  │  Layer 79:                              [──Attn──][─MLP─][Logits]   │   │
│  │                                                                     │   │
│  │  ◄────────────────── ~15ms total ─────────────────────►             │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Embedding Lookup (First Operation)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EMBEDDING LOOKUP                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  What: Convert token IDs to dense vectors                                   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  HBM Memory Layout                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │        Embedding Table (vocab_size × hidden_dim)                │ │  │
│  │  │        128,256 tokens × 8,192 dims × 2 bytes = 2.1 GB           │ │  │
│  │  │  ┌─────────────────────────────────────────────────────────┐    │ │  │
│  │  │  │ Token 0:    [0.12, -0.34, 0.56, ..., 0.78]  (8192 floats)│   │ │  │
│  │  │  │ Token 1:    [0.23, -0.45, 0.67, ..., 0.89]               │   │ │  │
│  │  │  │ Token 2:    [0.34, -0.56, 0.78, ..., 0.90]               │   │ │  │
│  │  │  │ ...                                                      │   │ │  │
│  │  │  │ Token 128K: [0.11, -0.22, 0.33, ..., 0.44]               │   │ │  │
│  │  │  └─────────────────────────────────────────────────────────┘    │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  │  Gather Operation:                                                    │  │
│  │  Input tokens: [1, 887, 526, ..., 12345]  (20 tokens)                │  │
│  │                    │    │    │         │                              │  │
│  │                    ▼    ▼    ▼         ▼                              │  │
│  │  Output:     [emb_1, emb_887, emb_526, ..., emb_12345]               │  │
│  │              └───────────── 20 × 8192 = 163,840 floats ─────────────┘│  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Hardware Utilization:                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ • Operation: Scattered memory reads (gather)                         │  │
│  │ • Memory Bound: Yes (random access pattern)                          │  │
│  │ • Bandwidth Used: ~500 GB/s (sparse access)                          │  │
│  │ • Time: ~0.05ms                                                       │  │
│  │ • SMs Active: 208 (all, but low utilization)                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Attention Layer (The Core Computation)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION COMPUTATION (per layer)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Configuration (Llama-70B style):                                           │
│  • 64 attention heads, 8 KV heads (GQA)                                     │
│  • Head dimension: 128                                                      │
│  • Hidden dimension: 8192                                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Input: X ∈ R^{20 × 8192}  (20 tokens, 8192 hidden dim)            │   │
│  │                                                                     │   │
│  │  Step 1: QKV Projection (Matrix Multiply - Tensor Cores)           │   │
│  │  ════════════════════════════════════════════════════════          │   │
│  │                                                                     │   │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐                  │   │
│  │  │    X       │   │   W_QKV    │   │   QKV      │                  │   │
│  │  │ [20×8192]  │ × │[8192×9216] │ = │ [20×9216]  │                  │   │
│  │  │ (input)    │   │ (weights)  │   │ (output)   │                  │   │
│  │  └────────────┘   └────────────┘   └────────────┘                  │   │
│  │                                                                     │   │
│  │  FLOPS: 20 × 8192 × 9216 × 2 = 3.0 GFLOPS                          │   │
│  │  Memory Read: 8192 × 9216 × 2 = 151 MB (weights)                   │   │
│  │  Arithmetic Intensity: 3.0G / 151M = 20 FLOPS/byte                 │   │
│  │                                                                     │   │
│  │  Hardware Mapping:                                                  │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ SM 0-207: All 208 SMs active                                  │ │   │
│  │  │ Tensor Cores: BF16 matrix multiply (4.5 PFLOPS peak)          │ │   │
│  │  │ Actual Utilization: ~60% (memory bound for small batch)       │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Step 2: Attention Score Computation (FlashAttention)               │   │
│  │  ════════════════════════════════════════════════════════          │   │
│  │                                                                     │   │
│  │  With Radix Cache Reuse:                                            │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │                                                               │ │   │
│  │  │  Q = [q_30, q_31, ..., q_49]  ← Only new tokens (20 queries)  │ │   │
│  │  │       └──────── 20 ────────┘                                  │ │   │
│  │  │                                                               │ │   │
│  │  │  K = [k_0, k_1, ..., k_29,  k_30, k_31, ..., k_49]            │ │   │
│  │  │       └─ FROM CACHE (30) ─┘  └─── NEW (20) ────┘              │ │   │
│  │  │                                                               │ │   │
│  │  │  V = [v_0, v_1, ..., v_29,  v_30, v_31, ..., v_49]            │ │   │
│  │  │       └─ FROM CACHE (30) ─┘  └─── NEW (20) ────┘              │ │   │
│  │  │                                                               │ │   │
│  │  │  Attention: softmax(Q @ K^T / sqrt(d_k)) @ V                  │ │   │
│  │  │                                                               │ │   │
│  │  │  Key Insight: We only compute Q for NEW tokens,               │ │   │
│  │  │  but attend over ALL K,V (including cached)                   │ │   │
│  │  │                                                               │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  FlashAttention Tiling (fits in SRAM):                              │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │  L1/Shared Memory (228 KB per SM)                             │ │   │
│  │  │  ┌──────────────────────────────────────────────────────────┐ │ │   │
│  │  │  │ Q_tile: 64 × 128 × 2 = 16 KB                             │ │ │   │
│  │  │  │ K_tile: 64 × 128 × 2 = 16 KB                             │ │ │   │
│  │  │  │ V_tile: 64 × 128 × 2 = 16 KB                             │ │ │   │
│  │  │  │ Softmax accumulators: 8 KB                               │ │ │   │
│  │  │  │ Output accumulator: 16 KB                                │ │ │   │
│  │  │  │ ─────────────────────────────                            │ │ │   │
│  │  │  │ Total: ~72 KB (fits in 228 KB SRAM!)                     │ │ │   │
│  │  │  └──────────────────────────────────────────────────────────┘ │ │   │
│  │  │                                                               │ │   │
│  │  │  No HBM reads for intermediate attention matrices!            │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Step 3: Write New KV to Cache                                      │   │
│  │  ════════════════════════════════════════════════════════          │   │
│  │                                                                     │   │
│  │  KV Cache Memory Layout (per layer):                                │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ K_buffer: [max_tokens × num_kv_heads × head_dim]              │ │   │
│  │  │           [131072    ×      8         ×   128  ] × 2 bytes    │ │   │
│  │  │         = 256 MB per layer                                    │ │   │
│  │  │                                                               │ │   │
│  │  │ V_buffer: Same size = 256 MB per layer                        │ │   │
│  │  │                                                               │ │   │
│  │  │ Total KV per layer: 512 MB                                    │ │   │
│  │  │ Total KV (80 layers): 40 GB                                   │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  Write operation:                                                   │   │
│  │  kv_cache[layer_id][allocated_slots] = new_kv                       │   │
│  │  allocated_slots = [100, 101, ..., 119]  (20 new slots)             │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MLP Layer (Feed-Forward Network)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MLP COMPUTATION (per layer)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Architecture (Llama-style SwiGLU):                                         │
│  • Up projection: 8192 → 28672                                              │
│  • Gate projection: 8192 → 28672                                            │
│  • Down projection: 28672 → 8192                                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Step 1: Gate and Up Projection (parallel GEMM)                     │   │
│  │                                                                     │   │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐                  │   │
│  │  │    X       │   │   W_up     │   │   up       │                  │   │
│  │  │ [20×8192]  │ × │[8192×28672]│ = │ [20×28672] │                  │   │
│  │  └────────────┘   └────────────┘   └────────────┘                  │   │
│  │                                                                     │   │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐                  │   │
│  │  │    X       │   │  W_gate    │   │   gate     │                  │   │
│  │  │ [20×8192]  │ × │[8192×28672]│ = │ [20×28672] │                  │   │
│  │  └────────────┘   └────────────┘   └────────────┘                  │   │
│  │                                                                     │   │
│  │  FLOPS: 2 × (20 × 8192 × 28672 × 2) = 18.9 GFLOPS                  │   │
│  │  Weight Memory: 2 × 8192 × 28672 × 2 = 938 MB                      │   │
│  │                                                                     │   │
│  │  Step 2: SiLU activation + elementwise multiply                     │   │
│  │                                                                     │   │
│  │  hidden = SiLU(gate) × up                                           │   │
│  │  (elementwise, compute-cheap)                                       │   │
│  │                                                                     │   │
│  │  Step 3: Down Projection                                            │   │
│  │                                                                     │   │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐                  │   │
│  │  │  hidden    │   │   W_down   │   │   output   │                  │   │
│  │  │[20×28672]  │ × │[28672×8192]│ = │ [20×8192]  │                  │   │
│  │  └────────────┘   └────────────┘   └────────────┘                  │   │
│  │                                                                     │   │
│  │  Total MLP Memory Read: 938 + 469 = 1.4 GB per layer               │   │
│  │  Total FLOPS: 28.3 GFLOPS per layer                                 │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  SM Utilization During MLP:                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐    │   │
│  │  │SM 0 │SM 1 │SM 2 │SM 3 │SM 4 │ ... │SM205│SM206│SM207│     │    │   │
│  │  │ ██  │ ██  │ ██  │ ██  │ ██  │ ... │ ██  │ ██  │ ██  │     │    │   │
│  │  │ 95% │ 95% │ 94% │ 95% │ 94% │ ... │ 95% │ 94% │ 95% │     │    │   │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘    │   │
│  │                                                                     │   │
│  │  MLP is highly parallel and compute-dense → excellent SM usage      │   │
│  │  Tensor Cores running at near-peak efficiency                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Prefill Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PREFILL PHASE SUMMARY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  For 20 new tokens (30 cached) across 80 layers:                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Per-Layer Breakdown:                                               │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ Operation       │ FLOPS    │ Memory Read │ Time    │ Bound    │ │   │
│  │  ├────────────────────────────────────────────────────────────────┤ │   │
│  │  │ QKV Projection  │ 3.0 G    │ 151 MB      │ 0.02ms  │ Memory   │ │   │
│  │  │ Attention       │ 0.5 G    │ 20 MB (KV)  │ 0.01ms  │ Compute  │ │   │
│  │  │ Output Proj     │ 1.0 G    │ 134 MB      │ 0.02ms  │ Memory   │ │   │
│  │  │ MLP             │ 28.3 G   │ 1.4 GB      │ 0.15ms  │ Memory   │ │   │
│  │  ├────────────────────────────────────────────────────────────────┤ │   │
│  │  │ TOTAL per layer │ 32.8 G   │ 1.7 GB      │ 0.20ms  │          │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  80 Layers: 80 × 0.20ms = 16ms prefill time                         │   │
│  │                                                                     │   │
│  │  Memory Bandwidth Utilization:                                      │   │
│  │  • Total data read: 80 × 1.7 GB = 136 GB                           │   │
│  │  • Time: 16ms                                                       │   │
│  │  • Effective bandwidth: 136 GB / 16ms = 8.5 TB/s                   │   │
│  │  • B200 peak: 8 TB/s → We're at ~100% utilization!                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  With Radix Cache (30 tokens hit):                                          │
│  • Without cache: Would process 50 tokens → ~40ms                          │
│  • With cache: Process only 20 tokens → ~16ms                              │
│  • Savings: 60% reduction in TTFT                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 4: Decode Phase (Memory-Bound)

### Single Token Generation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DECODE PHASE                                        │
│                    (Generate tokens one at a time)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Each decode step: Process 1 new token, attend over entire sequence         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Token 51 Generation (first decode step):                           │   │
│  │                                                                     │   │
│  │  Input: [token_50] (just the last generated token)                  │   │
│  │                                                                     │   │
│  │  Attention must read ALL previous KV:                               │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ K,V sequence: [pos 0] [pos 1] ... [pos 49] [pos 50]           │ │   │
│  │  │               └────────────── 51 positions ───────────────────┘ │   │
│  │  │                                                               │ │   │
│  │  │ Memory to read per layer:                                     │ │   │
│  │  │ K: 51 × 8 heads × 128 dim × 2 bytes = 104 KB                  │ │   │
│  │  │ V: 51 × 8 heads × 128 dim × 2 bytes = 104 KB                  │ │   │
│  │  │ Total KV per layer: 208 KB                                    │ │   │
│  │  │ Total KV (80 layers): 16.6 MB                                 │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  Plus model weights: 140 GB                                         │   │
│  │                                                                     │   │
│  │  Arithmetic Intensity (FLOPS / byte):                               │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ FLOPS per token: ~140 GFLOPS (same as prefill per token)      │ │   │
│  │  │ Bytes read: 140 GB (weights) + 16 MB (KV) ≈ 140 GB            │ │   │
│  │  │ Intensity: 140G / 140G = ~1 FLOP/byte                         │ │   │
│  │  │                                                               │ │   │
│  │  │ For B200 to be compute-bound, need ~40 FLOPS/byte             │ │   │
│  │  │ → Decode is SEVERELY MEMORY BOUND                             │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Solution: BATCHING                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Single request:                                                    │   │
│  │  • Read 140 GB weights, do 140 GFLOPS → 1 FLOP/byte → 17ms         │   │
│  │                                                                     │   │
│  │  8 requests batched:                                                │   │
│  │  • Read 140 GB weights ONCE, do 8 × 140 GFLOPS                     │   │
│  │  • Intensity: 8 FLOPS/byte → Much better!                          │   │
│  │  • Time: ~19ms for 8 tokens (vs 8 × 17ms = 136ms sequential)       │   │
│  │                                                                     │   │
│  │  64 requests batched (saturate B200):                               │   │
│  │  • Intensity: 64 FLOPS/byte → Near compute-bound                   │   │
│  │  • Throughput: ~3000 tokens/sec                                    │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Decode with Continuous Batching

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CONTINUOUS BATCHING (SGLang)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Traditional Batching (Static):                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Batch of 8 requests, must wait for ALL to finish:                  │   │
│  │                                                                     │   │
│  │  Req 1: ████████████████████ (100 tokens)                           │   │
│  │  Req 2: ████████████ (60 tokens)           [idle padding]           │   │
│  │  Req 3: ████████ (40 tokens)               [idle padding]           │   │
│  │  Req 4: ██████████████████████████ (130 tokens)                     │   │
│  │  ...                                                                │   │
│  │  ◄─────────────────── Wait for Req 4 ────────────────────►          │   │
│  │                                                                     │   │
│  │  Problem: Short requests wait for long ones, GPU idles              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Continuous Batching (SGLang):                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Requests join/leave batch dynamically:                             │   │
│  │                                                                     │   │
│  │  Time →  [Step 1] [Step 2] [Step 3] [Step 4] [Step 5] ...           │   │
│  │                                                                     │   │
│  │  Req 1:    █         █         █         █        ─done─            │   │
│  │  Req 2:    █         █         ─done─   [Req 5]    █                │   │
│  │  Req 3:    █         ─done─   [Req 4]     █        █                │   │
│  │  Req 4:             (waiting) [insert]    █        █                │   │
│  │                                                                     │   │
│  │  • Finished requests exit immediately                               │   │
│  │  • New requests inserted into batch mid-flight                      │   │
│  │  • GPU always running at full batch size                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  SGLang Implementation:                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  # In scheduler.py                                                  │   │
│  │  while True:                                                        │   │
│  │      # Check for completed requests, remove from batch              │   │
│  │      completed = [r for r in batch if r.is_done()]                  │   │
│  │      for r in completed:                                            │   │
│  │          tree_cache.cache_finished_req(r)  # Insert to radix        │   │
│  │          send_response(r)                                           │   │
│  │                                                                     │   │
│  │      # Fill empty slots with waiting requests                       │   │
│  │      while len(batch) < max_batch and waiting_queue:                │   │
│  │          new_req = waiting_queue.pop()                              │   │
│  │          batch.add(new_req)                                         │   │
│  │                                                                     │   │
│  │      # Run one decode step                                          │   │
│  │      run_decode_step(batch)                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 5: Output Generation and Response

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      OUTPUT GENERATION                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  After 80 layers, we have logits for the next token:                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Logits: tensor of shape [batch_size, vocab_size]                   │   │
│  │          [8, 128256] = 8 MB                                         │   │
│  │                                                                     │   │
│  │  Sampling (GPU):                                                    │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ 1. Apply temperature: logits = logits / temperature           │ │   │
│  │  │ 2. Apply top-p (nucleus) sampling                             │ │   │
│  │  │ 3. Apply top-k filtering                                      │ │   │
│  │  │ 4. Sample from distribution: next_token = sample(softmax)     │ │   │
│  │  │                                                               │ │   │
│  │  │ Time: ~0.1ms (very fast, small tensor operations)             │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  Output: next_token_id = 15043  (e.g., "Quantum")                  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Streaming Response (GPU → CPU → Network):                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  GPU HBM          CPU RAM              Network                      │   │
│  │  ┌────────┐       ┌────────┐           ┌────────┐                  │   │
│  │  │ 15043  │ ────► │ 15043  │ ────────► │ SSE    │                  │   │
│  │  └────────┘       └────────┘           │ stream │                  │   │
│  │   4 bytes        "Quantum"             │ to     │                  │   │
│  │                                        │ client │                  │   │
│  │                                        └────────┘                  │   │
│  │                                                                     │   │
│  │  Total latency per token: ~20ms (dominated by model forward)        │   │
│  │  Data movement: negligible (4 bytes per token)                      │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Execution Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE INFERENCE TIMELINE                              │
│                    (Single request, 50 input + 100 output tokens)           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Time (ms)                                                                  │
│  0         5        10        20                         2000               │
│  │         │         │         │                           │                │
│  ▼         ▼         ▼         ▼                           ▼                │
│                                                                             │
│  ┌─┐ Tokenize (0.1ms)                                                       │
│  └─┘                                                                        │
│    ┌─┐ Cache lookup (0.01ms)                                                │
│    └─┘                                                                      │
│      ┌─┐ Transfer to GPU (0.05ms)                                           │
│      └─┘                                                                    │
│        ┌────────────────┐                                                   │
│        │    PREFILL     │ 16ms (20 new tokens, 30 cached)                   │
│        │  (compute GPU) │                                                   │
│        └────────────────┘                                                   │
│                         │                                                   │
│                         ▼ TTFT = ~16ms                                      │
│                         ┌─┐ Token 51: "Quantum"                             │
│                         └─┘                                                 │
│                           ┌─┐ Token 52: " computing"                        │
│                           └─┘                                               │
│                             ┌─┐ Token 53: " is"                             │
│                             └─┘                                             │
│                               │                                             │
│                               │  ... 97 more tokens ...                     │
│                               │  (~20ms each with batching)                 │
│                               │                                             │
│                               │                                             │
│                               ┌─┐ Token 150: "<EOS>"                        │
│                               └─┘                                           │
│                                 │                                           │
│                                 ▼                                           │
│                               Total: ~2000ms                                │
│                                                                             │
│  Breakdown:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • TTFT (Time to First Token): 16ms                                  │   │
│  │ • Decode (100 tokens @ ~20ms each): 2000ms                          │   │
│  │ • Total: 2016ms                                                     │   │
│  │ • Throughput: 100 tokens / 2s = 50 tokens/sec (single request)      │   │
│  │                                                                     │   │
│  │ With 64-request batching:                                           │   │
│  │ • Throughput: 64 × 50 = 3200 tokens/sec                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Comparison: SGLang vs Naive PyTorch vs vLLM

### Naive PyTorch (No Inference Engine)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NAIVE PYTORCH SERVING                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  What you write:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ model = AutoModelForCausalLM.from_pretrained("llama-70b")           │   │
│  │ while True:                                                         │   │
│  │     request = receive_request()                                     │   │
│  │     inputs = tokenizer(request.prompt, return_tensors="pt")         │   │
│  │     outputs = model.generate(inputs, max_new_tokens=100)  # NAIVE   │   │
│  │     send_response(tokenizer.decode(outputs))                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  What actually happens:                                                     │
│                                                                             │
│  Request 1: "You are helpful. What is Python?"                              │
│  Request 2: "You are helpful. What is Java?"                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Request 1:                                                         │   │
│  │  [Full prefill: 16 tokens] ──────────────────────► 40ms            │   │
│  │  [Decode 100 tokens, ONE AT A TIME] ─────────────► 2000ms          │   │
│  │                                                                     │   │
│  │  Request 2 (WAITS for Request 1!):                                  │   │
│  │  [Full prefill: 16 tokens] ──────────────────────► 40ms            │   │
│  │  [Decode 100 tokens, ONE AT A TIME] ─────────────► 2000ms          │   │
│  │                                                                     │   │
│  │  Total: 4080ms for 2 requests (sequential!)                        │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Problems:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ❌ No batching: 1 request at a time                                 │   │
│  │ ❌ No KV cache reuse: Recompute prefill every time                  │   │
│  │ ❌ No continuous batching: Requests wait in queue                   │   │
│  │ ❌ Memory fragmentation: PyTorch allocates/frees constantly         │   │
│  │ ❌ No paging: Can OOM on long sequences                             │   │
│  │ ❌ No prefix caching: Repeated prompts = repeated compute           │   │
│  │                                                                     │   │
│  │ GPU Utilization: ~5-10% (mostly idle waiting for single request)    │   │
│  │ Throughput: ~25 tokens/sec (1 request at a time)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### vLLM (Paged Attention, Optional APC)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              vLLM                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Key Features:                                                              │
│  ✓ Paged Attention (memory efficiency)                                     │
│  ✓ Continuous Batching                                                     │
│  ✓ Automatic Prefix Caching (APC) - optional                               │
│                                                                             │
│  Request 1: "You are helpful. What is Python?"                              │
│  Request 2: "You are helpful. What is Java?"                                │
│                                                                             │
│  WITHOUT APC (--enable-prefix-caching=false):                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Both requests run in parallel (continuous batching):               │   │
│  │                                                                     │   │
│  │  Req 1: [Prefill 16] [Decode batch of 2] [Decode] ...              │   │
│  │  Req 2: [Prefill 16] [Decode batch of 2] [Decode] ...              │   │
│  │         ↑            ↑                                              │   │
│  │         Each request computes its own prefill                       │   │
│  │         (no reuse of "You are helpful.")                            │   │
│  │                                                                     │   │
│  │  Time: ~1100ms for both (batched decode helps)                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  WITH APC (--enable-prefix-caching=true):                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Block-based prefix matching:                                       │   │
│  │                                                                     │   │
│  │  Tokens: [You][are][help][ful.][What][is][Py][thon?]               │   │
│  │  Blocks: [───Block 0───][───Block 1───][Block 2]                   │   │
│  │          (4 tokens each, block_size=4)                              │   │
│  │                                                                     │   │
│  │  Req 1: Compute blocks 0,1,2 → cache by hash                       │   │
│  │  Req 2: hash(block0) HIT, hash(block1) HIT, block2 differs         │   │
│  │         → Reuse 8 tokens, compute 8 new                             │   │
│  │                                                                     │   │
│  │  Limitation: Block boundaries must align                            │   │
│  │  If prefix = 7 tokens → only 4 reused (block 0)                    │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Memory Layout (Paged):                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Physical KV Blocks:                                                │   │
│  │  [Block 0][Block 1][Block 2][Block 3][Block 4] ... [Block N]       │   │
│  │      ↑        ↑        ↑        ↑                                  │   │
│  │      │        │        │        │                                  │   │
│  │  Request Page Tables:                                               │   │
│  │  Req 1: [0] → [1] → [2] → [5]                                      │   │
│  │  Req 2: [0] → [1] → [3] → [6]   (shares blocks 0,1 with Req 1)     │   │
│  │                                                                     │   │
│  │  ✓ No memory fragmentation                                          │   │
│  │  ✓ Efficient memory sharing                                         │   │
│  │  ✗ Hash-based lookup (block granularity only)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### SGLang (Radix Cache)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SGLang                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Key Features:                                                              │
│  ✓ Radix Tree Prefix Cache (token-level matching)                          │
│  ✓ Continuous Batching                                                     │
│  ✓ Cache-Aware Scheduling (LPM policy)                                     │
│  ✓ Paged KV Memory                                                         │
│                                                                             │
│  Request 1: "You are helpful. What is Python?"                              │
│  Request 2: "You are helpful. What is Java?"                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Token-level prefix matching:                                       │   │
│  │                                                                     │   │
│  │  Tokens: [You][are][helpful][.][What][is][Python][?]               │   │
│  │  Prefix:  1    2      3      4   5    6     ↑                      │   │
│  │                                              │                      │   │
│  │                                    Divergence at token 7            │   │
│  │                                                                     │   │
│  │  Req 1: Build tree with full sequence (16 tokens)                  │   │
│  │                                                                     │   │
│  │  Radix Tree After Req 1:                                            │   │
│  │  ROOT → [You are helpful. What is ] → [Python?]                    │   │
│  │              (6 tokens)                  (2 tokens)                 │   │
│  │                                                                     │   │
│  │  Req 2: match_prefix() finds 6 tokens match                        │   │
│  │         → Reuse 6 tokens of KV, compute only 2 new                 │   │
│  │                                                                     │   │
│  │  Tree After Req 2:                                                  │   │
│  │  ROOT → [You are helpful. What is ] ─┬─► [Python?]                 │   │
│  │                                       └─► [Java?]  (NEW)           │   │
│  │                                                                     │   │
│  │  Key difference from vLLM APC:                                      │   │
│  │  • vLLM: Block-aligned (4-token granularity)                       │   │
│  │  • SGLang: Token-aligned (exact match at any boundary)             │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Cache-Aware Scheduling:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Waiting Queue:                                                     │   │
│  │  ┌────────────────────────────────────────────────────────────┐    │   │
│  │  │ Req A: "System prompt..." (1000 tokens) → 0 cached          │   │   │
│  │  │ Req B: "System prompt..." (1000 tokens) → 0 cached          │   │   │
│  │  │ Req C: "System prompt... Q1" (1020 tokens) → 1000 cached!   │   │   │
│  │  └────────────────────────────────────────────────────────────┘    │   │
│  │                                                                     │   │
│  │  LPM Policy: Schedule Req C first (longest prefix match)           │   │
│  │  → Req C needs only 20 tokens prefill (fast!)                      │   │
│  │  → Then schedule Req A (builds cache for Req B to use)             │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Side-by-Side Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPARISON: Same Workload                                │
│         (100 requests with 80% shared prefix, 50 tokens each)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                  Naive PyTorch    vLLM (no APC)   vLLM (APC)     SGLang     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Prefill tokens    5,000          5,000           ~2,000         ~1,000     │
│  computed          (100×50)       (100×50)        (block-align)  (exact)    │
│                                                                             │
│  TTFT (first req)  40ms           40ms            25ms           16ms       │
│                                                                             │
│  TTFT (100th req)  40ms           40ms            25ms           5ms        │
│                    (no cache)     (no cache)      (partial)      (90% hit)  │
│                                                                             │
│  GPU Utilization   5-10%          60-80%          60-80%         70-85%     │
│                                                                             │
│  Throughput        25 tok/s       2,500 tok/s     2,800 tok/s    3,500 tok/s│
│  (tokens/sec)                                                               │
│                                                                             │
│  Memory Efficiency ❌              ✓✓              ✓✓             ✓✓         │
│  (paging)          (fragments)    (paged)         (paged)        (paged)    │
│                                                                             │
│  Prefix Reuse      ❌              ❌              ✓              ✓✓         │
│                    (none)         (none)          (block-level)  (token)    │
│                                                                             │
│  Cache Scheduling  ❌              ❌              ❌              ✓          │
│                    (none)         (FCFS)          (FCFS)         (LPM)      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WHEN TO USE EACH:                                                          │
│                                                                             │
│  Naive PyTorch:    Prototyping, single-user demos only                      │
│                                                                             │
│  vLLM (no APC):    Diverse prompts, batch processing, simplicity            │
│                                                                             │
│  vLLM (APC):       Moderate prefix sharing, want simpler deployment         │
│                                                                             │
│  SGLang:           Chat, agents, heavy prefix sharing, latency-critical     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Hardware Bottlenecks Removed by SGLang

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BOTTLENECKS AND SOLUTIONS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BOTTLENECK 1: Repeated Prefill Compute                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Problem: Same prefix computed over and over                         │   │
│  │ Hardware waste: Tensor Cores doing redundant FLOPs                  │   │
│  │                                                                     │   │
│  │ Solution: Radix Cache                                               │   │
│  │ • Store KV at token granularity                                     │   │
│  │ • Match any prefix length                                           │   │
│  │ • Skip prefill for cached tokens                                    │   │
│  │ • Result: 3-5x TTFT reduction on shared workloads                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  BOTTLENECK 2: Memory Bandwidth During Decode                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Problem: Each token reads 140GB of weights, 1 FLOP/byte             │   │
│  │ Hardware waste: HBM at 8TB/s, Tensor Cores at <1% utilization       │   │
│  │                                                                     │   │
│  │ Solution: Continuous Batching                                       │   │
│  │ • Batch 64+ requests together                                       │   │
│  │ • Read weights once, apply to all requests                          │   │
│  │ • Result: 64 FLOPS/byte → near compute-bound                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  BOTTLENECK 3: Memory Fragmentation                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Problem: Variable-length sequences cause fragmentation             │   │
│  │ Hardware waste: HBM has gaps, OOM with 50% memory free              │   │
│  │                                                                     │   │
│  │ Solution: Paged KV Cache                                            │   │
│  │ • Fixed-size blocks, virtual-to-physical mapping                    │   │
│  │ • No fragmentation, near-100% memory utilization                    │   │
│  │ • Result: 2-4x more concurrent sequences                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  BOTTLENECK 4: Suboptimal Scheduling                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Problem: FCFS ignores cache state, misses reuse opportunities       │   │
│  │ Hardware waste: Prefill for requests that could have hit cache      │   │
│  │                                                                     │   │
│  │ Solution: Cache-Aware Scheduling (LPM)                              │   │
│  │ • Sort requests by prefix match length                              │   │
│  │ • Schedule cache-hitting requests first                             │   │
│  │ • Result: Higher cache hit rate, lower average latency              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  BOTTLENECK 5: Head-of-Line Blocking                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Problem: Short requests wait for long ones in static batching       │   │
│  │ Hardware waste: GPU idle between batches, request queuing           │   │
│  │                                                                     │   │
│  │ Solution: Continuous Batching (Iteration-Level)                     │   │
│  │ • Requests join/leave batch at any decode step                      │   │
│  │ • No waiting for batch completion                                   │   │
│  │ • Result: Near-constant GPU utilization                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Trade-Offs That Remain

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REMAINING TRADE-OFFS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Radix Tree Overhead                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Cost: CPU memory for tree nodes, O(prefix_len) lookup time          │   │
│  │ When it hurts: Diverse prompts with no shared prefixes              │   │
│  │ Mitigation: Falls back to FCFS at >128 queue, use page_size=16      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  2. Cache Memory Pressure                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Cost: Cached KV consumes GPU memory                                 │   │
│  │ When it hurts: More cache = fewer concurrent requests               │   │
│  │ Mitigation: LRU eviction, HiCache (CPU/storage offload)             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  3. Eviction Latency                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Cost: Evicting cached KV has CPU overhead                           │   │
│  │ When it hurts: High churn workloads with constant eviction          │   │
│  │ Mitigation: Priority-based eviction, larger cache                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  4. Batching Latency vs Throughput                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Cost: Waiting to fill batch increases latency                       │   │
│  │ When it hurts: Low request rate, latency-critical apps              │   │
│  │ Mitigation: Tune batch timeout, use smaller max batch size          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  5. Memory Bandwidth Still the Limit                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Reality: Even with all optimizations, decode is memory-bound        │   │
│  │ B200 at 8TB/s, 140GB model = 17ms minimum per token at batch=1      │   │
│  │ Solution: Batch larger, use quantization (FP8 halves memory read)   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary: Why SGLang Exists

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  THE CORE INSIGHT:                                                          │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  LLM inference has TWO bottlenecks:                                         │
│                                                                             │
│  1. PREFILL (compute-bound):                                                │
│     • Tensor Cores are the limit                                            │
│     • Solution: Don't repeat work → RADIX CACHE                             │
│                                                                             │
│  2. DECODE (memory-bound):                                                  │
│     • HBM bandwidth is the limit                                            │
│     • Solution: Batch requests → CONTINUOUS BATCHING                        │
│                                                                             │
│  SGLang solves BOTH with a unified system that:                             │
│  • Caches KV at token granularity                                           │
│  • Matches prefixes at any boundary                                         │
│  • Schedules cache-aware                                                    │
│  • Batches dynamically                                                      │
│                                                                             │
│  Result: 3-5x better TTFT + 2-3x better throughput on real workloads        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start Checklist (Your Week 1 Plan)

The checklist you provided is excellent for onboarding:

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
