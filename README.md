# 505M GPT Training Pipeline
## Comprehensive Technical Report (Kaggle 2×T4 DDP Edition)

---

## 1. Executive Summary

505M GPT is a decoder-only GPT language model at **~505M parameters**, trained on the `open-web-math/open-web-math` corpus, streamed live from Hugging Face rather than downloaded to disk. The pipeline runs on Kaggle's dual-T4 (2×15.6GB VRAM) instances using PyTorch's `DistributedDataParallel` (DDP), launched via `torchrun`, and is designed to be checkpoint-compatible with a parallel single-GPU Colab baseline notebook. It is architected as a stepping stone toward multi-node EC2 training.

The system is built around three constraints that shape almost every design decision below:
- **VRAM scarcity** — T4s have no flash-attention-v2 kernel and only 15.6GB each, so the model sits close to the OOM edge.
- **Ephemeral compute** — Kaggle sessions are time-boxed and `/kaggle/working` doesn't survive indefinitely, so checkpoint durability requires pushing off-box.
- **Streaming data** — the corpus is never fully materialized locally; it's pulled and tokenized on the fly.

---

## 2. Model Architecture

### 2.1 High-level specification

| Property | Value |
|---|---|
| Architecture | Decoder-only GPT (GPT-2 style) |
| Total parameters | ~505.1M |
| Layers | 28 |
| Embedding dimension | 1152 |
| Attention heads | 16 (head_dim = 72) |
| Context length (block_size) | 1024 |
| Vocabulary | 50,257 (GPT-2 tokenizer, via `tiktoken`) |
| FFN expansion | 4× (1152 → 4608 → 1152) |
| Positional encoding | Learned absolute position embeddings |
| Weight tying | Token embedding ↔ output head (shared weight matrix) |

### 2.2 Parameter accounting

```
Token embedding    :  50257 × 1152  =   57.9M  (tied with lm_head, not double-counted)
Position embedding :   1024 × 1152  =    1.2M
Per transformer block (×28):
  CausalSelfAttn  :  4 × 1152²      =    5.31M   (qkv proj + out proj, no bias)
  FeedForward     :  8 × 1152²      =   10.62M   (up-proj + down-proj, no bias)
  LayerNorms ×2   :  4 × 1152       =    4,608
28 blocks total                     =  447.0M
Final LayerNorm                     =    2,304
──────────────────────────────────────────────
TOTAL                                = ~505.1M
```

`num_params()` deliberately excludes `head.weight` from the count since it is the same tensor as `tok_emb.weight` (weight tying) — counting it would double-count 57.9M parameters.

### 2.3 Block internals

Each `TransformerBlock` is a standard pre-norm residual design:

```
x = x + Attn(LN1(x))
x = x + FFN(LN2(x))
```

**Attention (`CausalSelfAttention`)**
- Single fused `c_attn` linear projects to Q, K, V simultaneously (`3 × n_embd` output, no bias).
- Heads are split via `view` + `transpose`, standard multi-head layout.
- Uses `torch.nn.functional.scaled_dot_product_attention` (SDPA) with `is_causal=True` rather than a hand-written attention + causal-mask implementation — lets PyTorch dispatch to a fused, memory-efficient kernel.
- Supports an optional KV-cache path (`use_cache`) for future inference/generation use, concatenating past K/V along the sequence dimension. Not exercised during training.
- Output projection `c_proj` uses a **scaled initialization** (see §2.4) since it feeds directly into a residual stream.

**Feed-forward (`FeedForward`)**
- Two linear layers (no bias) with GELU activation in between, standard 4× expansion.
- Dropout is currently configured at `0.0` (disabled) — the model relies on the corpus scale and cosine LR decay rather than dropout regularization.

**Gradient checkpointing** is applied *per block* during training:
```python
if self.training and not use_cache:
    return grad_ckpt.checkpoint(self._block_fn, x, use_reentrant=False)
```
This trades compute for memory — activations for each block are recomputed on the backward pass rather than stored, which is essential to fit a 505M model with a 1024-token context on a 15.6GB card. It is only active in `training` mode; inference/generation paths bypass it.

### 2.4 Initialization

- Linear layers: `N(0, 0.02²)`.
- Residual-stream-facing projections (`c_proj`, and the FFN's second linear, matched by name suffix `net.2`) use a **depth-scaled std**: `0.02 / sqrt(2 × n_layer)`. This is the GPT-2 initialization scheme — it prevents the residual stream's variance from growing unboundedly as depth increases, since each block adds a contribution to the same residual path.
- Embeddings: `N(0, 0.02²)`.
- LayerNorm: weight = 1, bias = 0 (identity at init).
- All linear layers are bias-free (`bias=False`) throughout — a common efficiency choice for large transformers since LayerNorm largely absorbs the need for bias terms.

### 2.5 Attention backend pinning (T4-specific)

T4 GPUs (Turing, `sm_75`) have no flash-attention-v2 kernel. Left to PyTorch's automatic SDPA backend dispatch, the runtime can silently fall back to the naive **"math" backend**, which materializes a full T×T attention matrix per head — a large, avoidable VRAM cost. The script forces the **memory-efficient backend** explicitly:

```python
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    y = F.scaled_dot_product_attention(...)
```
with a fallback path for older PyTorch versions using `torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)`.

---

## 3. Training Configuration

| Hyperparameter | Value | Notes |
|---|---|---|
| `micro_batch_size` | 8 | per-GPU, per-accumulation-step |
| `grad_accum_steps` | 8 | halved vs. the single-GPU Colab config |
| `WORLD_SIZE` | 2 | one process per GPU |
| **Effective batch (tokens/step)** | **131,072** | `WORLD_SIZE × micro_batch × grad_accum × block_size` = 2×8×8×1024 |
| `lr` (peak) | 3e-4 | |
| `weight_decay` | 0.1 | applied only to 2D+ parameters |
| `grad_clip` | 1.0 | global L2 norm clipping |
| `warmup_steps` | 1000 | linear warmup |
| `num_epochs` | 2 | |
| `eval_interval` | 100 steps | also the checkpoint cadence |
| `eval_batches` | 50 | per evaluation |

### 3.1 Learning rate schedule

Linear warmup followed by cosine decay to a floor, not to zero:

```python
if step < warmup_steps:
    lr = base_lr * (step + 1) / warmup_steps
else:
    progress = (step - warmup_steps) / (train_steps - warmup_steps)
    lr = base_lr * (0.1 + 0.9 * 0.5 * (1 + cos(pi * progress)))
```
The `0.1 + 0.9 × ...` term means LR decays to **10% of peak**, not 0 — a common choice to avoid the model going fully "cold" at the end of training, which can hurt loss on the final steps.

### 3.2 Critical DDP batch-size compensation

This is flagged in the notebook as a hard-won fix, and is worth stating precisely because it's easy to silently get wrong: in DDP, **every rank runs its own gradient accumulation loop independently**, and gradients are averaged (not summed) across ranks via NCCL all-reduce. So the effective tokens-per-optimizer-step is:

```
tokens/step = WORLD_SIZE × micro_batch_size × grad_accum_steps × block_size
```

If `micro_batch_size` and `grad_accum_steps` are copied unchanged from a single-GPU config into a 2-GPU DDP config, the effective batch silently doubles — which desyncs the LR schedule, warmup, eval cadence, and step-for-step checkpoint comparability against the single-GPU baseline, even though total tokens over the full run still eventually converge. The fix here was to halve both `micro_batch_size` (16→8) and `grad_accum_steps` (64→8) so 2×T4 DDP matches the single-GPU Colab's tokens/step exactly.

### 3.3 Gradient accumulation & sync mechanics

Within one optimizer step, only the **last** micro-step of the accumulation loop triggers a gradient all-reduce; all earlier micro-steps run inside `model.no_sync()`:

```python
ctx = model.no_sync() if not is_last_micro else contextlib.nullcontext()
with ctx:
    _, loss = model(x, y)
    loss = loss / grad_accum_steps
    scaler.scale(loss).backward()
```
This avoids `grad_accum_steps × WORLD_SIZE` redundant NCCL all-reduces per optimizer step — instead of syncing gradients after every micro-batch, DDP accumulates local gradients across all micro-steps and only communicates once per optimizer step. This is a standard and important DDP performance pattern; without it, communication overhead scales linearly with accumulation depth.

### 3.4 Mixed precision

- `torch.amp.autocast("cuda")` wraps the forward pass (both training and eval), running most ops in fp16/bf16 while keeping master weights and the loss scale in fp32.
- `torch.amp.GradScaler` handles loss scaling to prevent fp16 gradient underflow, with `scaler.unscale_()` called before gradient clipping (clipping must operate on true-scale gradients, not scaled ones) and `scaler.step()` / `scaler.update()` completing the step.
- `torch.backends.cuda.matmul.allow_tf32 = True` and the cuDNN equivalent enable TF32 for matmuls on Ampere+ hardware — a no-op precision-wise degradation on Turing (T4), but harmless to leave set for portability to future Ampere-class training targets.

### 3.5 Precision deep-dive: what's actually fp16 vs. fp32, and where

Worth being precise here, since "mixed precision" is often described loosely as "some layers are fp16, some are fp32" — that's not quite what's happening. Nothing in the model is ever explicitly cast with `.half()`; `raw_model.to(device)` moves parameters to the GPU but leaves every parameter — token embedding, position embedding, every attention and FFN weight, every LayerNorm weight/bias, the tied output head — stored as **fp32 master weights**, permanently. What varies by op is the **transient compute dtype**, decided dynamically by autocast's per-op whitelist on every forward pass.

**Op-by-op trace through one transformer block, under autocast:**

| Stage | Compute dtype | Why |
|---|---|---|
| `tok_emb` / `pos_emb` lookup | fp32 | Embedding lookup is a gather, not a matmul — not on autocast's fp16 list, so it runs in the weight's storage dtype (fp32) |
| `x = tok_emb + pos_emb` (+ dropout) | fp32 | Sum of two fp32 tensors |
| `ln1(x)` | **fp32 (forced)** | `layer_norm` is explicitly pinned to fp32 in autocast's op table — normalization statistics are numerically sensitive; a PyTorch default, not a project-specific choice |
| `c_attn` (QKV projection) | **fp16** | `Linear`/`addmm` is on autocast's fp16 whitelist — input is cast down, the fp32 weight is cast down on the fly for this op only (the master weight itself is untouched), matmul executes in fp16 |
| `scaled_dot_product_attention` | fp16 in/out, **fp32 internal accumulation** | Q/K/V enter as fp16. The QK^T score matmul, softmax, and the weighted-sum-over-V accumulation are computed internally by the memory-efficient attention CUDA kernel using fp32 accumulators — a property of the fused kernel implementation itself (necessary to avoid softmax overflow and accumulated rounding error across the sequence dimension), not something visible or configurable at the PyTorch op level. The kernel's *output* is cast back down to fp16. |
| `c_proj` (output projection) | fp16 | Same as `c_attn` — on the fp16 whitelist |
| `x = x + attn_out` (residual add) | **fp32** | `x` (the running residual stream) is fp32; `attn_out` is fp16. Standard PyTorch dtype promotion on a mixed-dtype add always upcasts to the wider type, so this addition silently produces an fp32 result. **This is the single most important precision detail in the model**: the residual stream is never actually stored or accumulated in fp16, at any point, in any block. |
| `ln2(x)` | **fp32 (forced)** | Same as `ln1` |
| FFN linears + GELU | fp16 (linears); GELU follows its input (fp16), since GELU isn't on either autocast whitelist | Linears on the fp16 whitelist |
| `x = x + ffn_out` | **fp32** | Same promotion behavior as the attention residual add |
| `ln_final(x)` | **fp32 (forced)** | Same as above |
| `head` (tied lm_head projection) | fp16 | On the fp16 whitelist — `ln_final`'s fp32 output is cast down for this matmul |
| `F.cross_entropy(logits, targets)` | **fp32 (forced)** | Explicitly pinned to fp32 in autocast's op table — autocast automatically upcasts the fp16 logits before computing it, avoiding `log_softmax` over fp16 logits, a classic source of numerical instability (overflow in `exp`, precision loss in log-sum-exp) in naive fp16 training |

**The accurate mental model:** every matmul-heavy op (the four linear projections per block, plus the tied head) transiently computes in fp16 for throughput and memory bandwidth; every normalization and the loss stay pinned to fp32 by autocast's built-in policy; and — the detail that's easy to miss — **the residual stream itself never drops below fp32**, purely as a side effect of dtype promotion on the `x = x + sublayer(x)` additions, not because of any explicit precision policy written into this code. The parameters backing the fp16 matmuls are fp32 the entire time; autocast never mutates stored weights, it only casts activations/weight-views at the point of use.

**Gradients and backward:** autograd's convention is that a parameter's `.grad` always matches the parameter's own dtype — so even though the forward matmuls for `c_attn`/`c_proj`/FFN/`head` ran in fp16, the gradients landing on those fp32 parameters are themselves fp32 (upcast automatically during backward). `GradScaler` exists specifically to protect the *fp16-valued intermediate* gradients computed while backpropagating through the fp16 matmul ops — before that upcast happens — from flushing to zero due to fp16's narrow representable range near small magnitudes. `scaler.scale(loss)` multiplies the loss by a large factor before backward so those intermediate gradients land in a representable fp16 range; `scaler.unscale_()` divides them back down before clipping and the optimizer step see them.

**A separate, unrelated precision axis — optimizer state:** the 8-bit paged AdamW's quantization (§4) is *not* part of this autocast picture at all. It compresses the optimizer's fp32 first/second moment buffers (`exp_avg`, `exp_avg_sq`) into int8 with per-block scaling factors for storage, dequantizing back to fp32 on the fly during each update step. This is orthogonal to — and stacks with — the fp16/fp32 autocast split above: one governs forward/backward compute precision, the other governs how much VRAM the optimizer's bookkeeping state consumes at rest. **Note: the model weights themselves are never quantized by this — they remain fp32 on disk in every checkpoint (§8.1). Only the AdamW optimizer's internal moment buffers are int8.**

### 3.6 Steps-per-epoch accounting

```python
STEPS_PER_EPOCH = (DOCS_PER_EPOCH * 350) // (block_size + 1) // (
    WORLD_SIZE * micro_batch_size * grad_accum_steps
)
```
`DOCS_PER_EPOCH` (6.3M) and an assumed ~350 tokens/doc give a rough total token estimate for the corpus slice; dividing by tokens-per-step gives the epoch length in optimizer steps. This formula was previously missing the `WORLD_SIZE × micro_batch_size` factor in the divisor — undercounting tokens/step by that factor and causing the run to execute exactly 2× too many steps per epoch on 2×T4. Fixed in the current version.

---

## 4. Optimizer

**Primary: `bitsandbytes` 8-bit paged AdamW** (`PagedAdamW8bit`, falling back to plain `AdamW8bit`, then to fp32 `torch.optim.AdamW` with `fused=True` if `bitsandbytes` isn't installed).

- Standard AdamW keeps two fp32 moment buffers (`exp_avg`, `exp_avg_sq`) per parameter — for 505M params, that's roughly **4GB of optimizer state alone**, a large fraction of a T4's 15.6GB budget before any activations or weights are counted.
- 8-bit quantized optimizer state cuts this to **~1GB**. This is optimizer-state quantization only — the model's own weights stay fp32 throughout (see §3.5).
- The **paged** variant additionally spills optimizer state to CPU pinned memory under GPU memory pressure instead of raising an OOM — a real safety margin on hardware operating this close to its VRAM ceiling.
- Parameters are split into two groups: `weight_decay=0.1` for all 2D+ tensors (linear/embedding weights) and `weight_decay=0.0` for 1D tensors (LayerNorm weights/biases) — the standard practice of not decaying norm and bias parameters.
- Betas: `(0.9, 0.95)` — the lower β2 (vs. the more common 0.999) is a common LLM-training choice that makes the second-moment estimate more responsive, often paired with aggressive LR schedules and gradient clipping like this config uses.

---

## 5. Distributed Systems Architecture (DDP)

### 5.1 Process topology

- `torchrun --nproc_per_node=2` spawns **2 independent processes**, one per GPU, each running the full `train_ddp.py` script.
- Each process reads its `LOCAL_RANK` from the environment, pins itself to that CUDA device (`torch.cuda.set_device(LOCAL_RANK)`), and joins an NCCL process group (`dist.init_process_group(backend="nccl")`) — NCCL is the standard high-bandwidth GPU-to-GPU collective communication backend.
- `IS_MASTER = (LOCAL_RANK == 0)` gates all singleton responsibilities: printing, checkpoint saving, telemetry writing, and the Kaggle registry push. This avoids both duplicated console output and two processes racing to write the same checkpoint file.

### 5.2 Ordering constraint: allocator config before process-group init

`PYTORCH_CUDA_ALLOC_CONF` is set **before** `dist.init_process_group()` and before any CUDA call. This ordering matters: NCCL initialization creates each process's CUDA context, and the caching allocator reads its config lazily on first CUDA use — setting the env var after that point is silently ignored. The config used is `expandable_segments:True` (plus `max_split_size_mb:128` mentioned in the notebook intro), both allocator-fragmentation mitigations relevant to a long-running process pushing close to its VRAM ceiling.

### 5.3 Model wrapping order

`DDP(raw_model, device_ids=[LOCAL_RANK], find_unused_parameters=False)` wraps the model **after** weights are loaded from checkpoint (or freshly initialized) — not before. Wrapping first would mean checkpoint loading has to reconcile the `module.` prefix DDP adds to every parameter name; loading into the raw model and wrapping afterward keeps checkpoint files DDP-agnostic and cross-compatible with the single-GPU Colab notebook, which never uses DDP at all. `find_unused_parameters=False` is set because every parameter participates in every forward pass here (no conditional branches that skip parameters), which lets DDP skip the more expensive unused-parameter traversal.

### 5.4 Data sharding across ranks

Rather than a shared sampler, each rank runs its **own independent stream** from the same Hugging Face dataset, shuffled with a **rank-dependent seed**:

```python
ds = ds.shuffle(seed=42 + epoch + LOCAL_RANK * 1000, buffer_size=10_000)
```
This ensures the two GPUs see different documents each step (avoiding redundant computation on identical data) without needing a coordinated distributed sampler over a streaming, unbounded-length dataset — which would be considerably more complex to implement correctly.

### 5.5 Synchronization points

- **Gradient sync**: one NCCL all-reduce per optimizer step (see §3.3), not per micro-step.
- **Validation**: only rank 0 runs the eval loop (`estimate_loss`), then broadcasts the scalar loss to all ranks via `dist.broadcast`, so every rank has a consistent value even though only one rank did the compute. This avoids redundant eval work across ranks.
- **`dist.barrier()`** after each eval/checkpoint cycle ensures all ranks reach the same point before resuming training — preventing a fast rank from racing ahead while rank 0 is busy writing a multi-GB checkpoint to disk.
- **`dist.destroy_process_group()`** at the very end for clean NCCL teardown.

### 5.6 MFU (Model FLOPs Utilization) scaling

```python
T4_PEAK_FLOPS_TOTAL = 65e12 * WORLD_SIZE
mfu = 100.0 * (tokens_per_sec * 6 * num_params) / T4_PEAK_FLOPS_TOTAL
```
The peak-FLOPs denominator is scaled by `WORLD_SIZE` so the reported MFU is comparable to the single-GPU Colab's MFU number — both express "% of all attached T4s' combined peak," not per-GPU peak. The `6 × num_params` term is the standard forward+backward FLOPs-per-token approximation (2× for forward, 4× for backward) used throughout the LLM training literature (e.g. the Chinchilla/PaLM compute estimates).

---

## 6. Memory Engineering (T4 VRAM Constraints)

T4s carry 15.6GB VRAM each with no flash-attention-v2 support, so most of the "systems" work in this pipeline is defensive VRAM management. The stacked mitigations:

| Technique | Effect |
|---|---|
| 8-bit paged AdamW | Optimizer state ~4GB → ~1GB |
| Gradient checkpointing (per block) | Trades recompute for activation memory |
| `micro_batch 16→8, grad_accum 64→8` (historical tuning) | Same effective batch, lower peak per step |
| SDPA backend pinned to memory-efficient | Avoids silent fallback to VRAM-heavy "math" attention |
| `torch.cuda.empty_cache()` | Called after optimizer-state load, after DDP wrap, and every 50 steps |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, max_split_size_mb:128` | Fights allocator fragmentation over long runs |
| Bias-free linear layers | Marginal but consistent parameter/activation reduction |

### 6.1 Per-rank VRAM telemetry (console)

`_log_vram(label)` prints `torch.cuda.memory_allocated()` and `memory_reserved()` **from every rank individually** (not just rank 0), at key lifecycle points: after `model.to(device)`, after checkpoint resume, after the DDP wrap, after the first optimizer step, and every 50 steps thereafter. This per-rank granularity is deliberate — an OOM can hit either T4 independently (e.g. due to slightly different data-shard token distributions per rank), and a single combined/averaged number would mask which specific GPU is at risk. This is distinct from the CSV telemetry in §9, which averages VRAM/utilization across GPUs rather than logging per-rank — see §9 for why.

### 6.2 Data-throughput-side memory hygiene

- The `TokenBuffer` class incrementally accumulates a token buffer via a generator (`_generate()`), only ever holding `block_size + 1` tokens at a time before yielding a chunk and trimming the buffer — never materializing a full document's tokenization or the full dataset in memory.
- `gc.collect()` + `torch.cuda.empty_cache()` are called explicitly around expensive operations (model init, checkpoint load) to force cleanup rather than relying purely on Python's reference-counting GC timing.

---

## 7. Data Pipeline

### 7.1 Source and streaming

- Dataset: `open-web-math/open-web-math`, loaded via Hugging Face `datasets.load_dataset(..., streaming=True)` — the corpus is never downloaded in full; documents are pulled on demand as the iterator advances.
- `cache_dir` points at `/tmp/HF_Cache`, explicitly ephemeral (fine for streaming — nothing needs to persist there across sessions).
- Tokenizer: `tiktoken`'s GPT-2 encoding (`enc.encode_ordinary`), with an end-of-text (`EOT`) token appended after every document to mark document boundaries in the token stream.

### 7.2 TokenBuffer — streaming tokenization into fixed-length chunks

```python
class TokenBuffer:
    def _generate(self):
        for doc in self._iter:
            tokens = enc.encode_ordinary(doc["text"]) + [EOT]
            self._buf.extend(tokens)
            while len(self._buf) >= self._block_size + 1:
                chunk = self._buf[: self._block_size + 1]
                self._buf = self._buf[self._block_size + 1:]
                yield chunk
```
Documents of arbitrary length are tokenized and appended to a running buffer; whenever the buffer holds enough tokens for a full `block_size + 1` chunk (the `+1` because inputs and targets are the same window offset by one position), a chunk is sliced off and yielded, and the remainder stays in the buffer for the next document to top up. This means document boundaries don't need to align with training-chunk boundaries — chunks can span the end of one document and the start of the next (separated internally by the `EOT` token), maximizing token utilization from the stream.

### 7.3 Resume-aware skip logic

On resume, each rank fast-forwards its stream past documents already consumed in the current epoch:
```python
train_stream_iter = make_stream(
    epoch=start_epoch,
    skip_docs=start_epoch_step * grad_accum_steps * micro_batch_size,
)
```
`ds.skip(skip_docs)` is applied to the streaming dataset iterator before shuffling consumption begins, so a resumed run doesn't re-train on already-seen documents within the same epoch (though exact token-level resumption isn't guaranteed, since `TokenBuffer` chunk boundaries don't map 1:1 to document counts — this is an approximate, document-granularity resume, not a byte-exact one).

### 7.4 Validation set

Unlike training data, the validation set is **not streamed per-step** — it's materialized once upfront (`make_val_tensors`, 500 documents) into a single contiguous tensor of tokens, built only on rank 0. Each eval draws `eval_batches` random windows from this fixed tensor via `torch.randint`, giving a stable, comparable validation signal across evaluations rather than a fresh (and non-comparable) streaming sample every time.

---

## 8. Checkpointing & Persistence

### 8.1 What's saved

Each checkpoint (`ckpt_{step:06d}.pt`) contains: `step`, `epoch`, `epoch_step`, raw (DDP-`module.`-prefix-stripped) model state dict, optimizer state dict, GradScaler state dict, the full `cfg` dataclass, and current losses — saved by rank 0 only, at every `eval_interval`. The model state dict is fp32 in every checkpoint; nothing about this pipeline reduces on-disk model precision (see §3.5, §4).

### 8.2 Local rotation

After each save, `save_checkpoint` re-globs `CKPT_DIR`, sorts by modification time, and deletes all but the `max_checkpoints` (3) most recent — bounding local disk usage on `/kaggle/working` over a long run.

### 8.3 Selective config restore on resume

`load_checkpoint` copies **only architecture-critical fields** (`vocab_size, block_size, n_layer, n_head, n_embd, dropout`) from the checkpoint's saved config into the live `cfg` object — fields whose values must match the saved weight shapes for `load_state_dict` to succeed. Training hyperparameters (`micro_batch_size`, `grad_accum_steps`, `lr`, `num_epochs`, etc.) are deliberately **left untouched**, so the currently-running script's intended values win. This directly fixes a prior bug where `cfg = ckpt["cfg"]` wholesale-replaced the entire config object on every resume, silently reverting any hyperparameter tuning done between sessions (most notably `grad_accum_steps`) back to whatever was in effect when the checkpoint was originally saved.

### 8.4 Off-box persistence: Kaggle Model registry push

Since `/kaggle/working` doesn't outlive a session in a way that's convenient to rely on, every `PUSH_EVERY_N_EVALS`-th checkpoint (default: every 2nd) is additionally pushed to a Kaggle Model as a new version via the `kaggle` CLI, bundled together with the current `training_stats.csv` so telemetry history travels with the checkpoint that produced it.

**Staging mechanics** — this is the fix that resolved a major performance regression:
- Previously, staging (copying the multi-GB checkpoint file into a temp directory before upload) ran **synchronously on the main training thread**, using `tempfile.mkdtemp()` — which on Kaggle resolves to `/tmp`, a **RAM-backed tmpfs**. A multi-GB `shutil.copy` there competed directly with training for host RAM and could stall the training loop for a very long time under memory pressure — observed as eval intervals blowing out to 4000+ seconds instead of the expected ~70s.
- The fix moves staging entirely **inside the background push thread**, and relocates the staging directory to `/kaggle/working/GPT500M/_push_staging` — the same filesystem as `CKPT_DIR`. Same-filesystem staging means `os.link()` (an instant hardlink, O(1) regardless of file size) can be used instead of a byte-for-byte copy; `shutil.copy` is retained only as a fallback for cross-device edge cases.
- Hardlinking has a subtle correctness benefit too: the staged copy remains valid even if checkpoint rotation (§8.2) deletes the original file mid-upload, since the underlying inode isn't freed until every link to it is gone.

**Async push + graceful shutdown**: `push_checkpoint_async` spawns a daemon thread per push and tracks it in `_push_threads`. At the very end of training, the main process explicitly joins every tracked thread (up to `_PUSH_TIMEOUT_S=900s` each) before exiting — without this, the *last* checkpoint's upload could be silently killed mid-flight when a daemon thread's process exits.

### 8.5 Preflight validation

`_check_kaggle_push_available()` runs once at startup and checks, in order: `KAGGLE_MODEL_HANDLE` is set, the `kaggle` CLI is on `PATH`, `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars are present, and `www.kaggle.com:443` is reachable. If any check fails, push is disabled for the entire session with a one-line printed reason, and training proceeds with local-only checkpoints. This replaces a prior failure mode where a missing credential or CLI would only surface after a subprocess sat for up to an hour on the *first* checkpoint push attempt — discovered at the worst possible time, repeated on every subsequent checkpoint.

### 8.6 Cross-session seeding

A separate notebook cell (run once, before training starts) seeds `CKPT_DIR` and the telemetry CSV from any previously-pushed Kaggle Model artifacts mounted read-only under `/kaggle/input/`, using the same hardlink-safe copy pattern, with an explicit guard rail (`assert not CKPT_DIR.startswith("/kaggle/input")`) to catch the common misconfiguration of accidentally writing back into the read-only input mount.

---

## 9. Telemetry

Every eval interval (rank 0 only), a row is appended to `training_stats.csv` with the schema:

```
step, epoch, train_loss, train_perplexity, val_loss, val_perplexity,
learning_rate, grad_norm, grad_clipping_ratio, weight_norm,
gpu_utilization_pct, vram_allocated_gb, vram_reserved_gb,
mfu_percentage, step_time, tokens_per_sec
```

- **GPU utilization and VRAM are averaged across all GPUs visible to the process**, rather than split into per-GPU columns. This is a deliberate schema-compatibility choice: using the same fieldnames as the single-GPU Colab baseline means a 2-GPU Kaggle run and a 1-GPU Colab run produce directly comparable/mergeable CSVs regardless of `WORLD_SIZE`. In practice this tradeoff has cost little — across the actual runs, utilization has shown no meaningful spread between the two T4s, so the averaged figure tracks each GPU closely. Finer-grained, per-rank VRAM visibility is still available separately via the console logs (§6.1's `_log_vram`), which log individually per rank rather than averaging.
- **GPU utilization** comes from NVML (`pynvml`), queried only by rank 0, which can see both device handles on the box regardless of which GPU it personally owns — guarded so a driver/permissions failure degrades to `None` values rather than interrupting training.
- **VRAM** comes directly from `torch.cuda.memory_allocated()` / `memory_reserved()`, averaged across visible devices — no NVML dependency needed for this metric.
- **Grad norm** is captured as the return value of `clip_grad_norm_()` rather than recomputed separately, since that function already computes the global L2 norm internally as part of clipping. **`grad_clipping_ratio`** tracks the fraction of steps in each eval window where the gradient actually exceeded `grad_clip` and was clipped — a coarse signal for how often clipping is doing real work vs. being a no-op safety net.
- **Weight norm** (`_weight_norm`) computes the global L2 norm across all trainable parameters, useful as a coarse signal for weight growth/divergence over training.
- On resume, `init_stats_csv()` checks that an existing CSV's header matches the current script's schema before appending; a mismatch (e.g. from an older script version with a different column set) triggers archiving the old file and starting fresh, rather than silently misaligning columns.
- If the schema matches (either from the same session or seeded per §8.6), the existing file is left alone and appended to — so telemetry history accumulates across sessions rather than resetting.

---

## 10. Fault Tolerance & Reliability Engineering

Summarizing the defensive patterns spread throughout the pipeline:

- **Guard rails over silent failure**: the `CKPT_DIR` read-only-mount assertion (§8.6), the Kaggle push preflight check (§8.5), and NVML's guarded try/except all follow the same philosophy — fail fast and loud with an actionable message, rather than fail slow and silent (or not at all, degrading quietly).
- **Non-blocking I/O off the hot path**: both checkpoint push staging (§8.4) and the underlying philosophy of gradient-sync batching (§3.3) reflect a consistent rule — anything that isn't strictly required to compute the next optimizer step should not run synchronously on the main training thread.
- **Graceful degradation**: missing `bitsandbytes` → falls back to fp32 AdamW; missing `pynvml` → GPU utilization logs as `None` instead of crashing; missing Kaggle credentials → checkpoints stay local-only instead of blocking training.
- **Exit-time correctness**: explicit thread-joining before process exit (§8.4) prevents the final checkpoint of a run from being silently dropped.

---

## 11. Bug Fix Log (Chronological, from Development History)

| Issue | Root cause | Fix |
|---|---|---|
| Token/step mismatch | `micro_batch_size` and `grad_accum_steps` left unchanged when moving from 1-GPU to 2-GPU DDP, silently doubling effective batch | Halved both: `micro_batch_size` 16→8, `grad_accum_steps` 64→8 — compensating for `WORLD_SIZE` so 2×T4 DDP matches single-GPU Colab's tokens/step exactly |
| Silent config overwrite on resume | `load_checkpoint()` replaced the entire `cfg` object with the pickled checkpoint version | Selectively restore only architecture-critical fields; training hyperparameters stay governed by the current script |
| Checkpoint staging performance bug (~60× step-time spikes) | Synchronous multi-GB `shutil.copy` into RAM-backed `/tmp` on the main training thread | Moved staging into the background push thread, target `/kaggle/working` with `os.link()` hardlinks |
| `STEPS_PER_EPOCH` 2× overcounting | Formula divisor missing `WORLD_SIZE × micro_batch_size` | Added missing terms to the divisor |
| Kaggle push mid-run failures | No upfront validation of CLI/credentials/connectivity | Added one-time preflight check at startup |
| CLI flag error | `kaggle models instances versions create` used `--notes` (invalid) | Corrected to `-n` |

None of these fixes required a training restart from scratch — none touched model weights, optimizer state, or training math directly. The one behavioral side-effect: epoch boundaries land sooner after the `STEPS_PER_EPOCH` correction, since the prior formula was running roughly 2× too many steps per labeled "epoch."

---

## 12. Compatibility & Roadmap

- **Cross-platform checkpoint compatibility**: the DDP `module.` prefix is stripped before saving (`model.module.state_dict()`), so checkpoint `.pt` files load cleanly on both the 2×T4 Kaggle DDP setup and the single-GPU Colab baseline notebook without any translation step.
- **Stated trajectory**: single-GPU Colab (baseline) → dual-T4 Kaggle DDP (current) → multi-node EC2 via PyTorch DDP (future target). The current pipeline's explicit `WORLD_SIZE`-aware scaling (batch-size compensation, MFU denominator, telemetry schema shared across both platforms) is written generally enough to extend to a higher `WORLD_SIZE` across nodes, though multi-node introduces additional concerns not yet addressed here — network-topology-aware NCCL configuration, inter-node bandwidth as a new bottleneck (vs. intra-box NVLink/PCIe), and rendezvous/fault-tolerance across node failures (as opposed to just process failures on one box).

---

## 13. Summary Table — System-Level Design Decisions

| Concern | Decision | Rationale |
|---|---|---|
| Attention backend | Force memory-efficient SDPA | T4 lacks flash-attn-v2; avoid silent fallback to VRAM-heavy math backend |
| Activation memory | Per-block gradient checkpointing | Required to fit 505M params × 1024 context on 15.6GB |
| Optimizer memory | Paged 8-bit AdamW | ~4GB → ~1GB optimizer state, with CPU-spill safety margin; model weights stay fp32 |
| Precision | fp16/bf16 autocast + GradScaler | Throughput + memory, with clip performed on unscaled grads |
| Batch scaling | `micro_batch_size` and `grad_accum_steps` halved for 2×T4 | Keep tokens/step identical across single- and multi-GPU configs |
| Gradient sync | Only on last micro-step (`no_sync()`) | Avoid per-micro-step NCCL overhead |
| Data source | HF streaming dataset, no local materialization | Corpus far exceeds convenient local storage |
| Data sharding | Per-rank shuffle seed, no distributed sampler | Streaming + unbounded dataset makes a coordinated sampler impractical |
| Checkpoint durability | Async push to Kaggle Model registry | `/kaggle/working` doesn't reliably survive past a session |
| Checkpoint staging | Background thread, same-filesystem hardlinks | Avoid competing with training for RAM via `/tmp` tmpfs |
| Config on resume | Selective field restore, not wholesale replace | Prevent silent hyperparameter reversion |
| Telemetry | Averaged GPU/VRAM columns, shared schema with Colab | Cross-platform comparability; per-rank detail still available via console logs |
| Failure handling | Preflight checks, graceful degradation, guarded fallbacks | Fail fast and loud instead of slow and silent |

---

*Report generated from `optimized_telemetry_ipynb.ipynb` (15 cells: setup, environment check, path/registry configuration, embedded `train_ddp.py` training script, script verification, training launch, checkpoint inspection), cross-checked against `train_ddp.py` source and a sample `training_stats.csv` run.*
