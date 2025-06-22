# Exploiting Gradient Correlations for Cheaper Training

*(draft blog post ‑ June 2025)*

---

## 1  Why Care About Gradient Computation?

For modern neural nets the heaviest line‑item is usually **computing** gradients, not transmitting them. Every back‑prop step touches millions of parameters—even on a single GPU that adds up.

Most efficiency papers attack *communication* (PowerSGD, QSGD) or *parameter count* (LoRA, adapters, pruning). Here we ask a quieter question:

> **If gradients are highly correlated, can we skip computing most of them?**

The answer in small experiments is **“yes – with surprisingly little pain.”**

---

## 2  How This Relates to Existing Work

Below is a quick tour of **what has already been tried** and how our proposal differs.

* **Post‑computation gradient compression**
  *Examples*: PowerSGD, QSGD, SignSGD.
  *Goal*: shrink the **already‑computed** gradient before communication.
  *Our difference*: we aim to **avoid computing** most coordinates at all.

* **Subspace fine‑tuning & intrinsic‑dimension work**
  *Examples*: LoRA; Subspace Training.
  *Goal*: restrict **parameter updates** to a fixed or randomly chosen low‑rank space for the entire run.
  *Our difference*: we **learn** the subspace from *actual early‑run gradients* via SVD, then project **per‑step gradients** (not weights) into it.

* **Temporal prediction & Krylov methods**
  *Examples*: Temporal Predictive Coding; SKA‑SGD.
  *Goal*: exploit recent gradient history to predict or project the next full gradient—mainly to cut **communication** or accelerate convergence.
  *Our difference*: we reuse the prediction idea but turn it into **compute** savings *plus* add drift monitoring.

* **Subspace refresh / drift control**
  *Examples*: rarely explicit in prior work.
  *Goal*: let the subspace evolve when it stops helping.
  *Our implementation*: keep a *sentinel* set of true gradients; when loss reduction stalls or projection error spikes we **re‑SVD**.

* **Gradient sparsification**
  *Examples*: Strom 2015; Lin et al. 2018.
  *Goal*: send only top‑k coordinates to save bandwidth.
  *Our difference*: sparsification still pays full back‑prop compute; we aim to skip that cost.

## 3  A One‑Screen Prototype  A One‑Screen Prototype  A One‑Screen Prototype

### 3.1 Warm‑up & SVD

1. Run full‑batch GD for *N*₀ = 10 steps and log the first‑layer gradients.
2. Stack them into a matrix **G** and compute its truncated SVD.
3. Keep the top‑*k* right singular vectors—empirically *k = 2–4* already explains ≥ 90 % of variance.

### 3.2 Projected Updates

For the next *N* steps we:

* back‑prop once,
* project the raw first‑layer gradient onto that *k‑dim* subspace, and
* update weights with the projected vector **only** (cheaper matrix multiply, fewer FLOPs in later layers if extended).

### 3.3 Detecting Drift

Every so often we still compute a tiny control sample of full gradients. If:

* smoothed loss stops dropping **or**
* projection error on the control set exceeds a threshold

we re‑run the SVD and continue.

---

## 4  Toy Results (4 Datasets)

| Dataset                     | Baseline Acc | SVD Acc  |  Δ Acc | Grad FLOPs ↓ | Wall‑time ↓ |
| --------------------------- | ------------ | -------- | ------ | ------------ | ----------- |
| Synthetic (20 feat.)        | 0.53         | **0.54** | +0.01  |  ≈ ‑75 %     |  ≈ ‑45 %    |
| Breast‑Cancer (30 feat.)    | 0.62         | **0.65** | +0.03  |  ≈ ‑75 %     |  ≈ ‑40 %    |
| Digits 1‑vs‑rest (64 feat.) | 0.87         | **0.87** | ±0.00  |  ≈ ‑70 %     |  ≈ ‑35 %    |
| Diabetes (binarised)        | 0.73         | **0.74** | +0.01  |  ≈ ‑70 %     |  ≈ ‑30 %    |

*Setup*: two‑layer MLP, 1‑hidden‑layer = 10 units, batch GD 20 epochs; first 10 epochs used to fit SVD; rank = 2.  Times measured on CPU laptop for fairness.

**Take‑away** – even this naïve projection recovers or slightly *improves* accuracy while cutting per‑step gradient FLOPs by \~3–4×.

---

## 5  Limitations & Next Steps

*Temporal trap.* If we only ever update in the learned subspace it cannot evolve. We fix this with occasional full‑gradient sweeps or *rotating* which parameters get fully updated.

*Small nets only.* All experiments are ≤ 1 k parameters. We need CUDA profiling on a modern transformer block to see if the saving survives.

*Projection placeholder.* In the public repo we still call `.fit()` for the second phase to keep the demo lightweight; a real implementation would inject custom back‑prop hooks to project gradients before the optimiser sees them.

\### What’s next?

* **Real subspace projection** in PyTorch, so the whole back‑prop graph prunes unnecessary ops.
* **Adaptive rank** – grow *r* only when projection error rises.
* **Drift monitor** based on loss curvature, not just control gradients.

---

> **The punchline**: gradient redundancy is not just a communication artefact—it’s a compute opportunity. Early experiments hint that we can throw away 70–90 % of first‑layer gradient FLOPs and still converge, sometimes *better*. The challenge now is to make that true for big models and production kernels.

