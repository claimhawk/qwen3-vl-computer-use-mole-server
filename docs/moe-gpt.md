You already did the hard part (good, specialized LoRAs). Now you need a *learned dispatcher* that looks at the request (and screen) and decides **which LoRA(s)** to turn on.

Think of it as bolting a tiny Mixture-of-Experts router onto Qwen3-VL, where each “expert” is one of your LoRA adapters.

I’ll walk through:

1. **What exactly we’re routing (formalizing the setup)**
2. **Router architectures that actually work in this setting**
3. **How to wire the router into LoRA math**
4. **Training strategies (supervised vs performance-based)**
5. **Where to stick this in Qwen3-VL (vision + text)**
6. **Pitfalls and practical tricks**

I’ll stay implementation-oriented so you can actually build it.

---

## 1. Formalizing what you have now

You currently have:

* A shared base model ( f_\theta ) (Qwen3-VL)
* A set of LoRA adapters ({ \phi^{(1)}, \dots, \phi^{(K)} })

  * Each adapter (k) was trained on some subset of tasks / screens
  * When you attach adapter (k) and run, it does well on its intended workflows

At the linear layer level, for some weight matrix (W) in the base model, a LoRA adapter (k) gives:

[
\Delta W^{(k)} = A^{(k)} B^{(k)} \quad \text{(rank } r_k)
]

and the adapted layer becomes:

[
W^{(k)}_{\text{eff}} = W + \alpha_k \Delta W^{(k)}
]

Right now you pick **one** adapter by hand (or by task ID) and run with it.

We want to learn a *router* that, for each request+screen pair (x), outputs gating weights:

[
g(x) \in \mathbb{R}^K, \quad g_i(x) \ge 0, \quad \sum_i g_i(x) = 1
]

and uses either:

* **Soft mixture**: use all adapters with weights (g_i(x)), or
* **Hard routing**: pick (\arg\max_i g_i(x)) and only use that one.

---

## 2. Router architecture: what computes (g(x))?

You want the router to depend on **agent request + screen**:

* Request text (prompt / instruction / high-level task)
* Screen image (UI layout carries strong task info)

So define a feature vector (h_{\text{route}}(x)) the router will see.

A good, practical choice:

1. **Text side**: take the hidden state of the first token (or mean pool) after a few initial transformer layers
   (\to h_{\text{text}} \in \mathbb{R}^{d})

2. **Vision side**: take a pooled representation of the image tokens from the vision tower or the cross-modal fusion layer
   (\to h_{\text{vision}} \in \mathbb{R}^{d})

3. Concatenate and project:

[
h_{\text{route}} = \text{LayerNorm}([h_{\text{text}} ,|, h_{\text{vision}}]) \in \mathbb{R}^{d_r}
]

Then feed to a small MLP:

[
z = W_2 \sigma(W_1 h_{\text{route}} + b_1) + b_2 \in \mathbb{R}^K
]
[
g = \text{softmax}(z / \tau)
]

where (\tau) is a temperature that can be annealed towards sharp, almost one-hot choices.

You can share this same router across all layers where LoRAs sit (simpler, usually good enough). If you want per-layer routing, you can have separate heads (\text{MLP}_\ell) per layer, but start with *one global router*.

Key point: **router is small and trained after LoRAs are in place.**

---

## 3. How routing actually modifies the LoRA math

Take a single linear layer in the base model:

[
y = W x
]

With K LoRAs, each defines:

[
\Delta W^{(k)} = A^{(k)} B^{(k)}
]

Given gating vector (g), the **fused effective delta** is:

[
\Delta W^{(\text{fused})}(x) = \sum_{k=1}^{K} g_k(x) , \Delta W^{(k)}
]

So the layer becomes:

[
y = W x + \alpha \left( \sum_k g_k(x) , \Delta W^{(k)} \right) x
]

Implementation detail that matters for speed:

* For each example in the batch:

  * Compute (g(x)) once (global router)
  * Compute a **weighted sum of LoRA weights** per layer:

    * Either precompute fused (\Delta W^{(\text{fused})})
    * Or apply LoRA outputs and mix in activation space

The usual trick:

* Standard LoRA applies (A(Bx)).
* With K LoRAs: compute each (u^{(k)} = A^{(k)} (B^{(k)} x)), then:

[
u = \sum_k g_k(x) , u^{(k)}
]

and

[
y = W x + \alpha u
]

You can vectorize this to avoid a loop over K if you like, but conceptually that’s it.

If you want **hard routing**, replace:

[
g_k(x) \approx \text{one-hot of } \arg\max_k g_k(x)
]

During training use Gumbel-Softmax or straight-through estimator; at inference use pure argmax and only activate that adapter.

---

## 4. How to train the router

You already have good LoRAs. So the training objective is mostly about making the router **pick the right LoRA(s)** without wrecking them.

### 4.1. Data assumptions

You probably have:

* Multi-task dataset, where each example already “belongs” to a particular workflow/screen.

  * E.g., “calendar month view” vs “inbox list” vs “claim detail page”

This gives you two options:

1. **Router supervision by task label** (classification)
2. **Router supervision through performance** (end-to-end loss)

You can, and probably should, combine both.

---

### 4.2. Stage 1: Supervised router as task classifier

This is the simple starting point: teach router to approximate your existing manual routing.

For each training example ( (x, t) ) where:

* (x): (screen, prompt)
* (t \in {1, \dots, K}): “correct” adapter index (which LoRA you would have used)

**Freeze:**

* Base model weights ( \theta )
* All LoRA weights ( { \phi^{(k)} } )

**Train:**

* Router parameters ( \psi ) (the MLP)

Objective:

1. Forward pass:

   * Compute (h_{\text{route}}(x))
   * Compute (g(x) = \text{softmax}(z))
2. Classification loss:
   [
   L_{\text{route}} = -\log g_{t}(x)
   ]

You don’t even need to run through the LoRAs yet for this stage; you’re just training a task classifier.

This gives you a router that:

* Takes in the same inputs you will later use
* Outputs a distribution that matches your current “which LoRA?” labeling

Once this converges, you should see **high accuracy** on task IDs.

---

### 4.3. Stage 2: End-to-end routing with fixed experts

Now use the LoRAs and the router *together* and train the router on your actual supervision (actions, text outputs, etc).

Still:

* Base + LoRAs **frozen**
* Only router parameters update

For each example ( (x, y_{\text{target}}) ):

1. Compute (g(x)) via router

2. Use soft (or hard) routing to form fused LoRA effect

3. Run full model, get prediction (\hat{y}(x))

4. Compute task loss, e.g. cross-entropy or sequence-level loss:

   [
   L_{\text{task}} = \mathcal{L}(\hat{y}(x), y_{\text{target}})
   ]

5. Update router parameters using gradient from (L_{\text{task}})

To keep routing sane, add a regularizer:

* **Entropy penalty** to encourage near one-hot gating:

  [
  L_{\text{entropy}} = \lambda_H \sum_k g_k(x) \log g_k(x)
  ]

* Optional **KL to class labels** from Stage 1:

  [
  L_{\text{KL}} = \lambda_{\text{KL}} \sum_k q_k(t) \log \frac{q_k(t)}{g_k(x)}
  ]

  where (q(t)) is one-hot or smoothed distribution around the labeled adapter.

Overall:

[
L = L_{\text{task}} + L_{\text{route}} + L_{\text{entropy}} + L_{\text{KL}} \quad \text{(you can drop some terms if not needed)}
]

In practice:

* Start with: (L = L_{\text{task}} + \lambda_H L_{\text{entropy}})
* Optionally warm start with pure classification training (Stage 1) so router doesn’t start totally random.

---

### 4.4. Stage 3 (optional): Joint finetuning of router + LoRAs

Once router is stable, you can **lightly unfreeze** LoRAs to let them co-adapt:

* Unfreeze each (\phi^{(k)}) but use:

  * Low LR for LoRAs
  * Higher LR for router
* Still keep base model frozen

The intuition:

* Router has learned approximate partition of input space
* Now each LoRA can fine-tune under the gating pattern, resolving any mismatches or weird boundaries the router learned

This is basically doing a **Mixture-of-LoRAs MoE fine-tune**, but you’re starting from strong experts instead of random ones.

---

## 5. Where this lives inside Qwen3-VL specifically

For Qwen3-VL you have:

* **Vision tower** (e.g., a ViT variant) producing image tokens
* **Language transformer** that cross-attends to image tokens

Your LoRAs probably sit:

* On attention & MLP projections in the language transformer (and possibly some vision layers if you already patched them).

Where to plug the router:

### 5.1. Routing features

**Good default**:

* Take the text embedding of the first token **after** the first N transformer blocks (N small, like 2–4).
* Take mean-pooled or CLS image embedding after the vision tower (or the fused image tokens after first cross-attn).
* Concatenate, project to (h_{\text{route}}).

You want the router to see both:

* Prompt semantics (“open calendar and click date” vs “read claim status”)
* Visual layout (calendar grid vs table vs inbox, etc.)

### 5.2. Routing granularity

Several choices:

1. **Per-example global routing (recommended first)**
   One gating vector (g(x)) for all layers & tokens in that forward.

   * Simple and cheap
   * Matches your mental model: “pick the ‘calendar LoRA’ for this request”

2. **Per-layer routing**
   Different (g_\ell(x)) per transformer layer (\ell).

   * More expressive (some layers might share, others specialize)
   * More params and complexity
   * Only do this after you’re happy with global routing.

3. **Per-token routing**
   Gating dependent on token-level features.

   * Overkill initially, expensive, more unstable
   * Not needed for “choose the right tool/screen expert” kind of tasks.

For now: **global per-example routing** is exactly what you want.

---

## 6. Hard routing vs soft mixture

This matters for training stability vs inference efficiency.

### Soft mixture

* (g(x)) is a dense softmax over K adapters.
* Every forward uses *all* K LoRAs, but with different weights.
* Fully differentiable; easy to train router using standard backprop.
* At inference you can:

  * Keep soft mixture (highest quality, cost = K adapters)
  * Or approximate with top-1 or top-2 experts (prune small weights).

**Pros**:

* Training is easy and smooth.
* Router can hedge between similar adapters at boundaries.

**Cons**:

* More compute if you keep all K at inference.

### Hard routing

* At inference, you want **exactly one** LoRA active (cheap & interpretable).
* During training, you can:

  * Use Gumbel-Softmax sampling with straight-through estimator:

    * Forward pass uses one-hot (g)
    * Backward uses soft relaxation
  * Or treat router as a classifier, then just fine-tune LoRAs under that partition.

**Recommended recipe**:

1. Train with **soft mixture** + entropy penalty to push towards peaky distributions.
2. Monitor how often max probability > 0.9.
3. When it’s consistently high, you can:

   * Switch inference to pure argmax.
   * Optionally fine-tune a bit with straight-through argmax.

---

## 7. How this actually helps with your “tasks that hurt each other” split

Right now you’ve manually partitioned data/tasks into groups that don’t play nicely when trained together on the same LoRA.

What the router does:

* Learns a *boundary* in (screen, prompt) space such that:

  * Each region routes to a LoRA that was trained mostly on that region.
* As long as LoRAs are decent specialists, the router will learn to approximate your manual mapping, and possibly refine it.

At a transformer / tower level:

* Base model provides general perception + language.
* LoRAs provide task-specific “style” and behavior.
* Router is a low-capacity network whose job is **not** to solve the task, just to predict *which task manifold* the current input lies on.
* Because LoRAs are frozen initially, gradients flowing into router reflect “which adapter would have reduced loss most for this kind of input,” so it naturally learns the partition.

---

## 8. Practical training details / gotchas

### 8.1. Dataset balancing

* If some LoRAs see far more examples than others during router training, router may:

  * Default to a majority expert
* Fix by:

  * Balancing batches across LoRA/task indices
  * Adding a “load balancing” term (e.g., encourage roughly uniform marginal distribution of gate usage).

### 8.2. Collapse to one adapter

Even with multiple LoRAs, router might learn “adapter #2 is good enough everywhere.”

Mitigations:

* Slightly **sharpen** per-task data: feed router examples where that adapter is clearly best.
* Add a penalty if marginal gate distribution across batch becomes too skewed.
* Keep LoRAs **frozen** in the early routing training, so router is forced to exploit existing specialization instead of bending an expert to cover everything.

### 8.3. When to unfreeze LoRAs

* Only after router training plateaued (router’s task loss stops improving).
* Use low LR (e.g. 10x smaller than router’s).
* Keep a checkpoint of “frozen-expert router” state so you can roll back if joint training destabilizes specialization.

### 8.4. Qwen implementation constraints

* Make sure your LoRA framework (PEFT/your own) supports **multiple adapters loaded at once**.
* You want something like:

  * Attach adapters `{lora_calendar, lora_inbox, lora_claims, ...}` to the same base.
  * At forward() time, pass gating vector to a custom LoRA module that mixes them.

---

## 9. Mental model to keep in your head

You’re building:

* A fixed **multi-expert function**:
  ( f^{(k)}(x) = f_\theta(x; \phi^{(k)}) ) for k = 1..K
* A small **router** ( r_\psi(x) \to g(x) )
* A **mixture model**:

[
f(x) = \sum_k g_k(x) f^{(k)}(x)
]

with the constraint that all (f^{(k)}) share 99% of their parameters (the base model) and differ only via LoRAs.

Training the router with fixed experts is essentially just learning:

[
p(k \mid x) = g_k(x)
]

such that the mixture prediction matches your targets well.

That’s the core: you’re not retraining the big tower, just bolting on a differentiable **“which expert?” brain** that understands your agent request + screen.

---

If you want, next step I can sketch a pseudo-API / module design for a **MultiLoRAWithRouter** wrapper around Qwen3-VL that encodes exactly this behavior (no project-management fluff, just the moving parts you need to implement).
