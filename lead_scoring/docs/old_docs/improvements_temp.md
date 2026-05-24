# The Gap Between Conversion Prediction and Business Impact

## Overview

This note explores a structural problem in lead prioritization systems:

Most systems optimize for conversion probability, while businesses often care more about incremental conversion caused by intervention.

This creates a mismatch between:
- model optimization
- sales effort allocation
- actual business impact

The document discusses:
- why standard top-K lead scoring fails over time
- how selection bias emerges
- why ranker + reranker architectures are more suitable
- lightweight alternatives to full uplift modeling
- practical production tradeoffs

---

# Standard Lead Scoring Pipeline

A typical lead system follows:

```text
score leads -> pick top K -> send to sales
```

The model estimates:

\[
P(\text{conversion} | x)
\]

where:
- \(x\) = lead features
- output = conversion probability

Top-scoring leads are selected and passed to the sales team.

Initially, this works well.

However, the system gradually begins optimizing the wrong thing.

---

# The Feedback Loop Problem

The system creates a self-reinforcing loop.

## Step-by-step behavior

1. Model selects high-confidence leads
2. Sales engages them
3. Those leads convert at high rates
4. Outcomes become training data
5. Model becomes more confident in similar leads

Over time:
- the same type of leads keep getting surfaced
- high-intent leads dominate exposure
- uncertain or mid-tier leads disappear from the pipeline

The model increasingly learns:

```text
who converts anyway
```

instead of:

```text
who converts because of intervention
```

---

# Selection Bias

This is fundamentally a selection bias problem.

The model is trained on outcomes that are influenced by its own decisions.

The system only learns from:
- leads that were surfaced
- leads that received intervention
- leads that accumulated enough signal

This creates a closed feedback loop.

The ranking policy influences:
- what data gets observed
- which labels exist
- what the next model learns

This is closely related to:
- exposure bias
- counterfactual learning
- learning-to-rank feedback loops

---

# Objective Mismatch

Traditional lead scoring optimizes:

\[
P(\text{conversion})
\]

But the actual business objective is closer to:

\[
P(\text{conversion} | \text{intervention}) -
P(\text{conversion} | \text{no intervention})
\]

This corresponds to:
- incremental conversion
- uplift
- intervention value

The real question is not:

```text
Who will buy?
```

but:

```text
Who will buy because of us?
```

---

# Why Class Imbalance Makes This Worse

Lead systems often have:
- extremely sparse conversions
- high imbalance between paid and unpaid users
- mixed organic and intervention-driven outcomes

This creates:
- noisy labels
- unstable confidence estimates
- biased training distributions

As a result:
- high-confidence leads dominate training
- uncertain leads rarely accumulate enough exposure
- exploration collapses over time

Even though some uncertain leads may respond strongly to intervention.

---

# Reframing the Problem

The problem should not be framed as:

```text
individual prediction
```

Instead, it is closer to:

```text
set optimization under resource constraints
```

Sales teams operate under:
- limited outreach capacity
- limited human bandwidth
- constrained intervention budgets

So the goal is not:
- selecting the best individual leads

but:
- selecting the best set of leads

---

# Proposed Architecture

## Ranker + Reranker

A more suitable structure:

```text
All Leads
   ↓
Ranker
   ↓
Candidate Set
   ↓
Reranker
   ↓
Final Sales Set
```

---

# Stage 1 — Ranker

The ranker performs:
- large-scale retrieval
- coarse relevance estimation
- candidate filtering

Objective:

\[
f(x) \approx P(\text{conversion})
\]

The ranker reduces:
- very large lead pools
- into manageable candidate sets

Example:

```text
100K leads -> top 500 candidates
```

---

# Exploration During Candidate Generation

One issue:
- if the ranker only surfaces high-confidence leads
- the reranker never sees potentially valuable uncertain leads

A lightweight mitigation:

```text
80% exploitation
20% exploration
```

Where:
- exploitation = high-confidence leads
- exploration = uncertain or mid-tier leads

This improves recall of potentially intervenable leads.

However:
this alone is insufficient.

Why?

Because the reranker may still eliminate exploratory leads if it optimizes only short-term conversion.

---

# Stage 2 — Reranker

Instead of optimizing:

\[
\max \sum y_i
\]

the reranker should optimize a joint objective:

\[
\max
\sum_i
[
\text{conversion}(x_i)
+
\lambda \cdot \text{uplift}(x_i)
]
\]

Where:
- conversion = predicted likelihood
- uplift = estimated intervention value
- \(\lambda\) = tradeoff parameter

This transforms the reranker from:
- score selector

into:
- set optimizer

---

# Why Full Uplift Modeling Is Difficult

A complete uplift framework is attractive but operationally expensive.

Challenges include:
- sparse intervention labels
- counterfactual estimation
- experimentation requirements
- exploration costs
- infra complexity

Many production systems cannot support this directly.

---

# Segment-Level Baseline Adjustment

A lightweight approximation:

Estimate baseline conversion rates per segment using:
- leads not acted on
- historical organic conversion behavior

Then adjust scores relative to that baseline.

---

# Example

Suppose:

| Segment | Baseline Conversion |
|---|---|
| High Intent | 0.70 |
| Mid Funnel | 0.15 |
| Low Intent | 0.03 |

If a lead has:

\[
P(\text{conversion}) = 0.75
\]

inside a segment with:

\[
baseline = 0.70
\]

then incremental value is small.

Meanwhile:

\[
P(\text{conversion}) = 0.30
\]

inside a segment with:

\[
baseline = 0.05
\]

may represent stronger intervention value.

---

# Adjusted Scoring

Adjusted score:

\[
score'(x) = score(x) -
baseline(segment(x))
\]

This approximates:

```text
incremental conversion
```

without:
- causal inference pipelines
- pairwise intervention models
- major infrastructure changes

---

# Why This Helps

The adjustment naturally:
- downweights “sure-things”
- surfaces influenceable leads
- reallocates effort toward intervention-sensitive segments

The system shifts from:

```text
predicting buyers
```

to:

```text
identifying where intervention creates value
```

---

# Alternative Approaches Considered

## 1. Pairwise Similarity / Diversity Models

Potential formulation:

\[
score_i' = score_i -
\alpha \cdot similarity(i, S)
\]

Goal:
- reduce redundancy
- improve coverage
- diversify exposure

Problem:
- pairwise inference
- high serving cost
- approaches \(O(N^2)\)

Operationally difficult at scale.

---

## 2. Pure Random Exploration

Idea:
- randomly inject exploratory leads

Advantage:
- better learning signal

Problem:
- directly conflicts with short-term conversion metrics
- difficult stakeholder alignment

---

## 3. Increasing Candidate Pool Size

Idea:
- increase ranker K

Example:

```text
top 100 -> top 500
```

This improves recall.

However:
- does not solve selection bias
- reranker may still collapse toward high-confidence leads

---

## 4. Full Uplift Modeling

Most theoretically correct.

But:
- expensive
- data-intensive
- operationally difficult

---

# Relationship to Learning-to-Rank Systems

This problem closely resembles:
- Learning-to-Rank (LTR)
- recommendation systems
- exposure bias correction
- counterfactual learning

In ranking systems:
- models learn from exposed items
- exposure affects labels
- labels affect future exposure

This creates the same self-reinforcing loop.

Relevant areas:
- propensity correction
- unbiased learning-to-rank
- counterfactual ranking

---

# Key Conceptual Shift

Traditional systems ask:

```text
Who is most likely to convert?
```

This framework instead asks:

```text
Where does intervention create value?
```

That shift changes:
- training signals
- ranking objectives
- candidate generation
- evaluation metrics
- system behavior over time

---

# Potential Evaluation Metrics

Instead of only:
- Precision@K
- Recall@K
- Conversion Rate

Potential alternatives:
- incremental conversions
- uplift-adjusted conversion
- intervention efficiency
- exposure diversity
- long-term learning quality

---

# Open Questions

Several important questions remain:

1. How should uplift proxies be estimated?
2. How should exploration budgets be allocated?
3. How much short-term metric degradation is acceptable?
4. Can rerankers optimize causal objectives directly?
5. How should long-term policy quality be measured?

---

# Final Thought

Many production systems optimize:

```text
what is easiest to measure
```

rather than:

```text
what actually creates value
```

Lead prioritization is one example where that gap becomes visible.

A relatively small change in objective design can fundamentally alter:
- where effort is allocated
- what data gets collected
- and ultimately what the system learns.