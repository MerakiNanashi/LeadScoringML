
# Lead Scoring & Ranking System  
## A Production ML System for Constrained Decision-Making Under Extreme Class Imbalance

---

# 1. Introduction

This project presents a **production-grade machine learning system designed to prioritize high-conversion leads under extreme class imbalance (0.02% positive rate)**.

Instead of treating the problem as binary classification, the system reframes it as a **ranking and resource allocation problem**, optimized for real-world sales constraints such as limited bandwidth and fixed Top-K execution capacity.

The goal is not prediction accuracy, but **maximizing conversion capture within operational constraints**.

---

# 2. System Overview

The system ingests behavioral and interaction data, transforms it into structured features, applies machine learning models, and generates a ranked list of leads optimized for sales execution.

### System Architecture

Raw Data (WebEngage Behavioral Events)  
↓  
Feature Engineering Layer (Hybrid + Temporal)  
↓  
Time Window Aggregation (5–30 min behavior tracking)  
↓  
Balanced Random Forest Scoring  
↓  
Probability Calibration  
↓  
Ranking Engine (Top-K Prioritization)  
↓  
Daily Excel Reports  
↓  
CRM Integration (Future)

---

# 3. Problem Statement

## Business Problem

- Sales team can only process a limited number of leads per day
- Majority of leads (~99.98%) do not convert
- Manual prioritization leads to missed high-intent users

## ML Problem

- Highly imbalanced dataset (~0.02% positive class)
- Standard classification metrics are ineffective
- Need for **ranking-based optimization under constraint K**

---

# 4. Key Design Decisions

## 4.1 Classification → Ranking Reformulation

Instead of predicting:
> Will this user convert?

The system predicts:
> How should we rank users to maximize conversions in Top-K?

---

## 4.2 Metric Shift

| Traditional ML Metrics | System Metrics |
|----------------------|---------------|
| Accuracy             | Precision@K   |
| F1 Score             | Recall@K      |
| ROC-AUC              | Capture Rate@K |

---

## 4.3 Model Strategy Decision

## Model Strategy Decision

| Model Type | Reason for Use | Outcome |
|------------|----------------|---------|
| Logistic Regression | Strong baseline with interpretability and probabilistic calibration | Performed well as a baseline but was limited by its linear decision boundary, unable to capture complex feature interactions |
| XGBoost | High non-linearity modeling capability | Achieved strong training performance but showed instability and overfitting under extreme class imbalance (~0.02% positives) |
| Ensemble Models (Voting/Stacking) | Reduce variance and improve robustness | Provided marginal gains over individual models with increased system complexity |
| Balanced Random Forest | Designed for extreme class imbalance with better class balancing per tree | Selected for production due to stable recall performance and robust generalization under skewed distributions |

---

# 6. Feature Engineering System

## 6.1 Design Philosophy

Feature engineering was designed to:
- Capture early behavioral intent
- Handle extreme sparsity
- Avoid temporal leakage

---

## 6.2 Feature Sources

| Category | Examples |
|----------|---------|
| Device Signals | OS, browser, device type |
| Engagement Events | Clicks, form submissions |
| Funnel Behavior | OTP completion, drop-offs |
| Traffic Source | Paid, organic, direct |

---

## 6.3 Temporal Modeling

Behavior was aggregated across multiple windows:

| Window Size | Purpose |
|------------|--------|
| 5 min | Immediate intent detection |
| 10 min | Early behavioral signal |
| 15 min | Stabilized patterns |
| 30 min | Optimal predictive window |

### Key Insight

~70% of user engagement occurs within the first 30 minutes, making it the optimal signal window for prediction.

---

# 7. Modeling Approach

## 7.1 Class Imbalance Strategy

- Extreme imbalance (~0.02%)
- Direct accuracy optimization is ineffective
- Focus shifted to **recall-driven learning**

---

## 7.2 Final Model

Balanced Random Forest was selected due to:

- Strong recall stability under imbalance
- Robustness across time periods
- Lower operational complexity
- Comparable performance to ensembles

---

# 8. Ranking System Design

## 8.1 Transition from Classification

Instead of threshold-based classification, outputs were converted into:

> A ranked ordering of leads based on predicted conversion probability

---

## 8.2 Ranking Objective

- Maximize conversion capture within Top-K constraint
- Align model output with sales execution capacity

---

## 8.3 Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| Precision@K | Accuracy of top K predictions |
| Recall@K | Coverage of actual converters in Top-K |
| Capture Rate@K | % of total conversions captured |

---

# 9. Performance Results

## Dataset Scale

- Total leads: 108,000+
- Positive conversion rate: ~1.2%

---

## Top-K Performance

| K | Precision@K |
|--|-------------|
| 50 | 0.189 |
| 100 | 0.133 |
| 150 | 0.107 |
| 200 | 0.092 |

---

## Business Impact

- Captured ~24% of all converters within Top 50 ranked leads under extreme class imbalance (~0.02% positive rate)
- Estimated ~20–35% improvement in sales outreach efficiency by increasing concentration of high-intent leads in Top-K rankings  
  - This estimate is derived by comparing converter density in Top-K vs baseline (random/heuristics ordering), rather than a controlled A/B test
- Reduced wasted outreach by prioritizing high-probability leads within fixed sales bandwidth constraints
- Delivered a fully automated daily lead prioritization pipeline integrated into sales workflows
---

# 10. System Constraints & Tradeoffs

## 10.1 Core Constraints

- Extreme class imbalance
- Fixed sales bandwidth
- Batch-only inference (daily execution cycle)
- Feature leakage risk in temporal modeling

---

## 10.2 Key Tradeoffs

| Tradeoff | Decision |
|----------|----------|
| Accuracy vs Interpretability | Interpretability favored |
| Model complexity vs stability | Simpler BRF selected |
| Precision vs Recall | Recall prioritized |
| Batch vs Real-time | Batch chosen due to operational workflow |

---

# 11. Failure Cases & Iterations

## 11.1 Initial Failure

- Classification-first approach failed to align with sales constraints
- Threshold tuning was unstable across months

---

## 11.2 Ensemble Instability

- Complex ensemble models introduced variance without gain
- Increased training complexity with minimal improvement

---

## 11.3 Key Learning

The system was restructured into a **ranking-first architecture**, which aligned with real-world execution better than classification paradigms.

---

# 12. Limitations

- Cold-start leads with minimal behavioral signals remain hard to rank
- Model performance sensitive to distribution shifts across campaigns
- Batch processing limits real-time adaptability

---

# 13. Future Work

- Integration of first-call interaction signals
- Learning-to-rank models (LambdaRank, LightGBM Ranker)
- Online learning for adaptive ranking
- Real-time scoring system integration with CRM

---

# 14. Conclusion

This system demonstrates how a traditional classification problem can be transformed into a **ranking-based decision system optimized for real-world constraints**.

By shifting focus from predictive accuracy to **resource-constrained optimization (Top-K selection)**, the system significantly improved business outcomes, capturing a larger share of high-intent users while operating under severe class imbalance and operational limitations.

The key insight is that in production ML systems, **the best model is not the most accurate model, but the one best aligned with decision constraints**.

---

## Disclaimer

This repository represents a reconstructed version of work done in a professional setting. Sensitive data, internal APIs, and proprietary logic have been abstracted or omitted.
