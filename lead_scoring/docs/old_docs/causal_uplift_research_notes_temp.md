# Causal Uplift and Budgeted Lead Allocation

Graduate-level research notes for moving from conversion prediction to incremental-value allocation.

---

## 1. Problem Setup

The current production lead-scoring system ranks leads under a capacity constraint. A standard ranker estimates:

\[
s(x) \approx P(Y=1 \mid X=x)
\]

and selects the top \(B_t\) leads in time window \(t\). This is useful when the goal is to concentrate likely converters, but it does not answer the causal business question:

> Which leads convert because of intervention?

For lead \(i\), define:

| Symbol | Meaning |
|---|---|
| \(X_i\) | pre-treatment lead features |
| \(A_i \in \{0,1\}\) | intervention decision |
| \(Y_i(1)\) | potential outcome if intervened |
| \(Y_i(0)\) | potential outcome without intervention |
| \(B_t\) | capacity in time window \(t\) |
| \(\pi\) | allocation policy |

The original ecosystem objective is:

\[
\max_\pi \sum_i \left[A_iY_i(1)+(1-A_i)Y_i(0)\right]
\quad \text{s.t.} \quad
\sum_i A_i \le B_t
\]

Define individual treatment effect/uplift:

\[
\tau_i = Y_i(1)-Y_i(0)
\]

Then:

\[
Y_i(1)=Y_i(0)+\tau_i
\]

Substitute into the objective:

\[
\sum_i \left[A_i(Y_i(0)+\tau_i)+(1-A_i)Y_i(0)\right]
\]

\[
=\sum_i \left[Y_i(0)+A_i\tau_i\right]
\]

Because \(\sum_i Y_i(0)\) does not depend on the allocation policy, optimization reduces to:

\[
\max_\pi \sum_i A_i\tau_i
\quad \text{s.t.} \quad
\sum_i A_i \le B_t
\]

This is the central transformation:

> Conversion prediction ranks by expected outcome. Uplift allocation ranks by expected incremental outcome.

---

## 2. Potential Outcomes and Identification

For each lead, only one potential outcome is observed:

\[
Y_i = A_iY_i(1)+(1-A_i)Y_i(0)
\]

The unobserved outcome is counterfactual. This is the fundamental problem of causal inference.

The estimand for targeting is usually the conditional average treatment effect (CATE):

\[
\tau(x)=\mathbb{E}[Y(1)-Y(0)\mid X=x]
\]

For binary outcomes, \(\tau(x)\) is an incremental conversion probability. For continuous revenue outcomes, \(\tau(x)\) is incremental expected revenue.

### Core Assumptions

**SUTVA / no interference.**

\[
Y_i(a) \text{ is unaffected by } A_j \text{ for } j \ne i
\]

In lead allocation, this can fail if sales capacity, social influence, referral effects, or market saturation cause one lead's treatment to affect another lead's outcome.

**Consistency.**

\[
Y_i = Y_i(A_i)
\]

The observed outcome equals the potential outcome under the action actually taken. This can fail if "intervention" is not well-defined, for example if a sales call varies greatly by rep quality or call script.

**Conditional exchangeability / unconfoundedness.**

\[
\{Y_i(1),Y_i(0)\} \perp A_i \mid X_i
\]

Given observed lead features, intervention assignment is as good as random. This is credible in randomized experiments and much harder in historical sales data, because reps tend to intervene on leads they already believe are promising.

**Overlap / positivity.**

\[
0 < e(x) < 1
\quad \text{where} \quad
e(x)=P(A=1\mid X=x)
\]

Every feature region needs some treated and untreated examples. Without overlap, the counterfactual outcome for some lead types is extrapolated rather than learned.

### Identification Theorem

Under consistency, conditional exchangeability, and overlap:

\[
\tau(x)=\mathbb{E}[Y\mid X=x,A=1]-\mathbb{E}[Y\mid X=x,A=0]
\]

Proof sketch:

\[
\mathbb{E}[Y(1)\mid X=x]
=\mathbb{E}[Y(1)\mid X=x,A=1]
=\mathbb{E}[Y\mid X=x,A=1]
\]

and similarly:

\[
\mathbb{E}[Y(0)\mid X=x]
=\mathbb{E}[Y\mid X=x,A=0]
\]

Therefore:

\[
\mathbb{E}[Y(1)-Y(0)\mid X=x]
=\mathbb{E}[Y\mid X=x,A=1]-\mathbb{E}[Y\mid X=x,A=0]
\]

This theorem is the bridge from counterfactual quantities to observable data.

---

## 3. From Lead Scoring to Uplift Scoring

Ordinary lead scoring estimates:

\[
m(x)=\mathbb{E}[Y\mid X=x]
\]

Uplift scoring estimates:

\[
\tau(x)=\mathbb{E}[Y(1)-Y(0)\mid X=x]
\]

These are different rankings.

High \(m(x)\), low \(\tau(x)\): a likely organic converter, often a "sure thing."

Low or medium \(m(x)\), high \(\tau(x)\): a persuadable lead where intervention changes the outcome.

Negative \(\tau(x)\): a lead harmed by intervention. In marketing these are sometimes called "sleeping dogs"; in sales this can mean over-contact, discount anchoring, distrust, or timing mismatch.

The production target is not:

\[
\arg\max_i P(Y_i=1\mid X_i)
\]

but:

\[
\arg\max_i \mathbb{E}[Y_i(1)-Y_i(0)\mid X_i]
\]

or, with lead value and intervention cost:

\[
\arg\max_i \left(\hat{\tau}(X_i)v_i-c_i\right)
\]

---

## 4. Binary-Treatment CATE Estimators

Let:

\[
\mu_a(x)=\mathbb{E}[Y\mid X=x,A=a]
\]

\[
e(x)=P(A=1\mid X=x)
\]

\[
m(x)=\mathbb{E}[Y\mid X=x]
\]

The main estimator families differ in how they combine outcome modeling, propensity modeling, residualization, and direct treatment-effect learning.

### 4.1 S-Learner

Fit one supervised model:

\[
\hat{\mu}(x,a)\approx \mathbb{E}[Y\mid X=x,A=a]
\]

Then:

\[
\hat{\tau}_S(x)=\hat{\mu}(x,1)-\hat{\mu}(x,0)
\]

Advantages:

- simple
- uses all data in one model
- often strong with flexible learners such as LightGBM or CatBoost

Risks:

- the model may ignore \(A\) if treatment signal is weak
- treatment heterogeneity can be over-smoothed
- not ideal when treatment and control mechanisms differ sharply

### 4.2 T-Learner

Fit separate models:

\[
\hat{\mu}_1(x)\approx \mathbb{E}[Y\mid X=x,A=1]
\]

\[
\hat{\mu}_0(x)\approx \mathbb{E}[Y\mid X=x,A=0]
\]

Then:

\[
\hat{\tau}_T(x)=\hat{\mu}_1(x)-\hat{\mu}_0(x)
\]

Advantages:

- intuitive
- flexible
- good when treatment and control response surfaces differ

Risks:

- noisy when either arm is small
- uplift is a difference of two fitted functions, so errors compound
- sensitive to imbalance between treatment and control groups

### 4.3 X-Learner

The X-Learner is designed to handle imbalanced treatment/control groups.

Step 1: fit \(\hat{\mu}_0(x)\) and \(\hat{\mu}_1(x)\).

Step 2: impute individual treatment-effect pseudo-outcomes:

For treated units:

\[
\tilde{D}_i^1=Y_i-\hat{\mu}_0(X_i)
\]

For control units:

\[
\tilde{D}_i^0=\hat{\mu}_1(X_i)-Y_i
\]

Step 3: fit treatment-effect models:

\[
\hat{\tau}_1(x)\approx \mathbb{E}[\tilde{D}^1\mid X=x,A=1]
\]

\[
\hat{\tau}_0(x)\approx \mathbb{E}[\tilde{D}^0\mid X=x,A=0]
\]

Step 4: combine:

\[
\hat{\tau}_X(x)=g(x)\hat{\tau}_0(x)+(1-g(x))\hat{\tau}_1(x)
\]

Commonly:

\[
g(x)=\hat{e}(x)
\]

The X-Learner is useful when one group is much larger than the other and the CATE has learnable structure.

### 4.4 R-Learner

The R-Learner residualizes both outcome and treatment.

Start from the Robinson decomposition:

\[
Y_i-m(X_i)=\left(A_i-e(X_i)\right)\tau(X_i)+\epsilon_i
\]

Estimate \(\hat{m}(x)\) and \(\hat{e}(x)\), then solve:

\[\hat{\tau}_R=\arg\min_\tau\sum_i\left[\left(Y_i-\hat{m}(X_i)\right)-\left(A_i-\hat{e}(X_i)\right)\tau(X_i)\right]^2\]

often with weights:

\[
\left(A_i-\hat{e}(X_i)\right)^2
\]

and regularization:

\[
\hat{\tau}_R
=\arg\min_\tau
\sum_i
\left(
\left[Y_i-\hat{m}(X_i)\right]-\left[A_i-\hat{e}(X_i)\right]\tau(X_i)
\right)^2
\ + \Lambda_n(\tau)
\]

The R-Learner is attractive in observational data because it isolates the treatment-effect signal after removing predictable outcome and treatment-assignment components.

### 4.5 DR-Learner

The doubly robust CATE pseudo-outcome is:

\[
\phi_i
=\hat{\mu}_1(X_i)-\hat{\mu}_0(X_i)
+\frac{A_i\left(Y_i-\hat{\mu}_1(X_i)\right)}{\hat{e}(X_i)}
-\frac{(1-A_i)\left(Y_i-\hat{\mu}_0(X_i)\right)}{1-\hat{e}(X_i)}
\]

Then fit:

\[
\hat{\tau}_{DR}(x)\approx \mathbb{E}[\phi_i\mid X_i=x]
\]

The key property is double robustness:

> The pseudo-outcome has the correct conditional expectation if either the outcome models are correct or the propensity model is correct, under standard regularity conditions.

This is one of the strongest practical choices for observational uplift modeling, especially with cross-fitting.

### 4.6 Cross-Fitting

Cross-fitting prevents nuisance models from overfitting the same observations used to estimate effects.

Procedure:

1. Split data into \(K\) folds.
2. For each fold \(k\), train nuisance models on all other folds.
3. Predict \(\hat{\mu}_a^{(-k)}(X_i)\), \(\hat{e}^{(-k)}(X_i)\), or \(\hat{m}^{(-k)}(X_i)\) on held-out fold \(k\).
4. Build pseudo-outcomes using out-of-fold nuisance predictions.
5. Fit final CATE model.

This supports Neyman-orthogonal/debiased estimation, where first-stage nuisance errors affect the final estimate only at second order.

### 4.7 Causal Forests and Generalized Random Forests

Causal forests estimate \(\tau(x)\) by building trees whose splits are designed to reveal treatment-effect heterogeneity rather than merely outcome prediction.

Generalized random forests view forests as adaptive local weighting methods. For a target point \(x\), the forest produces weights:

\[
\alpha_i(x)
\]

and estimates a local parameter by solving a weighted moment equation:

\[
\sum_i \alpha_i(x)\psi_{\theta,\nu}(O_i)=0
\]

For CATE estimation, a simplified orthogonal moment is:

\[
\sum_i \alpha_i(x)
\left(A_i-\hat{e}(X_i)\right)
\left[
Y_i-\hat{m}(X_i)
-\tau(x)\left(A_i-\hat{e}(X_i)\right)
\right]
=0
\]

Solving gives a local residualized treatment-effect estimate:

\[
\hat{\tau}(x)
=\frac{
\sum_i \alpha_i(x)(A_i-\hat{e}(X_i))(Y_i-\hat{m}(X_i))
}{
\sum_i \alpha_i(x)(A_i-\hat{e}(X_i))^2
}
\]

Modern causal forests use ideas such as honesty, sample splitting, orthogonalization, and balanced treatment/control splits. Their practical advantages are:

- nonlinear CATE discovery
- interpretable heterogeneity analysis
- uncertainty intervals
- strong tabular-data performance

---

## 5. Formal Results and Theorem-Level Statements

### 5.1 Fundamental Problem of Causal Inference

For each unit \(i\), the pair \((Y_i(1),Y_i(0))\) is never fully observed. Observed data contains:

\[
(X_i,A_i,Y_i)
\]

where:

\[
Y_i=Y_i(A_i)
\]

Thus individual treatment effects:

\[
\tau_i=Y_i(1)-Y_i(0)
\]

are not directly observable without additional assumptions or experimental design.

### 5.2 Identification of CATE

If consistency, conditional exchangeability, and overlap hold, then:

\[
\tau(x)=\mu_1(x)-\mu_0(x)
\]

where:

\[
\mu_a(x)=\mathbb{E}[Y\mid X=x,A=a]
\]

This justifies two-model uplift estimation and the outcome-regression components of meta-learners.

### 5.3 Doubly Robust Consistency

For the AIPW/DR score:

\[
\phi_i
=\hat{\mu}_1(X_i)-\hat{\mu}_0(X_i)
+\frac{A_i(Y_i-\hat{\mu}_1(X_i))}{\hat{e}(X_i)}
-\frac{(1-A_i)(Y_i-\hat{\mu}_0(X_i))}{1-\hat{e}(X_i)}
\]

the estimator remains consistent for the treatment effect if either:

\[
\hat{\mu}_a(x)\to \mu_a(x)
\]

or:

\[
\hat{e}(x)\to e(x)
\]

This is not magic; finite-sample performance can still be poor with extreme propensities, weak overlap, or unstable nuisance models.

### 5.4 Neyman Orthogonality

Let \(\eta\) denote nuisance functions such as \(\mu_a(x)\), \(m(x)\), and \(e(x)\). A score \(\psi(W;\theta,\eta)\) is Neyman orthogonal if:

\[
\left.
\frac{\partial}{\partial r}
\mathbb{E}
\left[
\psi(W;\theta_0,\eta_0+r(\eta-\eta_0))
\right]
\right|_{r=0}
=0
\]

Intuition:

> Small first-stage nuisance errors have only second-order impact on the target effect estimate.

This is the mathematical reason R-Learners, DR-Learners, DML, and orthogonal forests are attractive with flexible machine learning nuisance models.

### 5.5 Equal-Cost Top-\(B_t\) Allocation Theorem

Given fixed estimated incremental values \(\hat{\tau}_i\) and binary decisions \(A_i\in\{0,1\}\), solve:

\[
\max_A \sum_i A_i\hat{\tau}_i
\quad \text{s.t.} \quad
\sum_i A_i \le B_t
\]

The optimal plug-in policy selects the \(B_t\) largest positive values of \(\hat{\tau}_i\).

Proof sketch:

If selected lead \(j\) has lower uplift than unselected lead \(k\):

\[
\hat{\tau}_k > \hat{\tau}_j
\]

then swapping \(j\) out and \(k\) in improves the objective by:

\[
\hat{\tau}_k-\hat{\tau}_j > 0
\]

Therefore any optimum cannot omit a higher positive uplift while selecting a lower one. Negative uplifts should not be selected when the constraint is \(\le B_t\) rather than \(=B_t\).

### 5.6 Variable-Cost Allocation as Knapsack

If interventions have cost \(c_i\) and value \(g_i\):

\[
g_i=\hat{\tau}_iv_i-c_i
\]

then:

\[
\max_A \sum_i A_ig_i
\quad \text{s.t.} \quad
\sum_i A_ic_i \le B_t,
\quad A_i\in\{0,1\}
\]

This is a 0-1 knapsack problem. If extra constraints are added, it becomes a general integer linear program.

### 5.7 Policy Value and Regret

Define policy value:

\[
V(\pi)=\mathbb{E}[Y(\pi(X))]
\]

For binary action:

\[
Y(\pi(X))=\pi(X)Y(1)+(1-\pi(X))Y(0)
\]

Policy regret is:

\[
R(\pi)=V(\pi^*)-V(\pi)
\]

where:

\[
\pi^*=\arg\max_{\pi\in\Pi} V(\pi)
\]

Efficient policy learning studies how fast empirical policies approach the best policy in a class \(\Pi\), often under constraints such as budget, fairness, interpretability, or tree depth.

---

## 6. Allocation Optimization

### 6.1 Equal-Cost Binary Allocation

When all interventions consume one unit of capacity:

\[
A_i =
\begin{cases}
1 & \text{if } \hat{\tau}_i \text{ is among the top } B_t \text{ positive uplifts}\\
0 & \text{otherwise}
\end{cases}
\]

If the business requires exactly \(B_t\) contacts, select the top \(B_t\) even if some are negative, but report the expected loss from forced allocation.

### 6.2 Incremental Profit Allocation

For lead value \(v_i\) and intervention cost \(c_i\):

\[
\hat{g}_i=\hat{\tau}_iv_i-c_i
\]

Select by \(\hat{g}_i\), not raw \(\hat{\tau}_i\). A low-uplift enterprise lead may dominate a high-uplift low-value lead.

### 6.3 Rep Capacity Constraints

Let \(r(i)\) be the assigned sales rep or territory:

\[
\sum_{i:r(i)=r} A_i \le C_r
\]

Full objective:

\[
\max_A \sum_i A_i\hat{g}_i
\]

\[
\text{s.t.} \quad
\sum_i A_i \le B_t
\]

\[
\sum_{i:r(i)=r} A_i \le C_r
\quad \forall r
\]

\[
A_i\in\{0,1\}
\]

### 6.4 Segment, Fairness, and Business Quotas

For segment \(s\), define \(S_s=\{i:s(i)=s\}\). Then:

Minimum quota:

\[
\sum_{i\in S_s} A_i \ge L_s
\]

Maximum quota:

\[
\sum_{i\in S_s} A_i \le U_s
\]

Share constraint:

\[
\ell_s
\le
\frac{\sum_{i\in S_s} A_i}{\sum_i A_i}
\le
u_s
\]

These constraints convert the simple top-\(B_t\) policy into an ILP. The advantage of post-model optimization is modularity: the same CATE model can be reused while business constraints change.

### 6.5 Robust Allocation with Uncertainty

If a model provides standard errors or intervals:

\[
\hat{\tau}_i \pm z_{\alpha/2}\hat{\sigma}_i
\]

a conservative score can be:

\[
\hat{g}_i^{LCB}
=\left(\hat{\tau}_i-z_{\alpha/2}\hat{\sigma}_i\right)v_i-c_i
\]

This reduces over-allocation to noisy high-uplift estimates.

---

## 7. Multi-Action Treatment Allocation

In production, "intervention" is rarely binary. A lead may receive:

\[
a\in\{0,1,\dots,K\}
\]

where \(0\) is no action and other actions may be email, call, WhatsApp, discount, demo invite, senior-rep escalation, or retargeting.

Potential outcomes:

\[
Y_i(0),Y_i(1),\dots,Y_i(K)
\]

Action-specific uplift against control:

\[
\tau_i(a)=Y_i(a)-Y_i(0)
\]

CATE:

\[
\tau_a(x)=\mathbb{E}[Y(a)-Y(0)\mid X=x]
\]

The assignment problem becomes:

\[
\max_{A_{ia}}
\sum_i\sum_{a=1}^K A_{ia}\hat{g}_{ia}
\]

subject to:

\[
\sum_{a=1}^K A_{ia}\le 1
\quad \forall i
\]

\[
\sum_i\sum_{a=1}^K c_{ia}A_{ia}\le B_t
\]

\[
A_{ia}\in\{0,1\}
\]

where:

\[
\hat{g}_{ia}=\hat{\tau}_a(X_i)v_i-c_{ia}
\]

### Multi-Action Identification

Let:

\[
e_a(x)=P(A=a\mid X=x)
\]

Under multi-action unconfoundedness:

\[
\{Y(0),Y(1),\dots,Y(K)\}\perp A\mid X
\]

and overlap:

\[
e_a(x)>0 \quad \forall a
\]

then:

\[
\mathbb{E}[Y(a)\mid X=x]
=\mathbb{E}[Y\mid X=x,A=a]
\]

and:

\[
\tau_a(x)
=\mathbb{E}[Y\mid X=x,A=a]
-\mathbb{E}[Y\mid X=x,A=0]
\]

### Multi-Action Estimator Families

**Separate outcome models.**

\[
\hat{\mu}_a(x)\approx \mathbb{E}[Y\mid X=x,A=a]
\]

\[
\hat{\tau}_a(x)=\hat{\mu}_a(x)-\hat{\mu}_0(x)
\]

**Single model with action features.**

\[
\hat{\mu}(x,a)\approx \mathbb{E}[Y\mid X=x,A=a]
\]

\[
\hat{\tau}_a(x)=\hat{\mu}(x,a)-\hat{\mu}(x,0)
\]

**Multi-action DR score.**

For each action \(a\):

\[
\phi_a(O_i)
=\hat{\mu}_a(X_i)
+\frac{\mathbf{1}\{A_i=a\}}{\hat{e}_a(X_i)}
\left(Y_i-\hat{\mu}_a(X_i)\right)
\]

Then estimate:

\[
\hat{\tau}_a(x)
\approx
\mathbb{E}[\phi_a-\phi_0\mid X=x]
\]

**Direct policy learning.**

Instead of estimating \(\tau_a(x)\), learn:

\[
\hat{\pi}
=\arg\max_{\pi\in\Pi}\hat{V}(\pi)
\]

This is attractive when the objective is decision quality, not effect-estimation accuracy.

---

## 8. Off-Policy Evaluation and Policy Learning

Historical sales data is usually logged under a behavior policy:

\[
\pi_b(a\mid x)
\]

The target policy is:

\[
\pi(a\mid x)
\]

The challenge: we only observe \(Y_i\) for the action actually taken.

### 8.1 Policy Value

For action set \(\mathcal{A}\):

\[
V(\pi)=\mathbb{E}\left[\sum_{a\in\mathcal{A}}\pi(a\mid X)Y(a)\right]
\]

For deterministic policy:

\[
V(\pi)=\mathbb{E}[Y(\pi(X))]
\]

### 8.2 Direct Method

Estimate outcome models \(\hat{\mu}_a(x)\):

\[
\hat{V}_{DM}(\pi)
=\frac{1}{n}
\sum_{i=1}^n
\sum_a
\pi(a\mid X_i)\hat{\mu}_a(X_i)
\]

Low variance, but biased if outcome models are wrong.

### 8.3 Inverse Propensity Weighting

For deterministic \(\pi\):

\[
\hat{V}_{IPW}(\pi)
=\frac{1}{n}
\sum_{i=1}^n
\frac{\mathbf{1}\{A_i=\pi(X_i)\}Y_i}{\hat{e}_{A_i}(X_i)}
\]

For stochastic \(\pi\):

\[
\hat{V}_{IPW}(\pi)
=\frac{1}{n}
\sum_{i=1}^n
\frac{\pi(A_i\mid X_i)}{\hat{\pi}_b(A_i\mid X_i)}Y_i
\]

Unbiased when propensities are correct, but high variance when propensities are small.

### 8.4 Doubly Robust Policy Value

\[
\hat{V}_{DR}(\pi)
=\frac{1}{n}
\sum_{i=1}^n
\left[
\sum_a \pi(a\mid X_i)\hat{\mu}_a(X_i)
+\frac{\pi(A_i\mid X_i)}{\hat{\pi}_b(A_i\mid X_i)}
\left(
Y_i-\hat{\mu}_{A_i}(X_i)
\right)
\right]
\]

This combines a model-based estimate with a propensity-weighted residual correction.

### 8.5 Policy Learning

Policy learning optimizes:

\[
\hat{\pi}
=\arg\max_{\pi\in\Pi}\hat{V}_{DR}(\pi)
\]

The policy class \(\Pi\) can encode:

- top-\(B_t\) budget rules
- decision trees
- linear scoring rules
- monotone business rules
- fairness constraints
- multi-action routing rules

Theoretical results from efficient policy learning and offline multi-action policy learning show that, under regularity conditions, doubly robust policy learning can attain strong regret guarantees and can handle observational data with constraints.

---

## 9. Evaluation Metrics

### 9.1 Why AUC Is Not Enough

ROC-AUC evaluates ranking by observed outcome:

\[
Y
\]

but uplift ranking requires ranking by:

\[
Y(1)-Y(0)
\]

Since individual uplift is not observed, evaluation must use grouped comparisons, randomized experiments, or off-policy estimators.

### 9.2 Uplift at Top \(q\)

Sort leads by predicted uplift \(\hat{\tau}(X_i)\). Let \(S_q\) be the top \(q\) fraction or top \(K\) leads.

In randomized evaluation:

\[
\widehat{Uplift}(S_q)
=\frac{1}{|S_q^1|}\sum_{i\in S_q,A_i=1}Y_i
-\frac{1}{|S_q^0|}\sum_{i\in S_q,A_i=0}Y_i
\]

Incremental conversions:

\[
\widehat{IncConv}(S_q)
=|S_q|
\cdot
\widehat{Uplift}(S_q)
\]

### 9.3 Uplift Curve

The cumulative uplift curve plots:

\[
q
\mapsto
\widehat{IncConv}(S_q)
\]

for increasing targeting depth \(q\).

The ideal model concentrates positive treatment effects early, so the curve rises steeply near the top of the ranking.

### 9.4 Qini Curve and Qini Coefficient

A common Qini-style curve compares incremental gain from targeting the top \(q\) fraction against random targeting.

One empirical form is:

\[
Q(q)
=Y_T(q)
-\frac{N_T(q)}{N_C(q)}Y_C(q)
\]

where:

| Symbol | Meaning |
|---|---|
| \(Y_T(q)\) | outcomes among treated in top \(q\) |
| \(Y_C(q)\) | outcomes among control in top \(q\) |
| \(N_T(q)\) | treated count in top \(q\) |
| \(N_C(q)\) | control count in top \(q\) |

The Qini coefficient is the area between the model's Qini curve and a random-targeting baseline.

### 9.5 AUUC

Area under the uplift curve:

\[
AUUC=\int_0^1 U(q)\,dq
\]

Empirically:

\[
\widehat{AUUC}
\approx
\sum_{j=1}^{J}
\frac{U(q_j)+U(q_{j-1})}{2}
(q_j-q_{j-1})
\]

AUUC and Qini are useful for comparing rankers, but they should be complemented by policy-value estimates at the actual capacity \(B_t\).

### 9.6 Uplift by Decile

Partition predictions into deciles \(D_1,\dots,D_{10}\), sorted by \(\hat{\tau}\). For each decile:

\[
\widehat{\tau}(D_j)
=\bar{Y}_{T,D_j}-\bar{Y}_{C,D_j}
\]

Good models show high uplift in top deciles and lower or negative uplift in bottom deciles.

### 9.7 Calibration

For bins \(B_j\):

\[
\text{calibration error}_j
=\left|
\frac{1}{|B_j|}
\sum_{i\in B_j}\hat{\tau}(X_i)
-\widehat{\tau}(B_j)
\right|
\]

Calibration matters because allocation uses effect magnitudes, not only ordering.

### 9.8 Online Randomized Holdout

The cleanest production evaluation:

1. Score eligible leads.
2. Randomly hold out a subset from intervention.
3. Apply the policy to the rest.
4. Estimate incremental impact using the randomized holdout.

For selected leads \(S\):

\[
\widehat{\tau}(S)
=\bar{Y}_{S,A=1}-\bar{Y}_{S,A=0}
\]

This is the most credible way to prove business impact.

---

## 10. Practical SOTA Ladder

### Level 0: Conversion Ranker

\[
\hat{s}(x)=P(Y=1\mid X=x)
\]

Use when no treatment/control data exists. This is the current baseline in many lead-scoring systems.

### Level 1: Segment Baseline Adjustment

\[
score'(x)=\hat{s}(x)-baseline(segment(x))
\]

This is not fully causal, but it helps downweight organic high-intent segments and surface potentially influenceable leads.

### Level 2: Two-Model Uplift

\[
\hat{\tau}(x)=\hat{\mu}_1(x)-\hat{\mu}_0(x)
\]

Strong first causal baseline when treatment/control data exists.

### Level 3: Meta-Learner Benchmark

Benchmark:

- S-Learner
- T-Learner
- X-Learner
- R-Learner
- DR-Learner

Use LightGBM, CatBoost, random forests, or calibrated generalized linear models as base learners.

### Level 4: Causal Forest / Orthogonal Forest

Use when interpretability, uncertainty intervals, and heterogeneous effect discovery matter.

### Level 5: Predict-Then-Optimize

Estimate:

\[
\hat{\tau}_i
\]

then solve:

\[
\max_A \sum_i A_i\hat{g}_i
\]

subject to capacity and business constraints.

This is often the best production architecture because causal estimation and allocation constraints are modular.

### Level 6: Multi-Action Treatment Allocation

Estimate:

\[
\hat{\tau}_a(x)
\]

then choose both who to target and what action to apply.

This is the natural SOTA direction for sales/marketing systems because different interventions have different costs and causal effects.

### Level 7: Direct Policy Learning

Optimize:

\[
\hat{\pi}
=\arg\max_{\pi\in\Pi}\hat{V}_{DR}(\pi)
\]

Use when the policy class itself matters, such as interpretable routing trees, audited rules, or constrained multi-action assignment.

### Level 8: Contextual Bandits and Offline RL

Use only when decisions are sequential or exploration is part of the system:

\[
S_t,A_t,R_t,S_{t+1}
\]

Examples:

- call now vs later
- email before call
- discount after failed contact
- repeated nurture journeys

This requires strong logging, off-policy evaluation, and careful guardrails. It is an extension, not the first implementation target.

---

## 11. Recommended Research-to-Production Design

### Data Requirements

Minimum logged fields:

| Field | Purpose |
|---|---|
| lead id | unit identity |
| timestamp | temporal ordering and leakage control |
| pre-treatment features | \(X_i\) |
| action/intervention | \(A_i\) |
| action propensity if randomized | \(e(X_i)\) or \(\pi_b(A_i\mid X_i)\) |
| outcome | \(Y_i\) |
| revenue/deal value | \(v_i\) |
| intervention cost | \(c_i\) |
| rep/territory | capacity constraints |

The most important design rule:

> Features used for CATE estimation must be measured before intervention assignment.

### Experimental Design

Preferred:

\[
A_i \sim Bernoulli(p)
\]

within eligibility strata.

For operational safety, use stratified randomization:

\[
P(A_i=1\mid strata=s)=p_s
\]

This improves overlap and gives credible estimates within important segments.

### Modeling Workflow

1. Build a conversion ranker baseline.
2. Create randomized or quasi-random treatment/control data.
3. Fit S/T/X/R/DR learners using cross-fitting.
4. Fit causal forest for heterogeneity and uncertainty analysis.
5. Compare by AUUC, Qini, uplift@K, and policy value at actual \(B_t\).
6. Convert uplift to incremental profit:

\[
\hat{g}_i=\hat{\tau}_iv_i-c_i
\]

7. Solve allocation:

\[
\max_A \sum_i A_i\hat{g}_i
\]

8. Validate with randomized holdout.

### Deployment Policy

For a first uplift deployment:

\[
A_i=1
\quad \text{if} \quad
\hat{g}_i>0
\]

and \(i\) is among the top feasible leads under constraints.

Keep a holdout group:

\[
P(\text{holdout}\mid X)=h
\]

so future impact can be estimated without relying entirely on observational corrections.

---

## 12. Common Failure Modes

### Selection Bias

Historical treatment assignment depends on lead quality:

\[
A_i \not\perp \{Y_i(1),Y_i(0)\}\mid X_i
\]

Mitigation:

- randomized holdout
- richer pre-treatment features
- propensity modeling
- DR estimators
- sensitivity analysis

### Positivity Failure

Some lead types are always or never contacted:

\[
e(x)\approx 0 \quad \text{or} \quad e(x)\approx 1
\]

Mitigation:

- exploration budget
- stratified randomization
- trimming
- avoid making causal claims in unsupported regions

### Leakage

Features include post-treatment behavior such as sales-call notes, follow-up status, or CRM stage changes after contact.

Mitigation:

- strict feature timestamping
- intervention-time feature snapshots
- leakage audits

### Optimizing CATE Instead of Policy Value

Accurate CATE estimation is not always the same as high-value allocation. A noisy CATE model can produce poor top-\(K\) decisions.

Mitigation:

- evaluate policy value at actual capacity
- use conservative scores
- compare against direct policy learning

### Misaligned Outcome

Optimizing conversion may ignore revenue, margin, churn, or discount cost.

Mitigation:

\[
Y_i = \text{profit}_i
\]

or:

\[
\hat{g}_i=\hat{\tau}_i\cdot \text{value}_i-\text{cost}_i
\]

---

## 13. Source Spine and Reading Map

### Uplift Modeling

- Devriendt, Moldovan, and Verbeke, "A Literature Survey and Experimental Evaluation of the State-of-the-Art in Uplift Modeling" (2018). Good broad survey of uplift methods and Qini/Gini-style evaluation. https://doi.org/10.1089/big.2017.0104
- Devriendt et al., "A survey and benchmarking study of multitreatment uplift modeling" (Data Mining and Knowledge Discovery, 2020). Covers multi-treatment uplift and expected response/Qini evaluation. https://link.springer.com/article/10.1007/s10618-019-00670-y

### Meta-Learners and CATE

- Kunzel, Sekhon, Bickel, and Yu, "Metalearners for estimating heterogeneous treatment effects using machine learning" (PNAS, 2019). Introduces and compares S/T/X-style meta-learning; no learner is uniformly best. https://doi.org/10.1073/pnas.1804597116
- Nie and Wager, "Quasi-oracle estimation of heterogeneous treatment effects" (Biometrika, 2021). R-Learner and residualized CATE estimation. https://doi.org/10.1093/biomet/asaa076
- Kennedy, "Towards optimal doubly robust estimation of heterogeneous causal effects" (Electronic Journal of Statistics, 2023). DR-Learner theory and model-free error bounds. https://arxiv.org/abs/2004.14497

### Causal Forests

- Athey, Tibshirani, and Wager, "Generalized Random Forests" (Annals of Statistics, 2019). Forests as adaptive local moment estimators. https://doi.org/10.1214/18-AOS1709
- Wager and Athey, "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests" (JASA, 2018). Causal forest inference. https://doi.org/10.1080/01621459.2017.1319839

### Policy Learning and Off-Policy Evaluation

- Dudik, Langford, and Li, "Doubly Robust Policy Evaluation and Learning" (ICML, 2011). Foundational contextual-bandit DR policy evaluation. https://arxiv.org/abs/1103.4601
- Athey and Wager, "Policy Learning with Observational Data" (Econometrica, 2021). Efficient policy learning with constraints and doubly robust scores. https://arxiv.org/abs/1702.02896
- Zhou, Athey, and Wager, "Offline Multi-Action Policy Learning: Generalization and Optimization" (Operations Research, 2023). Multi-action policy learning with observational data, constraints, and regret guarantees. https://doi.org/10.1287/opre.2022.2271

### Recent SOTA Context

- "From Prediction to Prescription: Machine Learning and Causal Inference for the Heterogeneous Treatment Effect" (Annual Review of Biomedical Data Science, 2025). Current bridge from HTE estimation to prescriptive decision-making. https://www.annualreviews.org/content/journals/10.1146/annurev-biodatasci-103123-095750
- "Causal Machine Learning in Information Systems Research" (Business & Information Systems Engineering, 2026). Recent overview of DML, meta-learners, causal forests, and software ecosystems. https://link.springer.com/article/10.1007/s12599-026-00999-x
- Singh, "A Large-Scale Empirical Comparison of Meta-Learners and Causal Forests for Heterogeneous Treatment Effect Estimation in Marketing Uplift Modeling" (arXiv, 2026). Recent large-scale marketing uplift benchmark. https://arxiv.org/abs/2604.06123
- "Uplift modeling with continuous treatments: A predict-then-optimize approach" (European Journal of Operational Research, 2025). Relevant for constrained decision-making, ILP post-processing, costs, and fairness constraints. https://www.sciencedirect.com/science/article/abs/pii/S0377221725008458

---

## 14. Minimal Implementation Blueprint

For this repository's current ranking system, the clean transition path is:

1. Keep the existing conversion ranker as the candidate generator.
2. Add treatment/control logging around sales intervention.
3. Reserve a randomized holdout or exploration slice.
4. Train uplift models on pre-treatment feature snapshots.
5. Score candidates by:

\[
\hat{g}_i=\hat{\tau}(X_i)v_i-c_i
\]

6. Allocate by constrained optimization:

\[
\max_A \sum_i A_i\hat{g}_i
\quad \text{s.t. operational constraints}
\]

7. Report both predictive metrics and causal metrics:

| Current metric | Add causal metric |
|---|---|
| Precision@K | uplift@K |
| Recall@K | incremental conversions@K |
| Capture rate@K | Qini/AUUC |
| conversion rate | policy value |

8. Use randomized holdout to estimate realized incremental lift.

The long-term target is not a better classifier. It is a decision system:

\[
\text{pre-treatment features}
\rightarrow
\text{uplift / action-value estimation}
\rightarrow
\text{constrained allocation}
\rightarrow
\text{randomized evaluation}
\rightarrow
\text{better policy}
\]

