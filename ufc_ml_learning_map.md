# UFC ML Project — Learning Map

A math, statistics, and ML curriculum built around the UFC prediction project.
Work through these in order — each layer builds on the last.

---

## Stage 1 — Visual Intuition (start here)

Get geometric and visual intuition before touching any formalism.
No math background required for any of these.

| Resource | What it covers | Why it matters for your project |
|---|---|---|
| [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) | Derivatives, chain rule, integrals | Gradient boosting literally descends a gradient — you need this |
| [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Vectors, matrices, dot products | How features are represented and transformed |
| [3Blue1Brown — Probability](https://www.3blue1brown.com/topics/probability) | Probability, Bayes' theorem, distributions | Your model outputs a probability — this explains what that means |
| [StatQuest with Josh Starmer](https://statquest.org) | Stats + ML, step by step with visuals | Covers bias/variance, cross-validation, decision trees, gradient boosting — all directly relevant |

**Priority StatQuest videos for this project:**
- Machine Learning Fundamentals: Bias and Variance
- Cross Validation
- Decision Trees
- Gradient Boost (Parts 1–4)
- ROC and AUC

---

## Stage 2 — Core ML Textbooks

Both are free. Read ISL first, then use ESL for depth when you want the full math.

| Resource | Link | Priority chapters |
|---|---|---|
| **An Introduction to Statistical Learning** (Python ed.) — James, Witten, Hastie, Tibshirani | [statlearning.com](https://www.statlearning.com) | Ch. 2 (statistical learning), Ch. 5 (cross-validation), Ch. 8 (tree-based methods) |
| **The Elements of Statistical Learning** — Hastie, Tibshirani, Friedman | [hastie.su.domains/ElemStatLearn](https://hastie.su.domains/ElemStatLearn/) | Ch. 7 (model assessment), Ch. 10 (boosting) |

> **Note:** ISL has a free PDF and a companion edX course. ESL is denser —
> treat it as a reference rather than a cover-to-cover read.

---

## Stage 3 — The Math Behind Your Model Specifically

Once you have the ISL foundation, these are the primary sources for exactly
what your GBM is doing under the hood.

| Resource | Link | What it covers |
|---|---|---|
| **Greedy Function Approximation: A Gradient Boosting Machine** — Friedman (2001) | [projecteuclid.org](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-a-gradient-bosting-machine/10.1214/aos/1013203451.full) | The original GBM paper. Derives the loss function, the tree-building algorithm, and the gradient descent framing from scratch |
| **XGBoost: A Scalable Tree Boosting System** — Chen & Guestrin (2016) | [arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754) | The paper behind modern gradient boosting. Adds regularization, second-order gradients, and the system design that makes it fast |

---

## Stage 4 — Time-Series Evaluation

The most important thing to get right for sports prediction that most ML
courses underemphasize. Covers why random train/test splits are wrong for
your use case, and how to measure whether your win probabilities are
actually calibrated.

| Resource | Link | Priority chapters |
|---|---|---|
| **Forecasting: Principles and Practice** — Hyndman & Athanasopoulos (free online) | [otexts.com/fpp3](https://otexts.com/fpp3/) | Ch. 5 (evaluating forecast accuracy, Brier score, proper scoring rules) |

---

## Stage 5 — Elo & Rating Systems

Directly relevant to what you built in `elo.py`.

| Resource | Link | What it covers |
|---|---|---|
| **FiveThirtyEight: How We Calculate NBA Elo Ratings** | [fivethirtyeight.com](https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/) | How 538 tuned K-factors, margin of victory adjustments, and home court — directly applicable to MMA Elo |
| **TrueSkill: A Bayesian Skill Rating System** — Herbrich, Minka, Graepel | [microsoft.com/research](https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/) | The Bayesian generalization of Elo. Read this once you're comfortable with Elo to understand its limitations |

---

## Recommended Order

```
1. 3B1B Calculus + Linear Algebra + Probability  (a few weeks, YouTube)
2. StatQuest: Bias/Variance, Cross Validation, Decision Trees, Gradient Boost
3. ISL Ch. 2, 5, 8  (free PDF at statlearning.com)
4. Hyndman Ch. 5    (free online at otexts.com/fpp3)
5. Friedman 2001 GBM paper
6. ESL Ch. 10       (boosting deep dive)
7. XGBoost paper
8. FiveThirtyEight Elo article + TrueSkill paper
```

---

## Concepts Mapped to Your Codebase

| Concept | Where it shows up in your project |
|---|---|
| Derivatives / gradient descent | How `GradientBoostingClassifier` minimizes log-loss across 200 trees |
| Bias-variance tradeoff | Why `max_depth=3` and `subsample=0.8` — shallow trees + subsampling reduce variance |
| Cross-validation | `walk_forward_cv()` in `model.py` — temporal folds, not random splits |
| Logistic function | The output layer of your classifier converting raw scores to probabilities |
| Log-loss / Brier score | How to evaluate whether your win probabilities are calibrated, not just accurate |
| Elo update formula | `elo.py` — expected score, K-factor, rating delta |
| Feature differencing | `diff_win_rate`, `diff_elo` etc. — linear algebra concept of projecting onto a difference basis |
