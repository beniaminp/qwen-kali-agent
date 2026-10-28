# 47 — Machine learning fundamentals

*Last verified: 28 October 2026.*

This chapter teaches classical machine learning the way you will use it as an AI engineer: as the fast, cheap, explainable tool that wins on tabular data and narrow decisions, and as the set of ideas (splits, leakage, baselines, overfitting, metrics) that every LLM evaluation reuses. You will fit linear and logistic regression by hand and in scikit-learn, compute a decision-tree split, a k-NN vote, two k-means iterations and a naive Bayes posterior on paper, measure overfitting, and compute every common metric, ROC-AUC included, from a handful of examples. It ends with a complete scikit-learn pipeline for routing support tickets, a cost comparison against an LLM, and the places classical models still sit inside LLM systems. Read [46 — The math you actually need](46-ai-math.md) first (dot products, derivatives, log-loss, standard errors); [48 — Deep learning](48-ai-deep-learning.md) continues from here. Every code block was run with scikit-learn 1.9 and numpy 2.5, and the output under it is the real output.

---

## 1. What "learning from data" means

### 1.1 Rules versus learned functions

In ordinary software you write the rule: `if "refund" in text: queue = "billing"`. In machine learning you write a function with adjustable numbers (**parameters**, or **weights**), show it examples of inputs with the right answers, and let an algorithm adjust the numbers until the function's answers match. The result is a **model**: a function \(f_\theta(x)\approx y\), where \(\theta\) is the list of parameters.

Why you care: a learned rule can use thousands of weak signals at once (every word, the customer's tier, the time of day) and can be retrained when the data changes, instead of growing a 2,000-line `if` chain that nobody dares to edit. The price is that its behaviour is only as good as its data, and you can no longer read its logic line by line.

### 1.2 The three ingredients

Every learning method, from linear regression to Qwen3-14B, is the same three choices. (Qwen3's pretraining is *self-supervised*: the label for each position is simply the next token of the text itself, so no one labels the data; its later instruction tuning is ordinary supervised learning. See [49](49-ai-llm-internals.md).)

| Ingredient | Question it answers | Linear regression | Qwen3-14B |
|---|---|---|---|
| Data | what examples do we learn from? | rows of \((x, y)\) | trillions of tokens |
| Model family | which functions are allowed? | \(wx+b\) | a 40-layer transformer |
| Loss + optimizer | what counts as wrong, and how do we reduce it? | squared error, closed form | cross-entropy, AdamW |

An analogy that holds up: training is like a CI job that tunes a config until a test suite passes. The training set is the test suite you are allowed to see, the loss is the number of failing assertions, and the danger is a config that passes those tests by special-casing each one.

### 1.3 Generalisation is the goal

A model is useful only if it does well on examples it has **not** seen. Memorising the training set is easy: a lookup table does it perfectly and is useless on the next ticket. Everything in §4–§5 and §14–§17 (splits, leakage, overfitting, regularization, cross-validation) exists to measure and protect performance on unseen data.

---

## 2. Problem types

The first decision in any ML project is what kind of output you need. It fixes the model family, the loss and the metric.

### 2.1 Classification

Predict one label from a fixed set. **Binary**: urgent or not, spam or not, "will this invoice extraction need human review?". **Multiclass**: one of billing, bug, account, how-to. **Multilabel**: any subset of tags ("billing" *and* "angry"). The model usually outputs a probability per class, and you turn probabilities into decisions with a threshold (§8.5).

### 2.2 Regression

Predict a number: handling time of a ticket, latency of a request, tokens an answer will need, next month's GPU bill. Loss and metrics measure the size of the error (§19).

### 2.3 Ranking

Order a list of items for a query: search results, the chunks a RAG system puts into the prompt, products to recommend. What matters is the order, especially at the top, not the absolute scores. Metrics are recall@k, MRR and nDCG ([51](51-ai-rag.md)).

### 2.4 Clustering

Group items with no labels at all: cluster 50,000 user questions to discover what people ask about, or cluster failed agent runs to find the common failure modes. k-means (§12) is the classic method.

### 2.5 Anomaly detection

Flag the rare item that does not look like the rest: a sudden spike in tokens per request (a prompt-injection loop, a runaway agent), a fraudulent payment, a log line never seen before. Often there are almost no labelled anomalies, so the model learns what "normal" looks like.

### 2.6 Generation

Produce a new structured output: text, code, an image. This is what LLMs do. It is classification repeated: at each step, a classifier over the vocabulary picks the next token ([49](49-ai-llm-internals.md)).

### 2.7 Choosing the type

| Product question | Type | Typical first model | Metric |
|---|---|---|---|
| Which queue does this ticket go to? | multiclass classification | logistic regression on TF-IDF | macro-F1 |
| Is this transaction fraud? | binary classification, imbalanced | gradient boosting | PR-AUC, recall at fixed precision |
| How long will this ticket take? | regression | gradient boosting | MAE |
| Which 5 documents answer this query? | ranking | BM25, then a reranker | recall@5, nDCG |
| What are users asking about? | clustering | k-means on embeddings | manual review |
| Is this agent run abnormal? | anomaly detection | z-score or isolation forest | precision of alerts |
| Write the reply | generation | an LLM | evals ([54](54-ai-evaluation.md)) |

Many products are a chain of these: classify the ticket, retrieve and rank documents, generate a reply.

---

## 3. Features and labels

### 3.1 The feature matrix and the label vector

A **feature** is one input column the model can use; a **label** (or target) is the answer you want predicted. A dataset of \(n\) examples with \(d\) features is a matrix \(X\) of shape \((n, d)\) and a label vector \(\mathbf y\) of length \(n\):

| ticket | `mentions_outage` | `tier_enterprise` | `exclamation_marks` | `account_age_days` | label: urgent |
|---|---:|---:|---:|---:|---:|
| #1041 | 1 | 1 | 2 | 812 | 1 |
| #1042 | 0 | 0 | 0 | 15 | 0 |
| #1043 | 0 | 1 | 1 | 301 | 0 |

Each row is one example and each column one feature; a model is a function from a row to a label. In scikit-learn, `X` is 2-D `(n_samples, n_features)` and `y` is 1-D `(n_samples,)` (text vectorizers are the exception: they take a 1-D list of strings and produce the 2-D matrix); [46 §2](46-ai-math.md) covers shapes.

### 3.2 Where labels come from, and label noise

Labels come from people (agents tagging tickets), from logs (the queue a ticket was finally resolved in), or from rules and LLMs ("weak" labels). All of them are noisy. If two support agents disagree on 8% of tickets, no model can score much above 92% against either of them, and a model that does is probably fitting one agent's quirks. Measure label agreement on a sample before you set a target ([54](54-ai-evaluation.md) covers Cohen's kappa). Labelling guidelines, annotation tools and labelling at scale are in [60 — Data preparation](60-ai-data-preparation.md).

---

## 4. Train, validation and test splits

### 4.1 Why three sets

You split the labelled data into three parts with three jobs:

- **Training set**: the model's parameters are fitted on it.
- **Validation set** (or dev set): you compare models and choose settings (hyperparameters) on it.
- **Test set**: you measure the final, chosen model on it, once, and report that number.

How big must the test set be? Big enough that its accuracy is precise. The standard error of an accuracy (from [46 §13.3](46-ai-math.md)) tells you:

\[
\operatorname{SE}=\sqrt{\frac{p(1-p)}{n_{\text{test}}}}
\]

!!! example "On paper"

    10,000 labelled tickets split 70/15/15 gives 7,000 / 1,500 / 1,500. For an accuracy near 0.9:

    \[
    \begin{aligned}
    n=1500 &: \sqrt{0.09/1500} \approx 0.0077\\
    1.96\cdot0.0077 &\approx 0.015\\
    n=200 &: \sqrt{0.09/200} \approx 0.0212\\
    1.96\cdot0.0212 &\approx 0.042
    \end{aligned}
    \]

    1,500 test tickets measure accuracy to about ±1.5 points; 200 only to ±4.2 points, so two models 3 points apart are indistinguishable on 200.

### 4.2 Do not tune on the test set

If you try 50 configurations and pick the one with the best test score, the test score of the winner is optimistic: you picked it partly for its lucky noise. The test set has become a validation set, and you no longer have an unbiased estimate.

\[
\text{optimism}\approx\mathbb E[\max\text{ of }m\text{ noise draws}]\cdot\operatorname{SE}
\]

!!! example "On paper"

    50 configurations that are all truly equal, on the 1,500-ticket test set (SE 0.0077). The expected maximum of 50 independent standard normal draws is about 2.25:

    \[
    \begin{aligned}
    \text{optimism} &\approx 2.25\cdot0.0077\\
    &\approx 0.017
    \end{aligned}
    \]

    The "best" configuration looks about 1.7 points better than it is, on average, from noise alone (less if the configurations make correlated mistakes). Choose on validation, report on a test set you touch once.

### 4.3 Stratified splits

A **stratified** split keeps the class proportions the same in every part. It matters when one class is rare.

\[
\operatorname{sd}(\text{count})=\sqrt{n\,p\,(1-p)}
\]

!!! example "On paper"

    5% of tickets are urgent, and a random test set has 200 tickets:

    \[
    \begin{aligned}
    \text{expected urgent} &= 200\cdot0.05 = 10\\
    \text{sd} &= \sqrt{200\cdot0.05\cdot0.95} \approx 3.08
    \end{aligned}
    \]

    A random split puts anywhere from about 4 to 16 urgent tickets in the test set (10 ± 2 sd), and recall measured on 4 positives can only be 0, 0.25, 0.5, 0.75 or 1. `train_test_split(..., stratify=y)` puts exactly 10 there.

### 4.4 Time-based splits

When the model will predict the future, split by time: train on January to August, validate on September, test on October. A random split lets the model see October tickets while training and learn things it could not know in production (a product launched in October, a new error message). Random splits on time-dependent data are optimistic; time splits show the real decay you will see after deployment. Re-run the same split monthly to see how fast a model goes stale.

### 4.5 Group splits

When several rows share an origin, keep each origin entirely on one side of the split: all tickets of one customer, all chunks of one document, all functions of one repository, all turns of one conversation. Otherwise the model learns the customer's writing style, the document's vocabulary or the repo's naming, and the test set rewards memorisation. `GroupShuffleSplit` and `GroupKFold` do this.

### 4.6 Splits in code

```python
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

rng = np.random.default_rng(0)
n = 1000
y = rng.permutation(np.array([1] * 50 + [0] * 950))     # exactly 5% urgent
customer = rng.integers(0, 100, size=n)                 # 100 customers, ~10 tickets each
X = rng.normal(size=(n, 3))

for name, strat in [("random    ", None), ("stratified", y)]:
    _, _, _, y_test = train_test_split(X, y, test_size=200, stratify=strat, random_state=3)
    print(f"{name}: urgent tickets in a 200-ticket test set = {y_test.sum()}")

train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
                           .split(X, y, groups=customer))
shared = set(customer[train_idx]) & set(customer[test_idx])
print("customers in both train and test:", len(shared))
```

```text
random    : urgent tickets in a 200-ticket test set = 6
stratified: urgent tickets in a 200-ticket test set = 10
customers in both train and test: 0
```

The random split happened to put 6 urgent tickets in the test set instead of 10, enough to move a measured recall by a wide margin; the stratified split puts exactly 10. The group split guarantees that no customer appears on both sides.

---

## 5. Data leakage: three stories

**Leakage** is information in the training data that will not be available, or will not look the same, when the model runs in production. It makes offline scores look great and production scores collapse. It is the most expensive bug in applied ML because nothing crashes.

### 5.1 The field that is filled in later

A team trains an urgency classifier on exported tickets and gets 0.99 AUC. One of the columns is `assigned_to_oncall`. Agents set it *after* they decided a ticket was urgent, so the model learned to read the answer. In production, a new ticket has no assignee yet and the model is close to random. This is **target leakage**. The fix: for every feature, ask "is this value known at the moment the prediction is made?" and build training rows from snapshots taken at that moment, not from the final state of the record.

### 5.2 The duplicate customer

A code-review assistant classifies diffs as risky or safe. Diffs from the same repository share file names, helper functions and authors. With a random split, near-identical diffs land on both sides and the test score measures memory. When the model is rolled out to a new repository, it drops 15 points. The same happens when a customer sends the same complaint by email and by chat, or when one long document is chunked and chunks are split randomly. The fix: deduplicate (exact and near-duplicate, for example by cosine similarity above 0.95; [60](60-ai-data-preparation.md) covers deduplication methods at scale), and use group splits (§4.5).

### 5.3 Preprocessing on all the data

A scaler, a TF-IDF vocabulary, an imputation mean or a feature-selection step fitted on the whole dataset has seen the test rows. Feature selection is the dramatic case: with enough random features, some correlate with the labels by chance, and choosing them on all the data makes pure noise look predictive. Here the labels are coin flips:

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5000))          # 5,000 random "features"
y = rng.integers(0, 2, size=100)          # random labels: there is nothing to learn

# WRONG: pick the 20 features most related to y on ALL rows, then cross-validate
X_leaky = SelectKBest(f_classif, k=20).fit_transform(X, y)
wrong = cross_val_score(LogisticRegression(), X_leaky, y, cv=5).mean()

# RIGHT: selection inside the pipeline, so each fold selects on its own training rows
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
right = cross_val_score(pipe, X, y, cv=5).mean()
print(f"leaky CV accuracy : {wrong:.2f}")
print(f"honest CV accuracy: {right:.2f}")
```

```text
leaky CV accuracy : 0.85
honest CV accuracy: 0.48
```

The leaky version reports a strong model on data with no signal at all. The honest one reports a coin flip, which is the truth. The fix is structural: put every fitted preprocessing step inside a scikit-learn `Pipeline` (§22), so cross-validation refits it on each training fold.

### 5.4 A leakage checklist

- Every feature: is it known at prediction time, with the same value it had then?
- Rows from the same user, document, repo or conversation: are they all on one side of the split?
- Exact and near-duplicates: removed, or grouped?
- Time: does the training data end before the test data begins?
- Preprocessing: fitted inside the pipeline, on training folds only?
- A score that looks too good (accuracy 0.99 on a messy task): look for leakage before celebrating.

The same list applies to LLM evals: a golden set that leaked into fine-tuning data, or into a benchmark the model was pretrained on, is the same bug ([54](54-ai-evaluation.md)).

---

## 6. Baselines

A **baseline** is the simplest thing that could work. Measure it before any model, because a model is only as good as its margin over the baseline.

### 6.1 The majority class

Always predict the most common class.

\[
\text{accuracy}_{\text{majority}}=\frac{n_{\text{majority}}}{n}
\]

!!! example "On paper"

    1,000 tickets, of which 100 are urgent. "Never urgent" gets:

    \[
    \begin{aligned}
    \text{accuracy} &= 900/1000 = 0.90\\
    \text{recall on urgent} &= 0/100 = 0
    \end{aligned}
    \]

    90% accuracy while missing every urgent ticket. A model that reports 0.91 accuracy on this data has barely learned anything (§18.2).

### 6.2 A simple heuristic

Write the rule an engineer would write in ten minutes, and score it like a model.

\[
\text{precision}=\frac{\text{correct flags}}{\text{flags}},
\quad
\text{recall}=\frac{\text{correct flags}}{\text{all positives}}
\]

!!! example "On paper"

    The rule "text contains outage, down or urgent" flags 80 of the 1,000 tickets; 50 of those are really urgent:

    \[
    \begin{aligned}
    \text{precision} &= 50/80 = 0.625\\
    \text{recall} &= 50/100 = 0.5
    \end{aligned}
    \]

    The bar for a model is now "beat 0.625 precision at 0.5 recall", not "beat 0". If a model and an LLM cannot clearly beat a keyword rule, ship the rule.

### 6.3 Baselines in code

```python
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score

rng = np.random.default_rng(0)
y = np.array([1] * 100 + [0] * 900)                     # 100 urgent of 1,000
X = rng.normal(size=(1000, 3))                          # features do not matter here

for strategy in ("most_frequent", "stratified"):
    pred = DummyClassifier(strategy=strategy, random_state=0).fit(X, y).predict(X)
    print(f"{strategy:13s} accuracy={accuracy_score(y, pred):.3f}  recall={recall_score(y, pred):.3f}")
```

```text
most_frequent accuracy=0.900  recall=0.000
stratified    accuracy=0.810  recall=0.090
```

`stratified` guesses at random in the class proportions: its accuracy is near \(0.9^2+0.1^2=0.82\) and its recall near 0.1. Put both rows in every results table.

---

## 7. Linear regression

### 7.1 The model and the loss

**Linear regression** predicts a number as a weighted sum of the features plus a constant: \(\hat y=wx+b\) with one feature, \(\hat y=\mathbf w\cdot\mathbf x+b\) with many. It is fitted by minimising the **mean squared error** (MSE).

\[
\operatorname{MSE}(w,b)=\frac1n\sum_{i=1}^{n}\bigl(wx_i+b-y_i\bigr)^2
\]

!!! example "On paper"

    An invoice extractor takes 2, 3 and 7 seconds on invoices of 1, 2 and 3 pages. Try the guess \(w=2\), \(b=0\):

    \[
    \begin{aligned}
    \hat y &= (2,4,6)\\
    \hat y-y &= (0,1,-1)\\
    \operatorname{MSE} &= (0+1+1)/3 \approx 0.667
    \end{aligned}
    \]

    The question is which \(w\) and \(b\) give the smallest MSE. For linear regression there is a formula (§7.2); for almost everything else you use gradient descent (§7.4).

### 7.2 The closed form by hand

For one feature, the best slope is the covariance of \(x\) and \(y\) divided by the variance of \(x\), and the line passes through the point of means.

\[
w=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sum_i(x_i-\bar x)^2},
\quad
b=\bar y-w\bar x
\]

!!! example "On paper"

    Same data: \(x=(1,2,3)\), \(y=(2,3,7)\), so \(\bar x=2\) and \(\bar y=4\):

    \[
    \begin{aligned}
    x-\bar x &= (-1,0,1)\\
    y-\bar y &= (-2,-1,3)\\
    w &= (2+0+3)/(1+0+1) = 2.5\\
    b &= 4-2.5\cdot2 = -1\\
    \hat y &= (1.5,4,6.5)\\
    \operatorname{MSE} &= (0.25+1+0.25)/3 = 0.5
    \end{aligned}
    \]

    0.5 beats the guess's 0.667, and no other line does better. The model predicts \(2.5\cdot4-1=9\) seconds for a 4-page invoice.

### 7.3 The same answer from the normal equations

With many features, put a column of ones in \(X\) for the intercept. The best weights solve the **normal equations**, which use the matrix inverse from [46 §4.6](46-ai-math.md).

\[
\boldsymbol\theta=(X^\top X)^{-1}X^\top\mathbf y
\]

!!! example "On paper"

    \(X\) has rows \((1,1)\), \((1,2)\), \((1,3)\); \(\mathbf y=(2,3,7)\):

    \[
    \begin{aligned}
    X^\top X &= \text{rows } (3,6),\ (6,14)\\
    X^\top\mathbf y &= (12,\ 29)\\
    \det &= 3\cdot14-6\cdot6 = 6\\
    (X^\top X)^{-1} &= \tfrac16\,\text{rows } (14,-6),\ (-6,3)\\
    \boldsymbol\theta &= \tfrac16(168-174,\ -72+87) = (-1,\ 2.5)
    \end{aligned}
    \]

    Intercept −1 and slope 2.5, as in §7.2. Libraries solve this system with a stable factorisation rather than an explicit inverse.

### 7.4 Gradient descent on the same data

The MSE's gradient comes from the chain rule ([46 §5.4](46-ai-math.md)): \(\partial\,\text{MSE}/\partial w=\frac2n\sum(\hat y_i-y_i)x_i\) and \(\partial\,\text{MSE}/\partial b=\frac2n\sum(\hat y_i-y_i)\). Gradient descent reaches the same line without any formula:

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])      # pages
y = np.array([2.0, 3.0, 7.0])      # seconds
w, b, lr = 0.0, 0.0, 0.05

for step in range(2001):
    err = w * x + b - y                         # prediction minus truth
    if step in (0, 1, 10, 100, 500, 2000):
        print(f"step {step:4d}: w={w:6.3f} b={b:6.3f} MSE={np.mean(err ** 2):7.4f}")
    w -= lr * 2 * np.mean(err * x)              # dMSE/dw
    b -= lr * 2 * np.mean(err)                  # dMSE/db
```

```text
step    0: w= 0.000 b= 0.000 MSE=20.6667
step    1: w= 0.967 b= 0.400 MSE= 4.8452
step   10: w= 1.814 b= 0.559 MSE= 0.8486
step  100: w= 2.269 b=-0.475 MSE= 0.5395
step  500: w= 2.498 b=-0.996 MSE= 0.5000
step 2000: w= 2.500 b=-1.000 MSE= 0.5000
```

The loss falls quickly at first and then crawls towards 0.5, while \(w\) and \(b\) creep to 2.5 and −1. That slow tail is typical: \(w\) and \(b\) are correlated (raising one and lowering the other changes little), the narrow-valley case of [46 §6.4](46-ai-math.md).

### 7.5 scikit-learn

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1.0], [2.0], [3.0]])         # 2-D: (n_samples, n_features)
y = np.array([2.0, 3.0, 7.0])
model = LinearRegression().fit(X, y)
print("w =", round(model.coef_[0], 4), " b =", round(model.intercept_, 4))
print("prediction for 4 pages:", round(model.predict([[4.0]])[0], 4))
```

```text
w = 2.5  b = -1.0
prediction for 4 pages: 9.0
```

Every scikit-learn model has the same interface: `fit(X, y)`, then `predict(X)` (and, for most classifiers, `predict_proba(X)`). Learn it once.

### 7.6 Pitfalls

- **Extrapolation.** The fitted line knows nothing beyond the data. A 300-page invoice is not \(2.5\cdot300-1\) seconds if the service paginates or times out.
- **Outliers.** Squared error punishes big errors heavily, so one bad row can tilt the line. Look at residuals; consider MAE-based or robust models.
- **Correlated features.** With two nearly identical features, the individual weights become unstable (one +50, the other −49) while predictions stay fine. Do not read such weights as importance. Regularization (§15) stabilises them.

---

## 8. Logistic regression

### 8.1 The sigmoid

**Logistic regression** is the standard first model for classification. It computes a linear score \(z=\mathbf w\cdot\mathbf x+b\) (the **logit**) and squashes it into a probability with the sigmoid, exactly the two-class softmax of [46 §10.3](46-ai-math.md).

\[
p=\sigma(z)=\frac{1}{1+e^{-z}}
\]

!!! example "On paper"

    Three logits, with \(\ln9\approx2.197\):

    \[
    \begin{aligned}
    \sigma(0) &= 1/(1+1) = 0.5\\
    \sigma(\ln9) &= 1/(1+1/9) = 0.9\\
    \sigma(-\ln9) &= 1/(1+9) = 0.1
    \end{aligned}
    \]

    A logit of 0 is a coin flip; ±2.2 is 90/10. Each weight says how much one unit of its feature adds to the log-odds.

### 8.2 Log-loss

Logistic regression is fitted by minimising **log-loss** (binary cross-entropy): the negative log of the probability the model gave to the true label. It is the NLL of [46 §11.3](46-ai-math.md).

\[
\ell=-\bigl[y\ln p+(1-y)\ln(1-p)\bigr]
\]

!!! example "On paper"

    Four tickets as (label, predicted probability of urgent): (1, 0.9), (0, 0.2), (1, 0.6), (0, 0.7):

    \[
    \begin{aligned}
    -\ln0.9 &\approx 0.105\\
    -\ln(1-0.2) &\approx 0.223\\
    -\ln0.6 &\approx 0.511\\
    -\ln(1-0.7) &\approx 1.204\\
    \text{mean} &\approx 2.043/4 \approx 0.511
    \end{aligned}
    \]

    The confident mistake (0.7 on a non-urgent ticket) costs more than the other three together. A prediction of 0.01 for a true positive would cost 4.6: log-loss punishes overconfidence hard, which is why it pushes models towards honest probabilities.

### 8.3 The gradient is (p − y) times x

Combine the sigmoid and the log-loss through the chain rule and the gradient comes out remarkably simple: the error in probability times the input.

\[
\frac{\partial\ell}{\partial\mathbf w}=(p-y)\,\mathbf x,
\quad
\frac{\partial\ell}{\partial b}=p-y
\]

!!! example "On paper"

    The ticket with \(y=1\), \(p=0.9\) and features \(\mathbf x=(2,0)\); then the one with \(y=0\), \(p=0.7\), \(\mathbf x=(1,1)\):

    \[
    \begin{aligned}
    p-y &= 0.9-1 = -0.1\\
    \partial\ell/\partial\mathbf w &= -0.1\cdot(2,0) = (-0.2,\ 0)\\
    p-y &= 0.7-0 = 0.7\\
    \partial\ell/\partial\mathbf w &= 0.7\cdot(1,1) = (0.7,\ 0.7)
    \end{aligned}
    \]

    Nearly-right examples barely move the weights; badly wrong ones move them a lot. Features that are 0 receive no update. The same \(p-y\) is the gradient of softmax cross-entropy in an LLM ([43 §2](43-math-examples.md#2-cross-entropy-gradient)).

### 8.4 Fitting it

Generate tickets whose true urgency follows known weights, then check that logistic regression recovers them:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
n = 5000
outage = rng.random(n) < 0.2                        # mentions an outage
enterprise = rng.random(n) < 0.3                    # enterprise customer
exclaims = rng.poisson(1.0, n)                      # number of "!"
X = np.column_stack([outage, enterprise, exclaims]).astype(float)

true_w, true_b = np.array([2.0, 1.0, 0.5]), -3.0
p_true = 1 / (1 + np.exp(-(X @ true_w + true_b)))
y = (rng.random(n) < p_true).astype(int)            # urgent with that probability

model = LogisticRegression(C=np.inf).fit(X, y)      # C=inf: no regularization
print("learned w:", np.round(model.coef_[0], 2), " b:", round(model.intercept_[0], 2))
ticket = np.array([[1, 1, 2]])                       # outage, enterprise, two "!"
print("P(urgent), model :", round(model.predict_proba(ticket)[0, 1], 3))
print("P(urgent), truth :", round(1 / (1 + np.exp(-(ticket @ true_w + true_b)))[0], 3))
```

```text
learned w: [1.99 1.02 0.53]  b: -3.02
P(urgent), model : 0.74
P(urgent), truth : 0.731
```

The truth for that ticket is \(\sigma(2+1+0.5\cdot2-3)=\sigma(1)\approx0.731\). With 5,000 examples the fit lands within a few hundredths on every weight. In scikit-learn, `C` is the *inverse* regularization strength: smaller `C`, stronger L2 penalty (§15). The default is `C=1.0`. Since version 1.8 the `penalty` argument is deprecated; you choose L1, L2 or elastic net with `l1_ratio` (0, the default, is L2; 1 is L1; values in between mix the two; anything above 0 needs a solver that supports it, such as `solver="saga"`) and switch the penalty off with `C=np.inf`.

### 8.5 The decision threshold

`predict` uses a threshold of 0.5, which is right only when both kinds of mistake cost the same. Predict "urgent" when the expected cost of ignoring the ticket exceeds the expected cost of escalating it: \(p\,C_{\text{FN}}>(1-p)\,C_{\text{FP}}\).

\[
t^*=\frac{C_{\text{FP}}}{C_{\text{FP}}+C_{\text{FN}}}
\]

!!! example "On paper"

    Escalating a non-urgent ticket wastes 5 minutes of an engineer ($5). Missing an urgent one breaches an SLA ($50):

    \[
    \begin{aligned}
    t^* &= 5/(5+50) = 5/55\\
    &\approx 0.091
    \end{aligned}
    \]

    Escalate everything above 9%, not above 50%. This only works if the probabilities are calibrated (§18.9). With equal costs the formula gives back 0.5.

---

## 9. Decision trees

### 9.1 How a tree predicts

A **decision tree** is a learned `if`/`else` chain: "is the customer enterprise? if yes, urgent; if no, does the text mention an outage? ...". Each internal node tests one feature against a threshold, and each leaf holds a prediction (the majority class, or the class proportions). Trees handle mixed feature types, need no scaling, capture interactions ("outage *and* enterprise"), and a shallow tree can be printed and read by a product manager.

### 9.2 Gini impurity

The training algorithm picks, at each node, the question that makes the children **purest**. The usual measure is **Gini impurity**: the chance that two items drawn at random from the node (with replacement) have different labels.

\[
G=1-\sum_k p_k^2
\]

!!! example "On paper"

    A node with 10 tickets: 6 urgent and 4 not:

    \[
    \begin{aligned}
    G &= 1-0.6^2-0.4^2\\
    &= 1-0.36-0.16 = 0.48
    \end{aligned}
    \]

    A pure node scores 0; a 50/50 node scores 0.5, the worst for two classes. Entropy ([46 §12.2](46-ai-math.md)) is the other common choice and usually picks the same splits.

### 9.3 Choosing a split by hand

The quality of a split is the parent's impurity minus the children's impurity weighted by their size: the **impurity decrease** (or gain).

\[
\Delta G=G_{\text{parent}}-\frac{n_L}{n}G_L-\frac{n_R}{n}G_R
\]

!!! example "On paper"

    Same 10 tickets (6 urgent). Candidate A, "mentions outage": yes = 5 tickets (4 urgent, 1 not), no = 5 (2 urgent, 3 not):

    \[
    \begin{aligned}
    G_L &= 1-0.8^2-0.2^2 = 0.32\\
    G_R &= 1-0.4^2-0.6^2 = 0.48\\
    \text{weighted} &= 0.5\cdot0.32+0.5\cdot0.48 = 0.40\\
    \Delta G_A &= 0.48-0.40 = 0.08
    \end{aligned}
    \]

    The outage question helps a little.

!!! example "On paper"

    Candidate B, "enterprise customer": yes = 4 tickets (all urgent), no = 6 (2 urgent, 4 not):

    \[
    \begin{aligned}
    G_L &= 1-1^2 = 0\\
    G_R &= 1-(1/3)^2-(2/3)^2 = 4/9 \approx 0.444\\
    \text{weighted} &= 0.4\cdot0+0.6\cdot0.444 \approx 0.267\\
    \Delta G_B &\approx 0.48-0.267 = 0.213
    \end{aligned}
    \]

    B's gain is larger, so the tree asks about the tier first. The algorithm repeats this for every feature and threshold at every node: greedy, fast, and not guaranteed to find the best tree overall.

### 9.4 A tree in code

The same 10 tickets, as columns `[mentions_outage, enterprise]`:

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

X = np.array([[1, 1], [1, 1], [1, 1], [0, 1], [1, 0],
              [0, 0], [1, 0], [0, 0], [0, 0], [0, 0]])
y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])        # 1 = urgent

tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
print(export_text(tree, feature_names=["mentions_outage", "enterprise"]))
print("feature_importances_:", np.round(tree.feature_importances_, 4))
```

```text
|--- enterprise <= 0.50
|   |--- mentions_outage <= 0.50
|   |   |--- class: 0
|   |--- mentions_outage >  0.50
|   |   |--- class: 0
|--- enterprise >  0.50
|   |--- class: 1

feature_importances_: [0.0725 0.9275]
```

The root splits on `enterprise`, as computed. Inside the non-enterprise branch the tree then tries `mentions_outage`; both leaves still predict "not urgent" (the outage leaf is a 1–1 tie), so that split changes no prediction. `feature_importances_` is each feature's impurity decrease, weighted by the share of tickets that reach the node, normalised to sum to 1. Enterprise: \(1.0\times0.2133\). Outage, in a branch holding 6 of 10 tickets: \(0.6\times(0.4444-0.4167)=0.0167\). Normalised by their total 0.23, that is 0.9275 and 0.0725.

### 9.5 Pitfall: deep trees memorise

Grown without limits, a tree keeps splitting until every leaf is pure, which usually means one leaf per training example: perfect training accuracy, poor validation accuracy (§14). Limit `max_depth`, set `min_samples_leaf` (for example 20), or, better, average many trees (§10).

---

## 10. Random forests and gradient boosting

### 10.1 Random forests: averaging decorrelated trees

A **random forest** trains hundreds of deep trees, each on a bootstrap sample of the rows (rows drawn with replacement) and each split choosing among a random subset of the features, then averages their votes. One deep tree has low bias and high variance; averaging cuts the variance. How much depends on how correlated the trees are.

\[
\operatorname{Var}(\text{average})=\rho\,\sigma^2+\frac{1-\rho}{B}\,\sigma^2
\]

!!! example "On paper"

    \(B=100\) trees, each with prediction variance \(\sigma^2=1\):

    \[
    \begin{aligned}
    \rho=0 &: 0+1/100 = 0.01\\
    \rho=0.3 &: 0.3+0.7/100 = 0.307\\
    \rho=1 &: 1+0 = 1
    \end{aligned}
    \]

    Identical trees gain nothing from averaging; correlation sets the floor. That is why forests randomise the features at each split: to make the trees disagree.

### 10.2 Gradient boosting: fit the residuals

**Gradient boosting** builds trees one after another. Each new small tree is fitted to the errors (residuals) of the ensemble so far, and its predictions are added with a small **learning rate** \(\eta\). For squared error the residual is the negative gradient (exactly, for half the squared error \(\tfrac12(y-F)^2\)), which gives the method its name.

\[
F_{m}(x)=F_{m-1}(x)+\eta\,h_m(x)
\]

!!! example "On paper"

    The invoice data again, \(y=(2,3,7)\). Start from the mean, \(F_0=4\). A one-split tree ("pages ≤ 2?") fits the residuals with leaf means; \(\eta=0.5\):

    \[
    \begin{aligned}
    \text{residuals} &= (-2,-1,3)\\
    h_1 &= (-1.5,-1.5,3)\\
    F_1 &= 4+0.5\,h_1 = (3.25,3.25,5.5)\\
    \text{new residuals} &= (-1.25,-0.25,1.5)\\
    \operatorname{MSE} &: 14/3 \approx 4.667 \to 3.875/3 \approx 1.292
    \end{aligned}
    \]

    One small step removed most of the error, and the next tree will fit what is left. Hundreds of such steps with depth-limited trees make the strongest general-purpose model for tabular data.

Gradient-boosted trees (scikit-learn's `HistGradientBoostingClassifier`, and the XGBoost, LightGBM and CatBoost libraries) are the default choice for tabular problems: fraud, churn, pricing, ranking features. On tables of numbers and categories they usually match or beat neural networks with far less tuning.

### 10.3 Comparing them in code

Five-fold cross-validated accuracy (§16) on a synthetic tabular problem with 20 features, 8 of them informative:

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=3000, n_features=20, n_informative=8,
                           n_redundant=4, flip_y=0.03, random_state=0)
models = {
    "logistic regression": LogisticRegression(max_iter=1000),
    "tree, max_depth=5": DecisionTreeClassifier(max_depth=5, random_state=0),
    "tree, unlimited": DecisionTreeClassifier(random_state=0),
    "random forest, 100 trees": RandomForestClassifier(n_estimators=100, random_state=0),
    "gradient boosting": HistGradientBoostingClassifier(random_state=0),
}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name:25s} accuracy {scores.mean():.3f} +/- {scores.std():.3f}")
```

```text
logistic regression       accuracy 0.922 +/- 0.011
tree, max_depth=5         accuracy 0.879 +/- 0.011
tree, unlimited           accuracy 0.873 +/- 0.005
random forest, 100 trees  accuracy 0.948 +/- 0.011
gradient boosting         accuracy 0.949 +/- 0.013
```

A single tree is the weakest model here, deep or shallow. Averaging (forest) or boosting fixes that. The linear model is respectable because part of the signal is linear; on your data, run the same table before choosing.

---

## 11. k-nearest neighbours

### 11.1 By hand on a tiny 2-D set

**k-NN** stores the training set and classifies a new point by a vote among its \(k\) closest training points. There is no training step at all; the cost moves to prediction time. Distance is usually Euclidean.

\[
d(\mathbf p,\mathbf q)=\sqrt{\textstyle\sum_i(p_i-q_i)^2}
\]

!!! example "On paper"

    Six tickets placed on two features, labelled billing (B) or technical (T): P1 (1,2) B, P2 (3,1) T, P3 (4,5) T, P4 (1,4) B, P5 (2,2) T, P6 (5,1) B. The new ticket is at q = (1,1):

    \[
    \begin{aligned}
    d(\text{P1}) &= 1\ \text{(B)}\\
    d(\text{P5}) &= \sqrt2 \approx 1.414\ \text{(T)}\\
    d(\text{P2}) &= 2\ \text{(T)}\\
    d(\text{P4}) &= 3\ \text{(B)}\\
    d(\text{P6}) &= 4\ \text{(B)}\\
    d(\text{P3}) &= \sqrt{9+16} = 5\ \text{(T)}
    \end{aligned}
    \]

    k = 1: billing (P1). k = 3: technical, 2 votes to 1. k = 5: billing, 3 votes to 2. The answer depends on k, so k is a hyperparameter to choose on validation data (an odd k avoids ties between two classes).

### 11.2 Why scaling matters

Distances add up the features' units. If one feature is account age in days (0 to 3,000) and another is the number of open tickets (0 to 5), the days dominate every distance and the other feature is ignored. Standardise features first (§21.1). The same six tickets, with the second feature accidentally expressed in thousandths:

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = np.array([[1, 2], [3, 1], [4, 5], [1, 4], [2, 2], [5, 1]], dtype=float)
y = np.array(["billing", "technical", "technical", "billing", "technical", "billing"])
q = np.array([[1.0, 1.0]])
print("distances:", np.round(np.linalg.norm(X - q, axis=1), 3))
for k in (1, 3, 5):
    print(f"k={k}:", KNeighborsClassifier(n_neighbors=k).fit(X, y).predict(q)[0])

X_units, q_units = X * [1, 1000], q * [1, 1000]          # second feature now in thousandths
print("k=1, raw units  :", KNeighborsClassifier(n_neighbors=1).fit(X_units, y).predict(q_units)[0])
scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)).fit(X_units, y)
print("k=1, standardised:", scaled.predict(q_units)[0])
```

```text
distances: [1.    2.    5.    3.    1.414 4.   ]
k=1: billing
k=3: technical
k=5: billing
k=1, raw units  : technical
k=1, standardised: billing
```

In raw units the nearest neighbour becomes P2, because only the second feature counts now. Standardising restores the original answer. You will meet k-NN again as semantic search: "find the 5 most similar past tickets and copy their label" is k-NN on embeddings with cosine distance ([51](51-ai-rag.md)).

---

## 12. k-means clustering

### 12.1 The algorithm, and two iterations by hand

**k-means** splits unlabelled points into \(k\) groups. Start with \(k\) centre points (**centroids**), then repeat two steps until nothing changes: assign every point to its nearest centroid; move every centroid to the mean of its points. It minimises the **inertia**, the total squared distance from points to their centroids.

\[
\text{inertia}=\sum_i\lVert\mathbf x_i-\boldsymbol\mu_{c(i)}\rVert^2
\]

!!! example "On paper"

    Six user questions as 2-D points (say, two coordinates of their embeddings): (1,1), (1,2), (2,1), (8,8), (8,9), (9,8). Start badly, with centroids \(c_1=(1,1)\) and \(c_2=(2,1)\). Squared distances for the points that are not on a centroid:

    \[
    \begin{aligned}
    (1,2) &: c_1\ 1,\ c_2\ 2 \to c_1\\
    (8,8) &: c_1\ 98,\ c_2\ 85 \to c_2\\
    (8,9),(9,8) &: \text{closer to } c_2\\
    c_1 &= \text{mean}\{(1,1),(1,2)\} = (1,1.5)\\
    c_2 &= \text{mean}\{(2,1),(8,8),(8,9),(9,8)\} = (6.75,6.5)
    \end{aligned}
    \]

    After iteration 1 the inertia is 72.25: 0.5 in cluster 1, 71.75 in cluster 2, mostly from (2,1) sitting far from \(c_2\).

!!! example "On paper"

    Iteration 2. The point (2,1) is now much closer to \(c_1\):

    \[
    \begin{aligned}
    (2,1) &: c_1\ 1.25,\ c_2\ 52.8 \to c_1\\
    c_1 &= \text{mean}\{(1,1),(1,2),(2,1)\} = (4/3,4/3)\\
    c_2 &= \text{mean}\{(8,8),(8,9),(9,8)\} = (25/3,25/3)\\
    \text{inertia} &= 12/9+12/9 \approx 2.667
    \end{aligned}
    \]

    A third iteration changes no assignment, so the algorithm has converged: two clean groups, and inertia fell from 72.25 to 2.667. The bad start cost one extra iteration here; on real data a bad start can converge to a bad answer, which is why libraries run several starts.

### 12.2 In code, and choosing k

```python
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1, 1], [1, 2], [2, 1], [8, 8], [8, 9], [9, 8]], dtype=float)
C = np.array([[1.0, 1.0], [2.0, 1.0]])                  # the bad start from the cards

for it in (1, 2, 3):
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)   # (6 points, 2 centroids)
    assign = d2.argmin(axis=1)
    C = np.array([X[assign == j].mean(axis=0) for j in range(2)])
    inertia = ((X - C[assign]) ** 2).sum()
    print(f"iteration {it}: assign={assign}  centroids={np.round(C, 3).tolist()}  inertia={inertia:.3f}")

for k in (1, 2, 3):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    print(f"sklearn k={k}: inertia={km.inertia_:.3f}")
```

```text
iteration 1: assign=[0 0 1 1 1 1]  centroids=[[1.0, 1.5], [6.75, 6.5]]  inertia=72.250
iteration 2: assign=[0 0 0 1 1 1]  centroids=[[1.333, 1.333], [8.333, 8.333]]  inertia=2.667
iteration 3: assign=[0 0 0 1 1 1]  centroids=[[1.333, 1.333], [8.333, 8.333]]  inertia=2.667
sklearn k=1: inertia=149.667
sklearn k=2: inertia=2.667
sklearn k=3: inertia=1.833
```

Inertia always falls as \(k\) grows (with \(k=6\) it is 0), so you cannot choose \(k\) by minimising it. Look for the "elbow" where adding a cluster stops helping (here from 1 to 2 the drop is huge, from 2 to 3 small), use the silhouette score, and above all read samples from each cluster: clustering is an exploration tool, and its output needs a human to name the clusters. Pitfalls: k-means assumes round clusters of similar size, and it needs scaled features or normalised embeddings.

---

## 13. Naive Bayes for text

### 13.1 The idea

**Naive Bayes** applies Bayes' rule ([46 §8.4](46-ai-math.md)) with one "naive" assumption: given the class, words occur independently. The assumption is false (words come in phrases), yet the classifier works well for short texts, trains in one pass over the data and is a strong baseline for spam and ticket routing.

\[
P(c\mid w_1,\dots,w_n)\propto P(c)\prod_{i=1}^{n}P(w_i\mid c)
\]

!!! example "On paper"

    \(P(w\mid c)\) is estimated by counting with **Laplace smoothing**: add 1 to every count so unseen words do not get probability 0. With a vocabulary of \(V=7\) words and 5 word occurrences in the class:

    \[
    \begin{aligned}
    P(w\mid c) &= \frac{\text{count}(w,c)+1}{\text{words in }c+V}\\
    \text{count }2 &: 3/12 = 0.25\\
    \text{count }0 &: 1/12 \approx 0.083
    \end{aligned}
    \]

    Without the +1, a single word never seen in spam would make any message containing it impossible to classify as spam.

### 13.2 A tiny worked example

!!! example "On paper"

    Training inbox: spam "free refund now", spam "free money", ham "refund my order", ham "order status". Vocabulary: free, refund, now, money, my, order, status (7 words). Each class has 5 word occurrences. Classify "free refund", priors 0.5 each:

    \[
    \begin{aligned}
    \text{spam} &: 0.5\cdot\tfrac{3}{12}\cdot\tfrac{2}{12} = 0.5\cdot\tfrac{6}{144}\\
    \text{ham} &: 0.5\cdot\tfrac{1}{12}\cdot\tfrac{2}{12} = 0.5\cdot\tfrac{2}{144}\\
    P(\text{spam}\mid\text{msg}) &= 6/(6+2) = 0.75
    \end{aligned}
    \]

    "refund" appears once in each class, so it cancels out; "free" (twice in spam, never in ham) decides. Real implementations add log-probabilities instead of multiplying ([46 §7.2](46-ai-math.md)).

### 13.3 In code

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

texts = ["free refund now", "free money", "refund my order", "order status"]
labels = ["spam", "spam", "ham", "ham"]
model = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0)).fit(texts, labels)

print("vocabulary:", sorted(model[0].vocabulary_))
for msg in ["free refund", "refund my order please", "free order"]:
    proba = model.predict_proba([msg])[0]
    print(f"{msg!r:26s} P(ham)={proba[0]:.3f} P(spam)={proba[1]:.3f}")
```

```text
vocabulary: ['free', 'money', 'my', 'now', 'order', 'refund', 'status']
'free refund'              P(ham)=0.250 P(spam)=0.750
'refund my order please'   P(ham)=0.857 P(spam)=0.143
'free order'               P(ham)=0.500 P(spam)=0.500
```

"please" is not in the vocabulary, so it is ignored. "free order" is an exact tie: "free" points to spam (3/12 against 1/12) exactly as strongly as "order" points to ham. `alpha=1.0` is the Laplace +1. Naive Bayes probabilities tend to be overconfident (§18.9): use its ranking, and calibrate before you use its probabilities as numbers.

---

## 14. Overfitting, underfitting and bias–variance

### 14.1 Reading training and validation scores

**Underfitting**: the model is too simple to capture the pattern; it is bad on training and validation data alike. **Overfitting**: the model fits the training data, noise included, and does much worse on new data. You diagnose them by comparing the two scores:

| Training error | Validation error | Diagnosis | What to try |
|---|---|---|---|
| high | high (similar) | underfitting (high bias) | more features, a more flexible model, less regularization |
| low | low | good fit | ship it, keep monitoring |
| very low | high | overfitting (high variance) | more data, regularization, a simpler model, early stopping |
| low | high, and rising over time | drift or leakage | time-based validation (§4.4), check §5 |

The same table reads a fine-tuning run: training loss falling while validation loss rises is overfitting ([53](53-ai-fine-tuning.md)).

### 14.2 Polynomial degree in code

Fit polynomials of growing degree to 12 noisy points from a sine curve (noise sd 0.3, so the best possible MSE is \(0.3^2=0.09\)):

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(1)

def sample(n):                                   # true curve + noise with sd 0.3
    x = rng.uniform(-1, 1, n)
    return x.reshape(-1, 1), np.sin(np.pi * x) + rng.normal(0, 0.3, n)

X_train, y_train = sample(12)
X_val, y_val = sample(500)
for degree in (0, 1, 3, 5, 8, 11):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression()).fit(X_train, y_train)
    tr = mean_squared_error(y_train, model.predict(X_train))
    va = mean_squared_error(y_val, model.predict(X_val))
    print(f"degree {degree:2d}: train MSE {tr:6.3f}   validation MSE {va:9.3f}")
```

```text
degree  0: train MSE  0.460   validation MSE     0.605
degree  1: train MSE  0.224   validation MSE     0.281
degree  3: train MSE  0.037   validation MSE     0.113
degree  5: train MSE  0.028   validation MSE     0.114
degree  8: train MSE  0.008   validation MSE     2.452
degree 11: train MSE  0.008   validation MSE    62.576
```

Training error only goes down as the degree grows. Validation error falls, bottoms out at degree 3 to 5, close to the 0.09 noise floor, then explodes: the degree-11 curve passes near every training point and swings wildly between them. Choose the degree (in general, the model capacity) on validation error, never on training error.

### 14.3 The bias–variance decomposition

The expected squared error of a model at a point splits into three parts: **bias** (how far the average prediction is from the truth), **variance** (how much the prediction changes with the training sample) and **noise** (which no model can remove).

\[
\mathbb E\bigl[(y-\hat y)^2\bigr]=\text{bias}^2+\text{variance}+\sigma^2_{\text{noise}}
\]

!!! example "On paper"

    The true value is 10 and the noise variance 1. Trained on three different samples, model A predicts 11, 12, 13; model B predicts 6, 10, 14:

    \[
    \begin{aligned}
    \text{A} &: \text{mean } 12,\ \text{bias}^2 = 4,\ \text{var} = 2/3\\
    \text{A total} &= 4+0.667+1 \approx 5.667\\
    \text{B} &: \text{mean } 10,\ \text{bias}^2 = 0,\ \text{var} = 32/3\\
    \text{B total} &= 0+10.667+1 \approx 11.667
    \end{aligned}
    \]

    B is right on average, A is not, yet A has half the error. Accepting some bias to cut variance is what regularization (§15) and simple models buy you.

The same experiment in code: train each polynomial on 200 different samples of 12 points and look at its prediction at \(x=0.5\), where the true value is \(\sin(\pi/2)=1\):

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(2)
x0 = np.array([[0.5]])                     # the true value here is sin(pi/2) = 1
for degree in (1, 3, 8):
    preds = []
    for _ in range(200):                   # 200 different training sets of 12 points
        x = rng.uniform(-1, 1, 12)
        y = np.sin(np.pi * x) + rng.normal(0, 0.3, 12)
        m = make_pipeline(PolynomialFeatures(degree), LinearRegression()).fit(x.reshape(-1, 1), y)
        preds.append(m.predict(x0)[0])
    preds = np.array(preds)
    print(f"degree {degree}: bias^2={(preds.mean() - 1) ** 2:6.3f}  variance={preds.var():8.3f}")
```

```text
degree 1: bias^2= 0.238  variance=   0.055
degree 3: bias^2= 0.000  variance=   0.033
degree 8: bias^2= 1.761  variance= 343.190
```

The straight line is biased (it cannot bend to reach 1) but stable. Degree 3 is nearly unbiased with low variance. Degree 8's predictions jump around from sample to sample, and a few wild fits even drag its average away from the truth.

### 14.4 What to do about each

- **High bias**: add features, use a more flexible model (trees instead of a line), reduce regularization, train longer.
- **High variance**: get more data (the strongest fix), regularize, simplify, average models (forests), stop training earlier.
- **Learning curves** tell you which: plot validation error against training-set size. If it is still falling, more data will help; if it is flat and far above training error, change the model.

---

## 15. Regularization

### 15.1 L2 and L1 penalties

**Regularization** adds a penalty on the size of the weights to the loss, so the optimizer only uses a large weight when the data really pays for it. **L2** (ridge) penalises the squared weights and shrinks them all smoothly. **L1** (lasso) penalises absolute values and pushes weak weights to exactly zero, which also selects features.

\[
L_{\text{ridge}}=\sum_i(y_i-\mathbf w\cdot\mathbf x_i)^2+\lambda\lVert\mathbf w\rVert_2^2,
\quad
L_{\text{lasso}}=\dots+\lambda\lVert\mathbf w\rVert_1
\]

!!! example "On paper"

    One feature, no intercept, \(x=(1,2,3)\), \(y=(2,3,7)\). Setting the derivative of each loss to zero gives closed forms, with \(\sum xy=2+6+21=29\) and \(\sum x^2=14\):

    \[
    \begin{aligned}
    \text{ridge} &: w = \textstyle\sum xy/(\sum x^2+\lambda)\\
    \lambda=0 &: 29/14 \approx 2.071\\
    \lambda=14 &: 29/28 \approx 1.036\\
    \text{lasso} &: w = (\textstyle\sum xy-\lambda/2)/\sum x^2\\
    \lambda=14 &: 22/14 \approx 1.571\\
    \lambda\ge58 &: w = 0
    \end{aligned}
    \]

    Ridge never reaches zero; it halves the weight when \(\lambda\) equals \(\sum x^2\). Lasso subtracts a fixed amount and stops at exactly 0 once \(\lambda/2\) exceeds the feature's total signal. (scikit-learn's `Ridge(alpha=14, fit_intercept=False)` returns exactly 1.036 here; its `Lasso` divides the squared error by \(2n\), so a `Lasso` `alpha` is not this \(\lambda\).)

Weight decay in neural network training is the L2 idea ([48](48-ai-deep-learning.md)).

### 15.2 The effect on weights in code

Eight features, of which only the first three matter:

```python
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge

rng = np.random.default_rng(0)
X = rng.normal(size=(60, 8))
true_w = np.array([3.0, -2.0, 1.5, 0, 0, 0, 0, 0])
y = X @ true_w + rng.normal(0, 1.0, 60)

def show(name, w):
    print(f"{name:12s}", " ".join(f"{0.0 if abs(v) < 0.005 else v:5.2f}" for v in w))

show("true", true_w)
show("no penalty", LinearRegression().fit(X, y).coef_)
for alpha in (10, 100):
    show(f"ridge {alpha}", Ridge(alpha=alpha).fit(X, y).coef_)
for alpha in (0.1, 0.5):
    show(f"lasso {alpha}", Lasso(alpha=alpha).fit(X, y).coef_)
```

```text
true          3.00 -2.00  1.50  0.00  0.00  0.00  0.00  0.00
no penalty    3.00 -2.22  1.52  0.19  0.19 -0.03  0.15 -0.16
ridge 10      2.39 -1.80  1.22  0.03  0.22  0.01  0.14 -0.02
ridge 100     0.87 -0.69  0.48 -0.15  0.17  0.07  0.10  0.12
lasso 0.1     2.82 -2.05  1.34  0.02  0.13  0.00  0.07  0.00
lasso 0.5     2.26 -1.57  0.91  0.00  0.00  0.00  0.00  0.00
```

Unpenalised, the five useless features get small non-zero weights fitted to noise. Ridge shrinks the weight vector as a whole (its length falls from about 4.0 to 3.2 and 1.2), the useful weights most visibly, but the useless weights stay non-zero and some even grow or flip sign, because in 60 rows the features are slightly correlated by chance. Lasso at 0.5 sets all five useless weights to exactly zero while keeping the three real ones, at the price of shrinking them too. The strength (`alpha`, or `C` for logistic regression) is a hyperparameter: choose it by cross-validation (§16–§17).

---

## 16. Cross-validation

### 16.1 k-fold by hand

A single validation split wastes data and gives a noisy score. **k-fold cross-validation** splits the data into \(k\) equal folds, trains \(k\) times, each time validating on a different fold and training on the rest, and averages the \(k\) scores. Every example is used for validation exactly once.

\[
\text{CV score}=\frac1k\sum_{j=1}^{k}\text{score}_j
\]

!!! example "On paper"

    1,000 labelled tickets, 5 folds of 200. Each training run uses 800 tickets. The five validation accuracies are 0.82, 0.85, 0.80, 0.84 and 0.84:

    \[
    \begin{aligned}
    \text{mean} &= 4.15/5 = 0.83\\
    \text{deviations} &= (-0.01,0.02,-0.03,0.01,0.01)\\
    s^2 &= 0.0016/4 = 0.0004\\
    s &= 0.02
    \end{aligned}
    \]

    Report "0.83 ± 0.02 across folds". The fold-to-fold spread shows how much one split could have misled you: 0.80 or 0.85 from a single split, both plausible.

The cost is \(k\) trainings instead of one. With \(k=5\) or 10 that is fine for classical models and out of the question for fine-tuning a 14B model, where a single held-out validation set is the norm.

### 16.2 Variants: stratified, group and time series

Cross-validation must respect the same rules as a single split (§4): stratify rare classes, keep groups together, and never validate on the past with a model trained on the future.

```python
import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

X = np.arange(10).reshape(-1, 1)                        # 10 examples, in time order
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"])   # e.g. customers

for name, cv, kw in [("KFold(5)", KFold(5), {}),
                     ("StratifiedKFold(2)", StratifiedKFold(2), {}),
                     ("GroupKFold(5)", GroupKFold(5), {"groups": groups}),
                     ("TimeSeriesSplit(3)", TimeSeriesSplit(3), {})]:
    print(name)
    for train, val in cv.split(X, y, **kw):
        print(f"   train={train.tolist()}  val={val.tolist()}")
```

```text
KFold(5)
   train=[2, 3, 4, 5, 6, 7, 8, 9]  val=[0, 1]
   train=[0, 1, 4, 5, 6, 7, 8, 9]  val=[2, 3]
   train=[0, 1, 2, 3, 6, 7, 8, 9]  val=[4, 5]
   train=[0, 1, 2, 3, 4, 5, 8, 9]  val=[6, 7]
   train=[0, 1, 2, 3, 4, 5, 6, 7]  val=[8, 9]
StratifiedKFold(2)
   train=[4, 5, 6, 7, 9]  val=[0, 1, 2, 3, 8]
   train=[0, 1, 2, 3, 8]  val=[4, 5, 6, 7, 9]
GroupKFold(5)
   train=[0, 1, 2, 3, 4, 5, 6, 7]  val=[8, 9]
   train=[0, 1, 2, 3, 4, 5, 8, 9]  val=[6, 7]
   train=[0, 1, 2, 3, 6, 7, 8, 9]  val=[4, 5]
   train=[0, 1, 4, 5, 6, 7, 8, 9]  val=[2, 3]
   train=[2, 3, 4, 5, 6, 7, 8, 9]  val=[0, 1]
TimeSeriesSplit(3)
   train=[0, 1, 2, 3]  val=[4, 5]
   train=[0, 1, 2, 3, 4, 5]  val=[6, 7]
   train=[0, 1, 2, 3, 4, 5, 6, 7]  val=[8, 9]
```

`StratifiedKFold` puts one of the two positives in each validation fold. `GroupKFold` keeps each customer's pair together. `TimeSeriesSplit` always trains on the past and validates on the next block. `cross_val_score(model, X, y, cv=...)` accepts any of these.

---

## 17. Hyperparameter search

### 17.1 Grid and random search

**Hyperparameters** are settings you choose rather than learn: a tree's depth, a forest's number of trees, the regularization strength, k in k-NN, the learning rate. **Grid search** tries every combination; **random search** samples a fixed number of combinations and usually finds as good a setting with far fewer trials, because only a few hyperparameters matter much.

\[
\text{fits}=\text{combinations}\cdot k_{\text{folds}}+1
\]

!!! example "On paper"

    A grid of 4 depths × 3 leaf sizes × 5 learning rates with 5-fold cross-validation:

    \[
    \begin{aligned}
    4\cdot3\cdot5 &= 60 \text{ combinations}\\
    60\cdot5+1 &= 301 \text{ fits}\\
    \text{random, 20 samples} &: 20\cdot5+1 = 101
    \end{aligned}
    \]

    The +1 is the final refit of the best setting on all the training data. Each extra hyperparameter multiplies a grid; random search keeps the budget fixed.

### 17.2 In code

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=2000, n_features=20, n_informative=8, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

grid = {"max_depth": [2, 4, 8, None], "min_samples_leaf": [1, 10, 50]}
search = GridSearchCV(DecisionTreeClassifier(random_state=0), grid, cv=5).fit(X_train, y_train)

print("combinations tried:", len(search.cv_results_["params"]))
print("best params       :", search.best_params_)
print(f"best CV accuracy  : {search.best_score_:.3f}")
print(f"test accuracy     : {search.score(X_test, y_test):.3f}   (touched once, at the end)")
```

```text
combinations tried: 12
best params       : {'max_depth': 4, 'min_samples_leaf': 1}
best CV accuracy  : 0.869
test accuracy     : 0.870   (touched once, at the end)
```

`RandomizedSearchCV` has the same interface with `n_iter` samples from distributions. Optuna and similar libraries do smarter sequential search; the principle does not change.

### 17.3 Keep the test set out of it

Search on the training data with cross-validation, then evaluate the single chosen model on the test set once (§4.2). If you need an unbiased estimate of the whole procedure, search included, use **nested cross-validation**: an outer loop of folds, each running its own inner search.

---

## 18. Classification metrics

### 18.1 The confusion matrix

Every binary metric is computed from four counts. Take 1,000 tickets, of which 100 are urgent, and a classifier that flags 120:

| | predicted urgent | predicted not urgent | total |
|---|---:|---:|---:|
| **actually urgent** | TP = 80 | FN = 20 | 100 |
| **actually not urgent** | FP = 40 | TN = 860 | 900 |
| **total** | 120 | 880 | 1,000 |

TP: true positives (caught), FN: false negatives (missed), FP: false positives (false alarms), TN: true negatives. Decide which error hurts more before choosing a metric: a missed outage (FN) or an engineer paged for nothing (FP)?

### 18.2 Accuracy, and why it lies

\[
\text{accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
\]

!!! example "On paper"

    The matrix above against the majority baseline of §6.1:

    \[
    \begin{aligned}
    \text{model} &: (80+860)/1000 = 0.94\\
    \text{"never urgent"} &: 900/1000 = 0.90
    \end{aligned}
    \]

    Four points of accuracy separate a useful model from one that misses every urgent ticket. With 1% positives, "never" scores 0.99. On imbalanced data, accuracy hides exactly the errors you care about.

### 18.3 Precision and recall

**Precision**: of the tickets flagged, how many were urgent? **Recall** (sensitivity, true positive rate): of the urgent tickets, how many were flagged?

\[
\text{precision}=\frac{TP}{TP+FP},
\quad
\text{recall}=\frac{TP}{TP+FN}
\]

!!! example "On paper"

    From the confusion matrix:

    \[
    \begin{aligned}
    \text{precision} &= 80/(80+40) = 80/120 \approx 0.667\\
    \text{recall} &= 80/(80+20) = 0.8\\
    \text{false positive rate} &= 40/900 \approx 0.044
    \end{aligned}
    \]

    Two in three alerts are real, and four in five urgent tickets are caught. Precision answers "can I trust an alert?"; recall answers "what am I missing?".

### 18.4 F1 and F-beta

**F1** is the harmonic mean of precision and recall: high only when both are. **F-beta** weights recall \(\beta\) times as much as precision: F2 when misses are costly, F0.5 when false alarms are.

\[
F_\beta=\frac{(1+\beta^2)\,P\,R}{\beta^2P+R}
\]

!!! example "On paper"

    With \(P=2/3\) and \(R=0.8\):

    \[
    \begin{aligned}
    F_1 &= 2PR/(P+R) = 1.0667/1.4667 \approx 0.727\\
    F_2 &= 5PR/(4P+R) = 2.667/3.467 \approx 0.769\\
    F_{0.5} &= 1.25PR/(0.25P+R) = 0.667/0.967 \approx 0.690
    \end{aligned}
    \]

    F1 also equals \(2TP/(2TP+FP+FN)=160/220\). F2 is higher because this model's recall is its stronger side. Pick the beta from the cost of each error, not from habit.

### 18.5 Thresholds and the precision–recall trade-off

A classifier outputs scores; the threshold turns them into decisions. Lowering it flags more items: recall rises, precision usually falls. Six tickets scored by a model, sorted by score:

| score | 0.9 | 0.8 | 0.7 | 0.6 | 0.4 | 0.2 |
|---|---|---|---|---|---|---|
| truly urgent? | yes | yes | no | yes | no | no |

| threshold | flagged | TP | FP | precision | recall |
|---:|---:|---:|---:|---:|---:|
| 0.85 | 1 | 1 | 0 | 1.00 | 0.33 |
| 0.75 | 2 | 2 | 0 | 1.00 | 0.67 |
| 0.65 | 3 | 2 | 1 | 0.67 | 0.67 |
| 0.50 | 4 | 3 | 1 | 0.75 | 1.00 |
| 0.30 | 5 | 3 | 2 | 0.60 | 1.00 |
| 0.10 | 6 | 3 | 3 | 0.50 | 1.00 |

Precision is not monotonic (it went up from 0.67 to 0.75 at 0.50). In production you choose the threshold for a business constraint: "highest recall with precision at least 0.75" picks 0.50 here. Choose it on validation data, never on the test set.

### 18.6 ROC curve and AUC by hand

The **ROC curve** plots the true positive rate (recall) against the false positive rate for every threshold. The area under it, **ROC-AUC**, has a plain meaning: the probability that a randomly chosen positive gets a higher score than a randomly chosen negative. 0.5 is random, 1.0 is a perfect ranking.

\[
\text{AUC}=\frac{\#\{(i,j): s_i>s_j\}+\tfrac12\#\{s_i=s_j\}}{n_+\,n_-},
\quad i\in+,\ j\in-
\]

!!! example "On paper"

    The six tickets above: positives score 0.9, 0.8, 0.6; negatives 0.7, 0.4, 0.2. Compare every positive with every negative, \(3\cdot3=9\) pairs:

    \[
    \begin{aligned}
    0.9 &: \text{beats } 0.7, 0.4, 0.2 \to 3\\
    0.8 &: \text{beats } 0.7, 0.4, 0.2 \to 3\\
    0.6 &: \text{beats } 0.4, 0.2 \to 2\\
    \text{AUC} &= 8/9 \approx 0.889
    \end{aligned}
    \]

    One pair is out of order (the negative at 0.7 above the positive at 0.6). AUC measures ranking only: dividing every score by 10 leaves it unchanged, so a model can have a fine AUC and useless probabilities.

The curve's corners for this example are (FPR, TPR) = (0, 0), (0, 1/3), (0, 2/3), (1/3, 2/3), (1/3, 1), (2/3, 1), (1, 1); the area under that staircase is \(\tfrac13\cdot\tfrac23+\tfrac13\cdot1+\tfrac13\cdot1=\tfrac89\), the same number.

### 18.7 PR-AUC and average precision

With rare positives, ROC-AUC can look excellent while precision is poor, because the false positive rate divides by the huge number of negatives. The **precision–recall curve** and its summary, **average precision** (AP), focus on the positives: AP averages the precision at the rank of each positive.

\[
\text{AP}=\sum_n(R_n-R_{n-1})\,P_n
\]

!!! example "On paper"

    Same six tickets. The positives sit at ranks 1, 2 and 4; each raises recall by 1/3:

    \[
    \begin{aligned}
    P@1 &= 1/1 = 1\\
    P@2 &= 2/2 = 1\\
    P@4 &= 3/4 = 0.75\\
    \text{AP} &= \tfrac13(1+1+0.75) \approx 0.917
    \end{aligned}
    \]

    On a large set, a random ranking has an AP close to the positive rate: 0.5 for a balanced set, 0.01 for a 1% class. (On tiny sets it is higher: for these six tickets a random order scores 0.645 on average.) Always compare AP to that baseline. For fraud or urgent-ticket detection, report AP or "recall at precision ≥ X" rather than ROC-AUC.

### 18.8 Metrics in code

```python
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score, roc_auc_score)

# the confusion matrix of 18.1, as 1,000 labels and predictions
y_true = np.array([1] * 80 + [1] * 20 + [0] * 40 + [0] * 860)
y_pred = np.array([1] * 80 + [0] * 20 + [1] * 40 + [0] * 860)
print(confusion_matrix(y_true, y_pred, labels=[1, 0]))
print(f"precision={precision_score(y_true, y_pred):.3f} recall={recall_score(y_true, y_pred):.3f} "
      f"F1={f1_score(y_true, y_pred):.3f} F2={fbeta_score(y_true, y_pred, beta=2):.3f}")

# the six scored tickets of 18.5-18.7
y6 = np.array([1, 1, 0, 1, 0, 0])
s6 = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.2])
pairs = [(p > n) + 0.5 * (p == n) for p in s6[y6 == 1] for n in s6[y6 == 0]]
print("AUC by pairs:", round(sum(pairs) / len(pairs), 4), " sklearn:", round(roc_auc_score(y6, s6), 4))
print("AP:", round(average_precision_score(y6, s6), 4), " AUC of scores/10:", round(roc_auc_score(y6, s6 / 10), 4))
```

```text
[[ 80  20]
 [ 40 860]]
precision=0.667 recall=0.800 F1=0.727 F2=0.769
AUC by pairs: 0.8889  sklearn: 0.8889
AP: 0.9167  AUC of scores/10: 0.8889
```

`confusion_matrix(..., labels=[1, 0])` puts the positive class first, matching the table in §18.1; by default scikit-learn sorts labels, so the first row is class 0. Check the order every time you read one.

### 18.9 Calibration and the Brier score

A model is **calibrated** if, among all items it scores 0.8, about 80% are positive. Calibration matters whenever probabilities are used as numbers: cost thresholds (§8.5), expected-value calculations, "confidence" shown to users, routing to a human. The **Brier score** is the mean squared error of the probabilities: lower is better.

\[
\text{Brier}=\frac1n\sum_i(p_i-y_i)^2
\]

!!! example "On paper"

    Four predictions as (probability, label): (0.9, 1), (0.8, 1), (0.3, 0), (0.6, 0):

    \[
    \begin{aligned}
    \text{errors}^2 &= 0.01,\ 0.04,\ 0.09,\ 0.36\\
    \text{Brier} &= 0.5/4 = 0.125
    \end{aligned}
    \]

    Always answering 0.5 scores 0.25; a perfect, confident model scores 0. The 0.6 on a negative contributes most. Brier rewards both ranking and honest probabilities, unlike AUC.

A **reliability diagram** checks calibration directly: bin the predictions and compare each bin's average prediction with its observed positive rate. Naive Bayes against logistic regression on the same data:

```python
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=20_000, n_features=20, n_informative=5,
                           n_redundant=10, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
for name, model in [("naive Bayes", GaussianNB()), ("logistic regression", LogisticRegression())]:
    p = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    observed, predicted = calibration_curve(y_test, p, n_bins=5)
    print(f"{name}: Brier={brier_score_loss(y_test, p):.3f}")
    for pr, ob in zip(predicted, observed):
        print(f"   predicted {pr:.2f} -> observed {ob:.2f}")
```

```text
naive Bayes: Brier=0.070
   predicted 0.01 -> observed 0.08
   predicted 0.30 -> observed 0.53
   predicted 0.50 -> observed 0.57
   predicted 0.70 -> observed 0.73
   predicted 0.99 -> observed 0.96
logistic regression: Brier=0.050
   predicted 0.03 -> observed 0.02
   predicted 0.29 -> observed 0.24
   predicted 0.50 -> observed 0.49
   predicted 0.71 -> observed 0.73
   predicted 0.96 -> observed 0.96
```

Naive Bayes pushes predictions towards 0 and 1 (its independence assumption counts correlated evidence twice, and this dataset has 10 redundant features): its lowest bin claims 1% and is really about 8%. Logistic regression, trained on log-loss, is close to the diagonal. To fix a miscalibrated model, wrap it in `CalibratedClassifierCV` (Platt scaling or isotonic regression), fitted on data the model was not trained on. LLM-reported confidence ("I am 90% sure") is often poorly calibrated too; measure it the same way ([54](54-ai-evaluation.md), [43 §20](43-math-examples.md#expected-calibration-error)).

### 18.10 Multiclass averages: macro, micro and weighted

With several classes, compute precision, recall and F1 per class, then average. **Macro** averages the classes equally, so a rare class counts as much as a common one. **Weighted** averages by class size. **Micro** pools all decisions; for single-label multiclass it equals accuracy. If the rare class matters (escalations, legal requests), report macro-F1 and the per-class table, as `classification_report` does (§22.3).

---

## 19. Regression metrics

**MAE** (mean absolute error) is the average size of the error, in the target's units. **MSE** squares the errors, so it punishes large ones; **RMSE** is its square root, back in the target's units. **R²** is the share of the variance explained: 1 is perfect, 0 is no better than always predicting the mean, and it can be negative.

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}
\]

!!! example "On paper"

    Four tickets' handling times in hours, \(y=(2,4,6,8)\), predicted as \(\hat y=(3,4,4,9)\); errors \((1,0,-2,1)\), \(\bar y=5\):

    \[
    \begin{aligned}
    \text{MAE} &= (1+0+2+1)/4 = 1.0\\
    \text{MSE} &= (1+0+4+1)/4 = 1.5\\
    \text{RMSE} &= \sqrt{1.5} \approx 1.225\\
    R^2 &= 1-6/20 = 0.7
    \end{aligned}
    \]

    Typical error: one hour. RMSE is above MAE because of the single 2-hour miss. Report MAE to stakeholders ("off by an hour on average") and watch RMSE for large misses.

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

y = np.array([2.0, 4.0, 6.0, 8.0])
y_hat = np.array([3.0, 4.0, 4.0, 9.0])
print("MAE =", mean_absolute_error(y, y_hat), " MSE =", mean_squared_error(y, y_hat))
print("RMSE =", round(root_mean_squared_error(y, y_hat), 4), " R2 =", round(r2_score(y, y_hat), 4))
print("R2 of always predicting the mean:", r2_score(y, np.full(4, y.mean())))
```

```text
MAE = 1.0  MSE = 1.5
RMSE = 1.2247  R2 = 0.7
R2 of always predicting the mean: 0.0
```

`root_mean_squared_error` replaced the old `mean_squared_error(..., squared=False)`, which recent scikit-learn versions no longer accept.

---

## 20. Class imbalance

### 20.1 Three fixes

When one class is rare (1% fraud, 5% urgent tickets), a model trained to minimise average loss leans towards the majority and rarely predicts the minority at a 0.5 threshold. Three fixes, in the order to try them:

1. **Move the threshold** (§8.5). The model's ranking is often fine; only the 0.5 cut is wrong. Cheapest and fully reversible.
2. **Class weights.** Make each minority example count more in the loss. One argument in most scikit-learn models. Like resampling, it inflates the predicted probabilities of the minority class, so do not read them as real-world rates without the correction of §20.3.
3. **Resampling.** Oversample the minority (duplicate it, or synthesise new rows with SMOTE from the imbalanced-learn library) or undersample the majority. Useful with very large data or models without weight support; it changes the class prior, so fix the probabilities afterwards (§20.3).

And always evaluate with precision, recall and AP on data with the **real** class balance; never resample the test set.

### 20.2 Class weights

scikit-learn's `class_weight="balanced"` gives each class a weight inversely proportional to its frequency, so both classes contribute equally to the loss.

\[
w_c=\frac{n}{k\,n_c}
\]

!!! example "On paper"

    1,000 tickets, \(k=2\) classes, 900 normal and 100 urgent:

    \[
    \begin{aligned}
    w_{\text{normal}} &= 1000/(2\cdot900) \approx 0.556\\
    w_{\text{urgent}} &= 1000/(2\cdot100) = 5\\
    900\cdot\tfrac{1000}{1800} &= 100\cdot5 = 500
    \end{aligned}
    \]

    Each urgent ticket counts 9 times as much as a normal one, and the two classes now carry equal total weight in the loss.

### 20.3 Correcting probabilities after resampling

A model trained on rebalanced data believes the positive rate is the training rate. Its probabilities are too high for the real world. Correct them by rescaling the odds by the ratio of the real prior odds to the training prior odds.

\[
\text{odds}_{\text{real}}=\text{odds}_{\text{model}}\cdot\frac{\pi/(1-\pi)}{\pi_s/(1-\pi_s)}
\]

!!! example "On paper"

    Trained on a 50/50 sample (\(\pi_s=0.5\)), deployed where 10% are urgent (\(\pi=0.1\)), so the factor is \((1/9)/1=1/9\):

    \[
    \begin{aligned}
    p=0.5 &: \text{odds } 1 \to 1/9 \to p = 0.1\\
    p=0.9 &: \text{odds } 9 \to 1 \to p = 0.5
    \end{aligned}
    \]

    A "90% urgent" from the rebalanced model is a coin flip in production. Skip this step and every threshold and every expected-cost calculation downstream is wrong.

### 20.4 In code

Synthetic tickets, 5% urgent. A plain logistic regression, the same model with balanced class weights, and the plain model with lower thresholds:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

rng = np.random.default_rng(0)

def make(n, rate=0.05):                                  # 5% urgent tickets
    y = (rng.random(n) < rate).astype(int)
    X = rng.normal(size=(n, 4)) + 1.2 * y[:, None] * np.array([1.0, 1.0, 0.5, 0.0])
    return X, y

X_train, y_train = make(20_000)
X_test, y_test = make(20_000)

def show(name, pred):
    print(f"{name:24s} flagged={pred.sum():5d}  precision={precision_score(y_test, pred):.2f}"
          f"  recall={recall_score(y_test, pred):.2f}")

plain = LogisticRegression().fit(X_train, y_train)
weighted = LogisticRegression(class_weight="balanced").fit(X_train, y_train)
p = plain.predict_proba(X_test)[:, 1]
print("urgent in test:", y_test.sum())
show("plain, t=0.5", plain.predict(X_test))
show("balanced weights, t=0.5", weighted.predict(X_test))
for t in (0.2, 0.1, 0.05):
    show(f"plain, t={t}", (p >= t).astype(int))
```

```text
urgent in test: 1016
plain, t=0.5             flagged=  333  precision=0.71  recall=0.23
balanced weights, t=0.5  flagged= 4331  precision=0.19  recall=0.82
plain, t=0.2             flagged= 1180  precision=0.43  recall=0.50
plain, t=0.1             flagged= 2416  precision=0.28  recall=0.66
plain, t=0.05            flagged= 4293  precision=0.19  recall=0.81
```

The plain model at 0.5 catches under a quarter of urgent tickets. Balanced weights catch over 80%, at much lower precision. The plain model at a threshold of 0.05 lands at almost the same point, because for a linear model reweighting mostly shifts the intercept. The lesson: class weights and threshold moving are two routes to the same trade-off. Pick the operating point on the precision–recall curve from the cost of each error.

---

## 21. Feature engineering

**Feature engineering** turns raw fields into numbers a model can use. For classical models it is often worth more than the choice of algorithm. It starts after the data is clean: parsing, cleaning and PII handling of raw sources are in [60 — Data preparation](60-ai-data-preparation.md).

### 21.1 Scaling

Linear models, k-NN, k-means and anything trained by gradient descent are sensitive to feature scale; trees are not. **Standardisation** gives every feature mean 0 and standard deviation 1. **Min-max scaling** maps it to \([0,1]\).

\[
z=\frac{x-\mu}{\sigma}
\]

!!! example "On paper"

    Account ages 100, 200 and 300 days; \(\mu=200\) and the population sd is \(\sqrt{(10^4+0+10^4)/3}\approx81.65\):

    \[
    \begin{aligned}
    z &= (-100,0,100)/81.65\\
    &\approx (-1.225,\ 0,\ 1.225)\\
    \text{min-max} &= (0,\ 0.5,\ 1)
    \end{aligned}
    \]

    `StandardScaler` uses the population sd (divide by \(n\)), unlike the sample sd of [46 §13.2](46-ai-math.md). Fit \(\mu\) and \(\sigma\) on training data only, and reuse them at prediction time.

### 21.2 One-hot encoding

A category such as `channel ∈ {email, chat, phone}` becomes one 0/1 column per value: chat = (0, 1, 0) in that order (`OneHotEncoder` sorts the values alphabetically, so in §21.6 chat is the first column). Never encode categories as 0, 1, 2 for a linear model: it would treat "phone" as "twice chat". In production you will meet values that were not in training (a new channel); `OneHotEncoder(handle_unknown="ignore")` maps them to all zeros instead of crashing. For categories with thousands of values (customer IDs), use grouping, target statistics computed inside cross-validation, or let a gradient-boosting library handle them natively.

### 21.3 Text: TF-IDF by hand

**TF-IDF** turns a text into a vector with one entry per vocabulary word: the word's count in the text (term frequency) times a weight that is high for rare words and low for words in every document (inverse document frequency). scikit-learn's default IDF, with smoothing, and then each row scaled to unit L2 length:

\[
\operatorname{idf}(t)=\ln\frac{1+n}{1+\operatorname{df}(t)}+1
\]

!!! example "On paper"

    Three tickets: d1 "refund order", d2 "refund delay", d3 "login error". \(n=3\); "refund" is in 2 documents, every other word in 1:

    \[
    \begin{aligned}
    \operatorname{idf}(\text{refund}) &= \ln(4/3)+1 \approx 1.2877\\
    \operatorname{idf}(\text{order}) &= \ln(4/2)+1 \approx 1.6931\\
    \lVert(1.2877,1.6931)\rVert &= \sqrt{1.6581+2.8667} \approx 2.1272\\
    \text{d1} &\approx (0.6053,\ 0.7960)
    \end{aligned}
    \]

    In d1, "order" outweighs "refund" because "refund" also appears in d2 and so says less about which ticket this is. TF-IDF plus logistic regression is still a hard baseline to beat for routing short texts.

### 21.4 Dates and times

A timestamp is not a useful feature on its own. Extract what plausibly drives the label: hour of day, day of week, "is a weekend", days since signup, days since the last ticket, minutes until the SLA deadline. For cyclic values, the raw number puts 23:00 and 01:00 far apart; encode the hour as a point on a circle.

\[
(\sin\tfrac{2\pi h}{24},\ \cos\tfrac{2\pi h}{24})
\]

!!! example "On paper"

    23:00 and 01:00, with \(\sin15°\approx0.2588\) and \(\cos15°\approx0.9659\):

    \[
    \begin{aligned}
    h=23 &: (-0.2588,\ 0.9659)\\
    h=1 &: (0.2588,\ 0.9659)\\
    \text{distance} &= 2\cdot0.2588 \approx 0.518
    \end{aligned}
    \]

    Two hours apart on the clock and close in feature space, instead of 22 units apart as raw numbers. 06:00 maps to (1, 0) and 18:00 to (−1, 0), the two ends of the circle. Trees can use the raw hour directly.

### 21.5 Missing values

Fill missing numbers with the training median (`SimpleImputer`), and add a 0/1 "was missing" column: missingness is often informative (no phone number on the account says something). Do not fill with 0 when 0 is a real value. scikit-learn's `HistGradientBoostingClassifier` (and, in recent versions, decision trees and random forests) accept `NaN` directly; the older `GradientBoostingClassifier` and linear models do not.

### 21.6 In code

```python
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

age = np.array([[100.0], [200.0], [300.0]])
print("standardised:", StandardScaler().fit_transform(age).ravel().round(4))
print("min-max     :", MinMaxScaler().fit_transform(age).ravel())

enc = OneHotEncoder(handle_unknown="ignore").fit([["email"], ["chat"], ["phone"]])
print("categories  :", enc.categories_[0].tolist())
print("one-hot     :", enc.transform([["chat"], ["fax"]]).toarray().tolist())   # "fax" unseen

tfidf = TfidfVectorizer().fit(["refund order", "refund delay", "login error"])
names = tfidf.get_feature_names_out().tolist()
print("idf         :", dict(zip(names, tfidf.idf_.round(4).tolist())))
row = tfidf.transform(["refund order"]).toarray()[0]
print("d1          :", {w: round(float(v), 4) for w, v in zip(names, row) if v})

t = datetime(2026, 10, 24, 23, 30)                # a Saturday, 23:30
h = t.hour + t.minute / 60
print("date feats  :", {"weekday": t.weekday(), "weekend": t.weekday() >= 5,
                        "hour_sin": round(float(np.sin(2 * np.pi * h / 24)), 4),
                        "hour_cos": round(float(np.cos(2 * np.pi * h / 24)), 4)})
```

```text
standardised: [-1.2247  0.      1.2247]
min-max     : [0.  0.5 1. ]
categories  : ['chat', 'email', 'phone']
one-hot     : [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
idf         : {'delay': 1.6931, 'error': 1.6931, 'login': 1.6931, 'order': 1.6931, 'refund': 1.2877}
d1          : {'order': 0.796, 'refund': 0.6053}
date feats  : {'weekday': 5, 'weekend': True, 'hour_sin': -0.1305, 'hour_cos': 0.9914}
```

---

## 22. A complete pipeline: support-ticket routing

This section puts everything together: a realistic multiclass problem, mixed feature types, a leak-proof pipeline, and an honest report.

### 22.1 The dataset

Real tickets would come from your help desk export. Here a generator with a fixed seed produces 3,000 tickets in four categories with text, channel, customer tier, account age and attachment count, overlapping vocabularies and 5% label noise, so that no model can be perfect. Save it as `tickets.py`; the next blocks import it.

```python
# save as tickets.py
import numpy as np

CATEGORIES = ["billing", "bug", "account", "how_to"]
WORDS = {
    "billing": ["invoice", "charged", "refund", "payment", "card", "price", "twice", "receipt"],
    "bug": ["error", "crash", "500", "broken", "fails", "exception", "timeout", "stack"],
    "account": ["password", "login", "locked", "email", "2fa", "username", "access", "reset"],
    "how_to": ["how", "export", "setup", "configure", "docs", "example", "integrate", "where"],
}
FILLER = ["please", "help", "my", "the", "account", "app", "today", "again", "api", "order"]
CHANNEL_P = [[0.5, 0.3, 0.2], [0.6, 0.3, 0.1], [0.3, 0.3, 0.4], [0.3, 0.6, 0.1]]  # email, chat, phone
TIER_P = [[0.2, 0.5, 0.3], [0.4, 0.4, 0.2], [0.5, 0.3, 0.2], [0.6, 0.3, 0.1]]     # free, pro, enterprise


def make_tickets(n, seed=42):
    rng = np.random.default_rng(seed)
    cls = rng.choice(4, size=n, p=[0.35, 0.30, 0.20, 0.15])
    rows = []
    for c in cls:
        words = list(rng.choice(WORDS[CATEGORIES[c]], size=rng.integers(1, 4)))
        if rng.random() < 0.4:                             # a word from a random topic
            words.append(rng.choice(WORDS[CATEGORIES[rng.integers(4)]]))
        words += list(rng.choice(FILLER, size=rng.integers(2, 6)))
        rng.shuffle(words)
        rows.append([" ".join(str(w) for w in words),
                     str(rng.choice(["email", "chat", "phone"], p=CHANNEL_P[c])),
                     str(rng.choice(["free", "pro", "enterprise"], p=TIER_P[c])),
                     round(float(rng.gamma(2.0, [200, 300, 60, 150][c])), 1),   # account age, days
                     int(rng.poisson([0.8, 1.5, 0.2, 0.3][c]))])                # attachments
    y = np.array([CATEGORIES[c] for c in cls])
    noisy = rng.random(n) < 0.05                           # 5% of labels are wrong
    y[noisy] = rng.choice(CATEGORIES, size=noisy.sum())
    return np.array(rows, dtype=object), y


if __name__ == "__main__":
    X, y = make_tickets(3000)
    for row, label in zip(X[:3], y[:3]):
        print(label, row.tolist())
    print({str(c): int(k) for c, k in zip(*np.unique(y, return_counts=True))})
```

```text
account ['the again access please please locked username', 'phone', 'free', 90.1, 0]
bug ['500 the my app stack please', 'email', 'pro', 735.3, 2]
how_to ['help the my setup example configure configure', 'chat', 'free', 153.1, 0]
{'account': 593, 'billing': 1036, 'bug': 915, 'how_to': 456}
```

The columns are: 0 text, 1 channel, 2 tier, 3 account age in days, 4 attachments. The classes are imbalanced (billing is more than twice as common as how-to), as in a real inbox.

### 22.2 The pipeline

A `ColumnTransformer` sends each column to the right preprocessing step, and a `Pipeline` chains it with the classifier. Everything that learns from data (the TF-IDF vocabulary, the one-hot categories, the scaler's mean) is inside, so `fit` learns it from training rows only and cross-validation cannot leak (§5.3). The same object is what you deploy.

```python
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tickets import make_tickets

X, y = make_tickets(3000)
features = ColumnTransformer([
    ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2), 0),      # column 0: a 1-D text column
    ("categorical", OneHotEncoder(handle_unknown="ignore"), [1, 2]),  # channel, tier
    ("numeric", StandardScaler(), [3, 4]),                            # age, attachments
])
model = Pipeline([("features", features), ("clf", LogisticRegression(C=5.0, max_iter=1000))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
cv = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
print(f"5-fold CV macro-F1 on the training set: {cv.mean():.3f} +/- {cv.std():.3f}")
model.fit(X_train, y_train)
print("features after the transformer:", model[:-1].transform(X_train[:1]).shape[1])
```

```text
5-fold CV macro-F1 on the training set: 0.926 +/- 0.010
features after the transformer: 1099
```

The text column alone becomes more than a thousand TF-IDF features (words and word pairs seen in at least two training tickets); the categorical and numeric columns add 8 more. `C=5.0` (weaker regularization than the default 1.0) is a reasonable setting for this data, not a tuned one; to tune it the way §17 describes, pass the whole pipeline and `{"clf__C": [0.5, 1, 5, 20]}` to `GridSearchCV`, and the search refits the TF-IDF inside every fold.

### 22.3 Training and the classification report

```python
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tickets import make_tickets

X, y = make_tickets(3000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
features = ColumnTransformer([("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2), 0),
                              ("categorical", OneHotEncoder(handle_unknown="ignore"), [1, 2]),
                              ("numeric", StandardScaler(), [3, 4])])
model = Pipeline([("features", features), ("clf", LogisticRegression(C=5.0, max_iter=1000))])
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test), digits=3))
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print(f"majority baseline macro-F1: {f1_score(y_test, base.predict(X_test), average='macro'):.3f}")
new = np.array([["I was charged twice, please refund", "email", "pro", 400.0, 0]], dtype=object)
print("new ticket ->", model.predict(new)[0], np.round(model.predict_proba(new)[0], 3), model.classes_)
```

```text
              precision    recall  f1-score   support

     account      0.943     0.899     0.920       148
     billing      0.913     0.931     0.922       259
         bug      0.931     0.939     0.935       229
      how_to      0.921     0.921     0.921       114

    accuracy                          0.925       750
   macro avg      0.927     0.922     0.924       750
weighted avg      0.926     0.925     0.925       750

majority baseline macro-F1: 0.128
new ticket -> billing [0.002 0.984 0.001 0.014] ['account' 'billing' 'bug' 'how_to']
```

Read the report row by row: precision and recall per class, `support` (how many test tickets of that class), then the averages of §18.10. All four classes sit in the low 0.90s; with 5% of labels randomised, about 96% (95% plus the quarter of noisy labels that land on the right class by chance) is the ceiling, so this model is near what the data allows. The majority baseline's macro-F1 shows how much the model adds. The new ticket comes out as billing with high confidence, from "charged", "twice" and "refund"; "please" carries little weight because every class uses it.

### 22.4 Looking at the errors

Metrics tell you how often; errors tell you why. Print the confusion matrix and read some mistakes:

```python
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tickets import make_tickets

X, y = make_tickets(3000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
features = ColumnTransformer([("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2), 0),
                              ("categorical", OneHotEncoder(handle_unknown="ignore"), [1, 2]),
                              ("numeric", StandardScaler(), [3, 4])])
model = Pipeline([("features", features), ("clf", LogisticRegression(C=5.0, max_iter=1000))])
pred = model.fit(X_train, y_train).predict(X_test)

labels = ["billing", "bug", "account", "how_to"]
print("rows = truth, columns = prediction:", labels)
print(confusion_matrix(y_test, pred, labels=labels))
wrong = np.flatnonzero(pred != y_test)
for i in wrong[:4]:
    print(f"truth={y_test[i]:8s} predicted={pred[i]:8s} text={X_test[i, 0]!r}")
```

```text
rows = truth, columns = prediction: ['billing', 'bug', 'account', 'how_to']
[[241  10   4   4]
 [ 10 215   3   1]
 [  7   4 133   4]
 [  6   2   1 105]]
truth=how_to   predicted=account  text='username order locked my account my where'
truth=billing  predicted=bug      text='the api api exception stack today exception'
truth=billing  predicted=bug      text='crash password exception please stack app'
truth=account  predicted=bug      text='password stack today help today please order'
```

The confusion matrix shows where the model hesitates: billing and bug are confused with each other most (10 each way). The examples show two kinds of error. The second and third texts are plainly bug reports ("exception", "stack", "crash") labelled billing: label noise, which the generator injected on purpose and which a real help desk has too. The first and fourth mix vocabularies ("username" and "where", "password" and "stack"): genuinely ambiguous tickets that a human would also hesitate on. On real data this is where you find the fixes: a missing feature, a confusing pair of categories that should be merged, or a labelling guideline that agents read differently. The same discipline is the core of LLM error analysis ([54](54-ai-evaluation.md)).

---

## 23. Model persistence and versioning

A trained model is a build artifact. Treat it like one: serialise it, hash it, record what produced it, and refuse to load what you cannot trust.

```python
import hashlib
import json
import pickle
import platform

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tickets import make_tickets

X, y = make_tickets(3000)
model = Pipeline([("features", ColumnTransformer([
    ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2), 0),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), [1, 2]),
    ("numeric", StandardScaler(), [3, 4])])),
    ("clf", LogisticRegression(C=5.0, max_iter=1000))]).fit(X, y)

blob = pickle.dumps(model, protocol=5)
with open("ticket_router.pkl", "wb") as f:
    f.write(blob)
meta = {"model": "ticket_router", "version": "2026-10-28.1",
        "sha256": hashlib.sha256(blob).hexdigest(),
        "sklearn": sklearn.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "training_data": "make_tickets(3000, seed=42)", "classes": model.classes_.tolist()}
with open("ticket_router.json", "w") as f:
    json.dump(meta, f, indent=2)

with open("ticket_router.pkl", "rb") as f:          # at load time: verify, then unpickle
    data = f.read()
assert hashlib.sha256(data).hexdigest() == meta["sha256"], "artifact changed"
loaded = pickle.loads(data)
print("same predictions after reload:", (loaded.predict(X) == model.predict(X)).all())
print({k: meta[k] for k in ("version", "sklearn", "classes")})
```

```text
same predictions after reload: True
{'version': '2026-10-28.1', 'sklearn': '1.9.1', 'classes': ['account', 'billing', 'bug', 'how_to']}
```

The rules behind this block:

- **Pickle runs code when it loads.** scikit-learn's documentation is blunt: never load a pickle (or joblib) file from an untrusted source. Load only artifacts your own pipeline produced, verified by hash. For untrusted or shared artifacts, use `skops.io` (which lets you review the types an artifact contains before loading) or export to ONNX for serving.
- **Pin versions.** scikit-learn does not support loading a model saved with a different version; it warns with `InconsistentVersionWarning`, and results may differ silently. Record the versions in the metadata and serve with the same lock file you trained with.
- **Record what rebuilds the model**: training data snapshot or query, code commit, hyperparameters, and the validation metrics. That metadata is your model card ([56](56-ai-safety.md) on documentation).
- **Version like a release**: a model registry (or a bucket with immutable, versioned paths), a canary rollout, and a rollback that is one config change. [55](55-ai-production.md) covers the serving side; the same applies to prompts and LoRA adapters.

`joblib.dump` and `joblib.load` are the same idea with faster handling of large numpy arrays, and the same security caveat.

---

## 24. When classical ML beats an LLM

### 24.1 The criteria

An LLM can classify tickets zero-shot, which makes it the fastest prototype. A classical model often wins once the task is defined and labelled data exists. It wins when:

- **Latency** matters: the arithmetic of a logistic regression takes microseconds per ticket on a CPU (one call to a scikit-learn pipeline adds a few milliseconds of Python overhead, which batching spreads over many tickets); an LLM needs hundreds of milliseconds to seconds.
- **Cost per prediction** matters at volume (§24.2).
- **The data is tabular**: numbers and categories (transactions, usage logs, prices). Gradient-boosted trees are the state of the practice there; serialising a row into a prompt throws away the structure.
- **Explainability is required**: coefficients, tree paths and feature importances can be audited; credit, insurance and HR decisions often need that.
- **Determinism and stability**: the same input gives the same output, and the model does not change when a provider updates theirs.
- **Calibrated probabilities** feed a downstream decision (§18.9).
- **Labels are plentiful** and the task is narrow: thousands of labelled tickets for six fixed queues.

An LLM wins when there are few or no labels, the categories change weekly, the input needs real language understanding or world knowledge (a sarcastic complaint, a legal clause), or the output is text. A common path: prototype with an LLM, log its decisions, have humans correct a sample, then train a classical model on the corrected labels and keep the LLM for the cases the classifier is unsure about (§25.1).

### 24.2 A cost comparison in numbers

\[
\text{monthly cost}=n\cdot(t_{\text{in}}\,p_{\text{in}}+t_{\text{out}}\,p_{\text{out}})
\]

!!! example "On paper"

    3 million tickets a month. The LLM prompt is 600 input tokens (instructions plus ticket) and the answer is a 10-token label. Assume a hosted model at $0.50 per million input tokens and $2 per million output tokens:

    \[
    \begin{aligned}
    \text{input} &: 3\cdot10^6\cdot600 = 1.8\cdot10^9 \text{ tokens}\\
    &\to 1800\cdot\$0.50 = \$900\\
    \text{output} &: 3\cdot10^7 \text{ tokens} \to 30\cdot\$2 = \$60\\
    \text{total} &= \$960 \text{ a month}
    \end{aligned}
    \]

    The classifier: assume 0.2 ms of one CPU core per ticket, so \(3\cdot10^6\cdot0.2\) ms = 600 CPU-seconds a month, which rounds to nothing. Self-hosting does not rescue the LLM at this volume: assume the M1 Pro prefills Qwen3-14B prompts at about 120 tokens/s (llama.cpp's Apple Silicon benchmark measures about 266 tokens/s for a 4-bit 7B model on the 16-core M1 Pro; scaled to 14.8B parameters), so 5 s of prefill per ticket before the label is even decoded, and \(1.5\cdot10^7\) s a month: about 5.8 Macs running non-stop (a 30-day month is \(2.592\cdot10^6\) s).

$960 a month may be acceptable; the point is the ratio, and that the classifier's cost does not grow with prompt length. Rerun the arithmetic with your own prices and volumes ([55](55-ai-production.md) covers API versus self-hosting break-even).

### 24.3 A decision table

| Situation | Start with |
|---|---|
| No labels, categories still changing | an LLM with a good prompt ([50](50-ai-prompting-apis.md)) |
| Thousands of labels, fixed categories, high volume | TF-IDF or embeddings + logistic regression |
| Tabular data (numbers, categories) | gradient-boosted trees |
| Needs an explanation per decision | logistic regression or a shallow tree |
| Long, nuanced text; reasoning; text output | an LLM, possibly fine-tuned ([53](53-ai-fine-tuning.md)) |
| High volume, some hard cases | a classifier first, the LLM for the uncertain rest |

---

## 25. Classical ML inside LLM systems

Classical models do not disappear when you build on LLMs; they become components.

### 25.1 Routers and cascades

A **router** is a cheap classifier that decides where a request goes: a small or a large model, a FAQ answer or a full RAG pipeline, a human. A **cascade** tries the cheap path first and escalates when the cheap path is unsure.

\[
\text{blended cost}=s\,c_{\text{small}}+(1-s)\,c_{\text{large}}
\]

!!! example "On paper"

    A router sends 70% of questions (\(s=0.7\)) to a small model. Assume $0.10 per thousand requests on the small model and $1.00 on the large one:

    \[
    \begin{aligned}
    \text{blended} &= 0.7\cdot0.10+0.3\cdot1.00\\
    &= 0.07+0.30 = \$0.37 \text{ per thousand}\\
    \text{saving} &= 1-0.37 = 63\%
    \end{aligned}
    \]

    The saving is real only if the small model's answers on the routed 70% are good enough. Evaluate the router like any classifier: its "errors" are hard questions sent to the small model.

### 25.2 Classifiers on embeddings

An embedding model turns text into a vector; a logistic regression on top of the vectors is a fast, accurate intent classifier that you can retrain in seconds when you add an intent. Here the embeddings are synthetic (three intents as clusters in 64 dimensions) so the block runs anywhere; with a real embedding model you would call it in place of the generator. The classifier's confidence drives a cascade: confident predictions are handled directly, the rest go to the LLM.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
intents = ["refund", "password_reset", "other"]
centres = rng.normal(size=(3, 64))
y = rng.choice(3, size=3000, p=[0.4, 0.3, 0.3])
E = centres[y] + rng.normal(0, 3.5, size=(3000, 64))        # noisy "embeddings"
E /= np.linalg.norm(E, axis=1, keepdims=True)                # unit length, like most embedding APIs

E_train, E_test, y_train, y_test = train_test_split(E, y, test_size=0.3, random_state=0)
clf = LogisticRegression(C=10.0, max_iter=1000).fit(E_train, y_train)
proba = clf.predict_proba(E_test)
pred, conf = proba.argmax(axis=1), proba.max(axis=1)
print(f"accuracy on all test queries: {(pred == y_test).mean():.3f}")

for t in (0.6, 0.8, 0.9):
    sure = conf >= t
    print(f"threshold {t}: classifier answers {sure.mean():.0%} with accuracy "
          f"{(pred[sure] == y_test[sure]).mean():.3f}; {1 - sure.mean():.0%} go to the LLM")
```

```text
accuracy on all test queries: 0.910
threshold 0.6: classifier answers 94% with accuracy 0.933; 6% go to the LLM
threshold 0.8: classifier answers 82% with accuracy 0.962; 18% go to the LLM
threshold 0.9: classifier answers 73% with accuracy 0.976; 27% go to the LLM
```

Raising the confidence threshold sends more traffic to the LLM and makes the classifier's share more accurate: the same precision-versus-coverage trade-off as §18.5. You choose the threshold from the cost of a wrong automatic answer against the cost of an LLM call.

### 25.3 Rerankers

A RAG system retrieves 50 candidate chunks cheaply, then a **reranker** reorders them so the best 5 go into the prompt ([51](51-ai-rag.md)). Modern rerankers are cross-encoder neural networks, but the idea is learning-to-rank, a classical ML problem: features of a (query, document) pair in, a relevance score out, trained on click or label data and evaluated with ranking metrics. A logistic regression over a handful of features (BM25 score, cosine similarity, document freshness, source authority) is a legitimate first reranker and easy to debug.

### 25.4 Other places you will find it

- **Guardrails**: a fast classifier on the input (prompt-injection or toxicity detector) or the output (PII present?) before an LLM call or before showing the answer ([56](56-ai-safety.md)).
- **Anomaly detection on telemetry**: z-scores on tokens per request, cost per user and tool-call counts catch runaway agents and abuse ([55](55-ai-production.md)).
- **Clustering for error analysis**: k-means on embeddings of failed conversations groups thousands of failures into a dozen themes you can read.
- **Calibration of LLM scores**: logistic regression on an LLM judge's raw scores, fitted to human labels, turns them into probabilities you can threshold ([54](54-ai-evaluation.md)).
- **Evaluation itself**: every metric in §18, and every split rule in §4, applies unchanged to LLM evals.

---

## 26. Key takeaways

- Machine learning is data + model family + loss. The goal is performance on **unseen** data; everything else serves that.
- Split into **train / validation / test**; stratify rare classes, split by **time** for forecasting and by **group** for users, documents and repos. Touch the test set once.
- **Leakage** (future fields, duplicates across splits, preprocessing fitted on all data) makes offline scores lie. Put every fitted step inside a `Pipeline`.
- Always report a **baseline** (majority class, a keyword rule).
- Linear regression has a closed form; logistic regression has the gradient \((p-y)\mathbf x\); trees split on impurity decrease; forests average decorrelated trees; boosting fits residuals; k-NN votes among neighbours (scale first); k-means alternates assign and average; naive Bayes multiplies smoothed word probabilities.
- Compare **training and validation** error to diagnose under- and overfitting; regularize (L2 shrinks, L1 zeroes) and choose its strength by **cross-validation**.
- On imbalanced data, accuracy lies: use **precision, recall, F-beta, PR-AUC**, choose the **threshold** from error costs, and check **calibration** (Brier, reliability bins) when probabilities are used as numbers.
- Classical models win on **latency, cost, tabular data and explainability**, and they live on inside LLM systems as routers, rerankers, guardrails and classifiers on embeddings.

---

## 27. Exercises

!!! question "Exercise 27.1"

    A classifier flags 50 tickets as urgent; 40 really are, and 60 urgent tickets exist in total. Compute precision, recall and F1.

??? success "Answer"

    Precision \(=40/50=0.80\). Recall \(=40/60\approx0.667\). F1 \(=2\cdot0.8\cdot0.667/(0.8+0.667)\approx0.727\); equivalently \(2TP/(2TP+FP+FN)=80/(80+10+20)=80/110\approx0.727\).

!!! question "Exercise 27.2"

    Two positives score 0.8 and 0.4; three negatives score 0.6, 0.3 and 0.1. Compute the ROC-AUC with the pairwise definition, and the average precision.

??? success "Answer"

    6 pairs. 0.8 beats all three negatives (3); 0.4 beats 0.3 and 0.1 (2). AUC \(=5/6\approx0.833\). Ranked by score: 0.8 (pos), 0.6 (neg), 0.4 (pos), ... Precision at the first positive is 1/1, at the second 2/3. AP \(=\tfrac12(1+0.667)\approx0.833\).

!!! question "Exercise 27.3"

    A node holds 8 billing and 2 bug tickets. A split sends 5 billing tickets left and 3 billing plus 2 bug tickets right. Compute the parent's Gini impurity, the children's, and the impurity decrease.

??? success "Answer"

    Parent: \(1-0.8^2-0.2^2=0.32\). Left (pure): 0. Right: \(1-0.6^2-0.4^2=0.48\). Weighted children: \(0.5\cdot0+0.5\cdot0.48=0.24\). Decrease: \(0.32-0.24=0.08\).

!!! question "Exercise 27.4"

    Fit \(\hat y=wx+b\) by the closed form to the points (1, 1), (2, 3), (3, 3), (4, 5). Give \(w\), \(b\), the MSE and \(R^2\).

??? success "Answer"

    \(\bar x=2.5\), \(\bar y=3\). \(\sum(x-\bar x)(y-\bar y)=(-1.5)(-2)+(-0.5)(0)+(0.5)(0)+(1.5)(2)=6\); \(\sum(x-\bar x)^2=2.25+0.25+0.25+2.25=5\). \(w=6/5=1.2\), \(b=3-1.2\cdot2.5=0\). Predictions \((1.2,2.4,3.6,4.8)\), residuals \((-0.2,0.6,-0.6,0.2)\), squared sum \(0.8\), MSE \(=0.2\). \(\sum(y-\bar y)^2=4+0+0+4=8\), so \(R^2=1-0.8/8=0.9\).

!!! question "Exercise 27.5"

    A logistic model predicts 0.8 for a positive ticket and 0.4 for a negative one. Compute the mean log-loss, and the gradient of the loss with respect to the logit for each ticket.

??? success "Answer"

    \(-\ln0.8\approx0.223\) and \(-\ln(1-0.4)=-\ln0.6\approx0.511\); mean \(\approx0.367\). Gradients \(p-y\): \(0.8-1=-0.2\) (push the logit up) and \(0.4-0=0.4\) (push it down, harder).

!!! question "Exercise 27.6"

    Missing a fraudulent payment costs $40 on average; reviewing a legitimate one costs $2. Which probability threshold minimises the expected cost? What goes wrong if the model was trained on a 50/50 resampled set and you forget to correct its probabilities?

??? success "Answer"

    \(t^*=2/(2+40)=2/42\approx0.048\): review every payment above about 5%. A model trained on 50/50 data overstates the fraud probability (§20.3), so far too many payments cross 0.048, and the review cost explodes. Correct the probabilities to the real prior first, then apply the threshold.

!!! question "Exercise 27.7"

    You run 5-fold cross-validation on 2,000 examples over a grid of 4 values of `C` and 5 values of `l1_ratio`. How many examples does each training run see, and how many model fits does `GridSearchCV` perform in total?

??? success "Answer"

    Each fold validates on 400 and trains on 1,600. The grid has \(4\cdot5=20\) combinations, so \(20\cdot5=100\) fits, plus 1 refit of the best combination on all 2,000 examples: 101.

!!! question "Exercise 27.8"

    A dataset has 10,000 tickets, of which 200 are escalations. What weights does `class_weight="balanced"` give the two classes, and how many times does an escalation count relative to a normal ticket?

??? success "Answer"

    \(w_{\text{esc}}=10000/(2\cdot200)=25\); \(w_{\text{normal}}=10000/(2\cdot9800)\approx0.510\). Ratio \(25/0.510=49\), which is \(9800/200\): each escalation counts 49 times as much.

!!! question "Exercise 27.9"

    A churn model for a SaaS product uses these features. Which ones leak, and how would you split the data? (a) number of support tickets in the last 30 days; (b) `cancellation_reason`; (c) plan price; (d) whether the account was flagged by the retention team; (e) days since last login, computed from today's database.

??? success "Answer"

    (b) leaks: it is only filled in after the customer cancels. (d) probably leaks: the retention team flags accounts they already think are churning, often after the signals the model should learn. (e) leaks if computed "as of today" for historical rows; compute it as of each row's prediction date. (a) and (c) are fine if computed as of the prediction date. Split by time (train on earlier months, test on later), and keep all rows of one customer on one side.

!!! question "Exercise 27.10"

    1 million emails a month must be classified into 5 folders. An LLM prompt is 500 input tokens and the answer 5 output tokens. Assume $1 per million input tokens and $4 per million output tokens. What is the monthly LLM cost? A router sends 80% of emails to a classifier (cost assumed 0) and 20% to the LLM; what is the new cost?

??? success "Answer"

    Input: \(10^6\cdot500=5\cdot10^8\) tokens \(=500\) million, \(\$500\). Output: \(5\cdot10^6\) tokens, \(5\cdot\$4=\$20\). Total \(\$520\) a month. With the router, 20% of that: \(\$104\), a saving of \(\$416\) (80%), if the classifier's accuracy on its 80% is acceptable.

---

Next: [48 — Deep learning](48-ai-deep-learning.md)
