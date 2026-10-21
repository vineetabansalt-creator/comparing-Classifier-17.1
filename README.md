# Comparing Classifiers: Bank Telemarketing Term-Deposit Prediction

**Practical Application III** · UC Berkeley Professional Certificate in Machine Learning & AI

📓 **Notebook:** [comparing_classifiers.ipynb](comparing_classifiers.ipynb)

## Business Problem
A Portuguese bank sells long-term deposits through outbound phone campaigns. Only about **1 in 9 calls succeeds**, and every call costs agent time. The goal is to predict, **before dialling**, which clients are likely to subscribe, so the call centre can contact the most promising clients first and reach most buyers with far fewer calls.

## Data
- [UCI Bank Marketing dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) (`bank-additional-full.csv`): 41,188 contacts from **17 campaigns (May 2008 – Nov 2010)**, 20 features plus the target.
- Reference paper: Moro, Laureano & Cortez, *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*.
- Data preparation: removed 12 duplicates; kept `'unknown'` as its own category; replaced the `pdays = 999` sentinel with a `prev_contacted` flag; **excluded `duration`**, because it is only known after the call (target leakage).

## Approach (CRISP-DM)
1. **EDA:** distributions, subscription rates by category, correlation heatmap, plus chi-square and Mann–Whitney tests with effect sizes.
2. **Baseline:** always predict "No" gives **88.7% accuracy** with **0% of subscribers found**, which shows why accuracy is the wrong metric.
3. **Default models on bank-client features only:** Logistic Regression, KNN, Decision Tree and SVM all failed to beat the baseline (test ROC-AUC 0.58–0.65).
4. **Improved models:** used all pre-call features (campaign context and economic indicators), tuned with `GridSearchCV` using stratified 5-fold CV, and **optimised ROC-AUC**, since the business task is ranking clients and the classes are imbalanced (11% "yes").

## Results

| Model | Test ROC-AUC | Recall | Precision | Train time |
|---|---|---|---|---|
| Decision Tree (max_depth=7) | **0.806** | 22% | 64% | 0.3s |
| Logistic Regression (C=0.1, balanced) | 0.803 | **65%** | 36% | 0.5s |
| KNN (k=101) | 0.795 | 21% | 68% | 0.1s |
| SVM (RBF, C=1) | 0.693 | 24% | 67% | 144s |

![Cumulative gains](images/cumulative_gains.png)

**Recommended model: Logistic Regression.** It is statistically tied with the Decision Tree on ranking, while being the most transparent model and finding the most subscribers at the default threshold. SVM overfit and was about 300x slower to train.

## Key Findings (for the marketing team)
- **Call the top-scored 20% of clients to reach ~65% of all subscribers.** Those calls convert at ~37%, versus 11% for random calling (3.3x more efficient).
- **Past responders convert at 65%.** Always put them first.
- **Timing matters:** May had a third of all calls but the lowest conversion (6%). March, September, October and December convert at 44–50%. Uptake was much higher when interest rates (Euribor) and employment were low.
- **Mobile beats landline:** 15% vs 5% conversion.
- **Diminishing returns from repeat calls:** conversion drops from 13% on the first call to about 5% by the sixth. Consider capping at around 3 calls per client per campaign.
- **Most responsive segments:** students (31%) and retirees (25%).

## Next Steps
- Set the call cut-off using real cost-per-call and profit-per-deposit figures.
- Validate in a live A/B pilot against the current calling approach.
- Try ensemble models (Random Forest, Gradient Boosting).
- Add richer client data (balances, product holdings).
- Validate on later time periods and retrain as economic conditions change.

## Repository Structure
```
├── README.md
├── comparing_classifiers.ipynb   # full analysis
├── data/
│   └── bank-additional-full.csv
└── images/                       # figures used in the notebook and README
```

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
jupyter notebook comparing_classifiers.ipynb
```
Full execution takes about 10–15 minutes on a single CPU core, mostly SVM training.
