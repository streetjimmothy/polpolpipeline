import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import sys

full_g1 = pd.read_csv(sys.argv[1])
full_g2 = pd.read_csv(sys.argv[2])

print (f"Group1 rows: {len(full_g1)}")
print (f"Group2 rows: {len(full_g2)}")
print (f"Group1 0 rows: {len(full_g1[full_g1['bert_prediction']==0])} ({(len(full_g1[full_g1['bert_prediction']==0])/len(full_g1))*100:.2f}%)")
print (f"Group1 1 rows: {len(full_g1[full_g1['bert_prediction']==1])} ({(len(full_g1[full_g1['bert_prediction']==1])/len(full_g1))*100:.2f}%)")
print (f"Group1 2 rows: {len(full_g1[full_g1['bert_prediction']==2])} ({(len(full_g1[full_g1['bert_prediction']==2])/len(full_g1))*100:.2f}%)")
print (f"Group1 3 rows: {len(full_g1[full_g1['bert_prediction']==3])} ({(len(full_g1[full_g1['bert_prediction']==3])/len(full_g1))*100:.2f}%)")
print (f"Group2 0 rows: {len(full_g2[full_g2['bert_prediction']==0])} ({(len(full_g2[full_g2['bert_prediction']==0])/len(full_g2))*100:.2f}%)")
print (f"Group2 1 rows: {len(full_g2[full_g2['bert_prediction']==1])} ({(len(full_g2[full_g2['bert_prediction']==1])/len(full_g2))*100:.2f}%)")
print (f"Group2 2 rows: {len(full_g2[full_g2['bert_prediction']==2])} ({(len(full_g2[full_g2['bert_prediction']==2])/len(full_g2))*100:.2f}%)")
print (f"Group2 3 rows: {len(full_g2[full_g2['bert_prediction']==3])} ({(len(full_g2[full_g2['bert_prediction']==3])/len(full_g2))*100:.2f}%)")

#columns: tweet, label
#the labels are is sort-of ordered-metric (3 is nominally epistemically worse than 2, which is worse than 1). But 0 is more like 'null' than a part of the order, so we remove that.
def zero_filter(source):
	return pd.concat([pd.DataFrame(), source[source['bert_prediction'] != 0]], ignore_index=True)

nat_g1 = zero_filter(full_g1)
nat_g2 = zero_filter(full_g2)

print(f"Group1 Non-zero rows: {len(nat_g1)} of {len(full_g1)}")
print(f"Group2 Non-zero rows: {len(nat_g2)} of {len(full_g2)}")



#basic stats
print("\nBasic stats:")
print(f"Group1 mean: {nat_g1['bert_prediction'].mean():.3f}, std: {nat_g1['bert_prediction'].std():.3f}")
print(f"Group1 median: {nat_g1['bert_prediction'].median():.3f}")
print(f"Group2 mean: {nat_g2['bert_prediction'].mean():.3f}, std: {nat_g2['bert_prediction'].std():.3f}")
print(f"Group2 median: {nat_g2['bert_prediction'].median():.3f}")


# ---------------------------
# 2. H1: Wilcoxon (median > 0)
# ---------------------------
# One-sample, one-sided: H0 median = 0, H1 median > 0

w_g1, p_g1= wilcoxon(nat_g1['bert_prediction'] - 1, alternative="greater")
w_g2, p_g2 = wilcoxon(nat_g2['bert_prediction'] - 1, alternative="greater")

print("\nH1: Wilcoxon vs 0")
print(f"Group1: W={w_g1}, p={p_g1}")
print(f"Group2: W={w_g2}, p={p_g2}")

# ---------------------------
# 3a. H2: Mann–Whitney U (directional)
# ---------------------------
# H1: conservative > other

u, p_u = mannwhitneyu(nat_g1['bert_prediction']-1, nat_g2['bert_prediction']-1)

print("\nH2: Mann–Whitney U (conservative > other)")
print(f"U={u}, p={p_u}")


# Effect size (rank-biserial)
n1, n2 = len(nat_g1['bert_prediction']), len(nat_g2['bert_prediction'])
r_rb = 1 - (2 * u) / (n1 * n2)
print(f"Rank-biserial r = {r_rb:.3f}")

exit()

# ---------------------------
# 3b. H2: Ordinal Logistic Regression
# ---------------------------
# Binary predictor: 1 = conservative, 0 = other
df["group_bin"] = (df["group"] == CONS).astype(int)

model = OrderedModel(
    df["score"],
    sm.add_constant(df["group_bin"]),
    distr="logit"
)

res = model.fit(method="bfgs", disp=False)
print("\nOrdinal Logistic Regression")
print(res.summary())

beta = res.params["group_bin"]
se = res.bse["group_bin"]
z = beta / se
p_one_tailed = 1 - sm.stats.norm.cdf(z)
odds_ratio = np.exp(beta)
ci_low = np.exp(beta - 1.96 * se)
ci_high = np.exp(beta + 1.96 * se)

print("\nDirectional test (Conservative > Other)")
print(f"β = {beta:.3f}, z = {z:.3f}, p(one-tailed) = {p_one_tailed:.5f}")
print(f"Odds Ratio = {odds_ratio:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]")
