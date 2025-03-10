from scipy.stats import spearmanr, pearsonr, shapiro  # statistical evaluation
import seaborn as sns  # data visualization
import matplotlib.pyplot as plt  # plotting
import pandas as pd  # statistical evaluation/maths
import numpy as np

# Load datasets
post_sssq = pd.read_excel(r'CortisolAmaylaseSSSQPANAS_post.xlsx', engine='openpyxl')
pre_sssq = pd.read_excel(r'CortisolAmaylaseSSSQPANAS_pre.xlsx', engine='openpyxl')

# Merge pre and post datasets
combined_data = pd.merge(pre_sssq, post_sssq, on="VP", suffixes=('_pre', '_post'))
combined_data = combined_data[combined_data['Group_post'] != '?']  # exclude person 7
combined_data = combined_data.dropna(subset=['v_26_post'])
combined_data = combined_data[combined_data['v_26_post'].astype(str).str.strip() != '']

# Define variables for analysis
positiveAffect_pre = [f"v_{i}_pre" for i in [26, 28, 29, 31, 35, 50, 52, 54, 56, 57]]
negativeAffect_pre = [f"v_{i}_pre" for i in [27, 30, 32, 33, 34, 51, 53, 55, 58, 59]]
positiveAffect_post = [f"v_{i}_post" for i in [26, 28, 29, 31, 35, 50, 52, 54, 56, 57]]
negativeAffect_post = [f"v_{i}_post" for i in [27, 30, 32, 33, 34, 51, 53, 55, 58, 59]]

# Compute means per person
combined_data['positiveAffect_pre'] = combined_data[positiveAffect_pre].mean(axis=1)
combined_data['negativeAffect_pre'] = combined_data[negativeAffect_pre].mean(axis=1)
combined_data['positiveAffect_post'] = combined_data[positiveAffect_post].mean(axis=1)
combined_data['negativeAffect_post'] = combined_data[negativeAffect_post].mean(axis=1)
combined_data['total_pre'] = combined_data[positiveAffect_pre + negativeAffect_pre].mean(axis=1)
combined_data['total_post'] = combined_data[positiveAffect_post + negativeAffect_post].mean(axis=1)

# Compute differences
combined_data['positiveAffect_diff'] = combined_data['positiveAffect_post'] - combined_data['positiveAffect_pre']
combined_data['negativeAffect_diff'] = combined_data['negativeAffect_post'] - combined_data['negativeAffect_pre']
combined_data['total_diff'] = combined_data['total_post'] - combined_data['total_pre']
combined_data['amylase_diff'] = combined_data['Amylase_post'] - combined_data['Amylase_pre']
combined_data['cortisol_diff'] = combined_data['Cortisol_post'] - combined_data['Cortisol_pre']


for scale in ['positiveAffect', 'negativeAffect', 'total']:
    print(f"Test für {scale}")

    for group in ['cg', 'eg']:
        print(f"Analyse innerhalb der Gruppe {group.upper()}")
        group_data = combined_data[combined_data['Group_pre'] == group]

        # Variablennamen dynamisch festlegen
        cortisol = "cortisol_diff"
        scale_column = f"{scale}_diff"

        # Prüfen, ob genug Werte für Shapiro-Test vorhanden sind
        if len(group_data) < 3:
            print(f"Zu wenige Datenpunkte für Shapiro-Wilk-Test in {group.upper()} ({len(group_data)} Datenpunkte).")
            continue  # Überspringt den Shapiro-Test und die Korrelation

        # Shapiro-Wilk-Test für Normalverteilung
        cortisol_normal = shapiro(group_data[cortisol]).pvalue >= 0.05
        scale_normal = shapiro(group_data[scale_column]).pvalue >= 0.05

        # Testentscheidung basierend auf Normalverteilung
        if cortisol_normal and scale_normal:
            r, p = pearsonr(group_data[cortisol], group_data[scale_column])
            test_name = "Pearson"
        else:
            r, p = spearmanr(group_data[cortisol], group_data[scale_column])
            test_name = "Spearman"

        print(f"{test_name} für {scale}:")
        print(f"T-Wert: {r:.3f}, P-Wert: {p:.3f}\n")


# Define color palette
custom_palette = ["#009e8e", "#004a8f"]

# Filter for TSST group (EG)
combined_data_eg = combined_data[combined_data['Group_pre'] == 'eg']

# Separate scatter plots for each scale in TSST group
scale_titles = {"positiveAffect_diff": "Positive Affect", "negativeAffect_diff": "Negative Affect", "total_diff": "Total"}

for scale, title in scale_titles.items():
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=combined_data_eg['cortisol_diff'], y=combined_data_eg[scale], color=custom_palette[0])
    plt.xlabel("Cortisol Difference")
    plt.ylabel(f"{title} Difference")
    plt.title(f"Relationship Between Cortisol Difference and {title} (TSST Group)")
    plt.show()