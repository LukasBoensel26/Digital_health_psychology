from scipy.stats import ttest_rel, wilcoxon, shapiro, ttest_ind, mannwhitneyu, spearmanr, pearsonr #statistical evaluation
import seaborn as sns #data visualization
import matplotlib.pyplot as plt #plotting
import pandas as pd #statistical evaluation/maths


#load both datasets
post_sssq = pd.read_excel(r'CortisolAmaylaseSSSQPANAS_post.xlsx', engine='openpyxl')
pre_sssq = pd.read_excel(r'CortisolAmaylaseSSSQPANAS_pre.xlsx', engine='openpyxl')

combined_data = pd.merge(pre_sssq, post_sssq, on="VP", suffixes=('_pre', '_post')) #merch pre and post dataset to one
combined_data = combined_data[combined_data['Group_post'] != '?'] #exclude person 7

#SSSQ subscales pre and post
engagement_pre = [f"Pre_SSSQ_{i}_pre" for i in [2, 5, 11, 12, 13, 17, 21, 22]]
distress_pre = [f"Pre_SSSQ_{i}_pre" for i in [1, 3, 4, 6, 7, 8, 9, 10]]
worry_pre = [f"Pre_SSSQ_{i}_pre" for i in [14, 15, 16, 18, 19, 20, 23, 24]]

engagement_post = [f"Pre_SSSQ_{i}_post" for i in [2, 5, 11, 12, 13, 17, 21, 22]]
distress_post = [f"Pre_SSSQ_{i}_post" for i in [1, 3, 4, 6, 7, 8, 9, 10]]
worry_post = [f"Pre_SSSQ_{i}_post" for i in [14, 15, 16, 18, 19, 20, 23, 24]]

#Mean SSSQ subscales pre
combined_data['engagement_pre'] = combined_data[engagement_pre].mean(axis=1)
combined_data['distress_pre'] = combined_data[distress_pre].mean(axis=1)
combined_data['worry_pre'] = combined_data[worry_pre].mean(axis=1)

#Mean SSSQ subscales post
combined_data['engagement_post'] = combined_data[engagement_post].mean(axis=1)
combined_data['distress_post'] = combined_data[distress_post].mean(axis=1)
combined_data['worry_post'] = combined_data[worry_post].mean(axis=1)

#Total mean SSSQ pre and post
combined_data['totalSSSQ_pre'] = combined_data[engagement_pre + distress_pre + worry_pre].mean(axis=1)
combined_data['totalSSSQ_post'] = combined_data[engagement_post + distress_post + worry_post].mean(axis=1)

# calculate differences between pre and post SSSQ
combined_data['engagement_diff'] = combined_data['engagement_post'] - combined_data['engagement_pre']
combined_data['distress_diff'] = combined_data['distress_post'] - combined_data['distress_pre']
combined_data['worry_diff'] = combined_data['worry_post'] - combined_data['worry_pre']
combined_data['totalSSSQ_diff'] = combined_data['totalSSSQ_post'] - combined_data['totalSSSQ_pre']

#calculate differences between pre and post cortisol
combined_data['cortisol_diff'] = combined_data['Cortisol_post'] - combined_data['Cortisol_pre']

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

combined_data = remove_outliers(combined_data, 'cortisol_diff')

#Test auf NV und Korrelation SSSQ
for scale in ['engagement', 'distress', 'worry', 'totalSSSQ']:
    print(f"Test für {scale}")

    for group in ['cg', 'eg']:
        print(f"Analyse innerhalb der Gruppe {group.upper()}")
        group_data = combined_data[combined_data['Group_pre'] == group]

        # Variablennamen dynamisch festlegen
        cortisol = "cortisol_diff"
        scale_column = f"{scale}_diff"

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

# Separate scatter plots for each SSSQ scale in TSST group (Cortisol)
sssq_scales = {"engagement_diff": "Engagement", "distress_diff": "Distress", "worry_diff": "Worry", "totalSSSQ_diff": "Total SSSQ"}

for scale, title in sssq_scales.items():
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=combined_data_eg['cortisol_diff'], y=combined_data_eg[scale], color=custom_palette[0])
    plt.xlabel("Cortisol Difference")
    plt.ylabel(f"{title} Difference")
    plt.title(f"Relationship Between Cortisol Difference and {title} (TSST Group)")
    plt.show()

