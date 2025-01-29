from scipy.stats import spearmanr, pearsonr, shapiro #statistical evaluation
import seaborn as sns #data visualization
import matplotlib.pyplot as plt #plotting
import pandas as pd #statistical evaluation/maths

#load both datasets
post_sssq = pd.read_excel(r'CortisolAmaylaseSSSQPANAS_post.xlsx', engine='openpyxl')
pre_sssq = pd.read_excel(r'CortisolAmaylaseSSSQPANAS_pre.xlsx', engine='openpyxl')

combined_data = pd.merge(pre_sssq, post_sssq, on="VP", suffixes=('_pre', '_post')) #merch pre and post dataset to one
combined_data = combined_data[combined_data['Group_post'] != '?'] #exclude person 7


# Reverse-Coding-Funktion (1 wird zu 5, 2 zu 4, usw.)
def reverse_code(series):
    return 6 - series


reverse_coded_items = [2, 11, 13, 17, 21, 22]

# Reverse-Coding
for i in reverse_coded_items:
    pre_col = f"Pre_SSSQ_{i}_pre"
    post_col = f"Pre_SSSQ_{i}_post"

    if pre_col in combined_data.columns:
        combined_data[pre_col] = reverse_code(combined_data[pre_col])

    if post_col in combined_data.columns:
        combined_data[post_col] = reverse_code(combined_data[post_col])


engagement_pre = [f"Pre_SSSQ_{i}_pre" for i in [2, 5, 11, 12, 13, 17, 21, 22]]
distress_pre = [f"Pre_SSSQ_{i}_pre" for i in [1, 3, 4, 6, 7, 8, 9, 10]]
worry_pre = [f"Pre_SSSQ_{i}_pre" for i in [14, 15, 16, 18, 19, 20, 23, 24]]

#here the same is done for the post questionnaire
engagement_post = [f"Pre_SSSQ_{i}_post" for i in [2, 5, 11, 12, 13, 17, 21, 22]]
distress_post = [f"Pre_SSSQ_{i}_post" for i in [1, 3, 4, 6, 7, 8, 9, 10]]
worry_post = [f"Pre_SSSQ_{i}_post" for i in [14, 15, 16, 18, 19, 20, 23, 24]]

combined_data['engagement_pre'] = combined_data[engagement_pre].mean(axis=1)
combined_data['distress_pre'] = combined_data[distress_pre].mean(axis=1)
combined_data['worry_pre'] = combined_data[worry_pre].mean(axis=1)

#same here is done for post questionnaire
combined_data['engagement_post'] = combined_data[engagement_post].mean(axis=1)
combined_data['distress_post'] = combined_data[distress_post].mean(axis=1)
combined_data['worry_post'] = combined_data[worry_post].mean(axis=1)

#and here we have a mean of the total questionnaire per person, independent of the scales
combined_data['total_pre'] = combined_data[engagement_pre + distress_pre + worry_pre].mean(axis=1)
combined_data['total_post'] = combined_data[engagement_post + distress_post + worry_post].mean(axis=1)

# calculate differences between pre and post
combined_data['engagement_diff'] = combined_data['engagement_post'] - combined_data['engagement_pre']
combined_data['distress_diff'] = combined_data['distress_post'] - combined_data['distress_pre']
combined_data['worry_diff'] = combined_data['worry_post'] - combined_data['worry_pre']
combined_data['total_diff'] = combined_data['total_post'] - combined_data['total_pre']
combined_data['amylase_diff'] = combined_data['Amylase_post'] - combined_data['Amylase_pre']

for scale in ['engagement', 'distress', 'worry', 'total']:
    print(f"Test für {scale}")

    for group in ['cg', 'eg']:
        print(f"Analyse innerhalb der Gruppe {group.upper()}")
        group_data = combined_data[combined_data['Group_pre'] == group]

        # Variablennamen dynamisch festlegen
        amylase = "amylase_diff"
        scale_column = f"{scale}_diff"

        # Shapiro-Wilk-Test für Normalverteilung
        amylase_normal = shapiro(group_data[amylase]).pvalue >= 0.05
        scale_normal = shapiro(group_data[scale_column]).pvalue >= 0.05

        # Testentscheidung basierend auf Normalverteilung
        if amylase_normal and scale_normal:
            r, p = pearsonr(group_data[amylase], group_data[scale_column])
            test_name = "Pearson"
        else:
            r, p = spearmanr(group_data[amylase], group_data[scale_column])
            test_name = "Spearman"

        print(f"{test_name} für {scale}:")
        print(f"T-Wert: {r:.3f}, P-Wert: {p:.3f}\n")
