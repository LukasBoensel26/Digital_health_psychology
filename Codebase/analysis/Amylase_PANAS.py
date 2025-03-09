from scipy.stats import spearmanr, pearsonr, shapiro #statistical evaluation
import seaborn as sns #data visualization
import matplotlib.pyplot as plt #plotting
import pandas as pd #statistical evaluation/maths

#load both datasets
post_sssq = pd.read_excel(r'CortisolAmaylaseSSSQPANAS_post.xlsx', engine='openpyxl')
pre_sssq = pd.read_excel(r'CortisolAmaylaseSSSQPANAS_pre.xlsx', engine='openpyxl')

combined_data = pd.merge(pre_sssq, post_sssq, on="VP", suffixes=('_pre', '_post')) #merch pre and post dataset to one
combined_data = combined_data[combined_data['Group_post'] != '?'] #exclude person 7

combined_data = combined_data.dropna(subset=['v_26_post'])  # drop all people without answers for PANAS
combined_data = combined_data[combined_data['v_26_post'].astype(str).str.strip() != '']

# calculate the factors
'''
for each Pre-SSSQ_{number}_pre/post in the dataset, look which {number} it has and if it fits to 
the scale (e.g. 2 and 5 for engagement) add it to the list variable  

alternative (maybe mor comprehensible) code: 
for i in [2, 5, 11, 12, 13, 17, 21, 22]: 

    engagement_pre.append(f"Pre_SSSQ_{i}_pre")

--> loops through numbers list and when there is a match for the number in the column name, 
it will add it to the list of variable names

for PANAS: change f"Pre_SSSQ_{i}_pre" to f"v_{i}_pre" and the corresponding numbers for the scales as well
'''
positiveAffect_pre = [f"v_{i}_pre" for i in [26, 28, 29, 31, 35, 50, 52, 54, 56, 57]]
negativeAffect_pre = [f"v_{i}_pre" for i in [27, 30, 32, 33, 34, 51, 53, 55, 58, 59]]

# here the same is done for the post questionnaire
positiveAffect_post = [f"v_{i}_post" for i in [26, 28, 29, 31, 35, 50, 52, 54, 56, 57]]
negativeAffect_post = [f"v_{i}_post" for i in [27, 30, 32, 33, 34, 51, 53, 55, 58, 59]]

'''
Here we calculate the mean for each person for each scale
combined_data[engagement_pre] stands for the column in the dataset
.mean obviously calculates the mean 
and axis = 1 means, that the mean will be calculated rowwise (so per person); axis = 0 would be
a calculation columnwise so per variable :) 
'''
combined_data['positiveAffect_pre'] = combined_data[positiveAffect_pre].mean(axis=1)
combined_data['negativeAffect_pre'] = combined_data[negativeAffect_pre].mean(axis=1)

# same here is done for post questionnaire
combined_data['positiveAffect_post'] = combined_data[positiveAffect_post].mean(axis=1)
combined_data['negativeAffect_post'] = combined_data[negativeAffect_post].mean(axis=1)

# and here we have a mean of the total questionnaire per person, independent of the scales
combined_data['total_pre'] = combined_data[positiveAffect_pre + negativeAffect_pre].mean(axis=1)
combined_data['total_post'] = combined_data[positiveAffect_post + negativeAffect_post].mean(axis=1)

# calculate differences between pre and post
combined_data['positiveAffect_diff'] = combined_data['positiveAffect_post'] - combined_data['positiveAffect_pre']
combined_data['negativeAffect_diff'] = combined_data['negativeAffect_post'] - combined_data['negativeAffect_pre']
combined_data['total_diff'] = combined_data['total_post'] - combined_data['total_pre']
combined_data['amylase_diff'] = combined_data['Amylase_post'] - combined_data['Amylase_pre']

for scale in ['positiveAffect', 'negativeAffect', 'total']:
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
