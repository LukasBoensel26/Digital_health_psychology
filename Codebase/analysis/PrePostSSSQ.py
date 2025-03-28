from scipy.stats import ttest_rel, wilcoxon, shapiro, ttest_ind, mannwhitneyu #statistical evaluation
import seaborn as sns #data visualization
import matplotlib.pyplot as plt #plotting
import pandas as pd #statistical evaluation/maths

#load both datasets
pre_sssq = pd.read_csv("pre_sssq.csv")
post_sssq = pd.read_csv("post_sssq.csv")

#add missing group info to dataset
groups = [
    'cg', 'eg', 'eg', 'cg', 'eg', 'cg', '?', 'cg', 'cg', 'eg', 'cg',
    'eg', 'cg', 'eg', 'cg', 'eg', 'cg', 'eg', 'cg', 'eg', 'cg',
    'eg', 'cg', 'eg', 'cg', 'eg', 'cg', 'eg', 'cg', 'eg', 'cg'
]

combined_data = pd.merge(pre_sssq, post_sssq, on="VPN_Kennung", suffixes=('_pre', '_post')) #merch pre and post dataset to one
combined_data['Group'] = groups #add the group info
combined_data = combined_data[combined_data['Group'] != '?'] #exclude person 7


# Reverse-Coding-Funktion (1 wird zu 5, 2 zu 4, usw.)
def reverse_code(series):
    return 6 - series


#Reverse Coding
reverse_coded_items = [2, 11, 13, 17, 21, 22]

for i in reverse_coded_items:
    pre_col = f"Pre_SSSQ_{i}_pre"
    post_col = f"Pre_SSSQ_{i}_post"

    if pre_col in combined_data.columns:
        combined_data[pre_col] = reverse_code(combined_data[pre_col])

    if post_col in combined_data.columns:
        combined_data[post_col] = reverse_code(combined_data[post_col])

print("Reverse-Coding durchgeführt für:", reverse_coded_items)

# calculate the factors
'''
for each Pre-SSSQ_{number}_pre/post in the dataset, look which {number} it has and if it fits to 
the scale (e.g. 2 and 5 for engagement) add it to the list variable  

alternative (maybe mor comprehensible) code: 
for i in [2, 5, 11, 12, 13, 17, 21, 22]: 

    engagement_pre.append(f"Pre_SSSQ_{i}_pre")
    
--> loops through numbers list and when there is a match for the number in the column name, 
it will add it to the list of variable names

for PANAS: change f"Pre_SSSQ_{i}_pre" to f"v_{i}" and the corresponding numbers for the scales as well
'''
engagement_pre = [f"Pre_SSSQ_{i}_pre" for i in [2, 5, 11, 12, 13, 17, 21, 22]]
distress_pre = [f"Pre_SSSQ_{i}_pre" for i in [1, 3, 4, 6, 7, 8, 9, 10]]
worry_pre = [f"Pre_SSSQ_{i}_pre" for i in [14, 15, 16, 18, 19, 20, 23, 24]]

#here the same is done for the post questionnaire
engagement_post = [f"Pre_SSSQ_{i}_post" for i in [2, 5, 11, 12, 13, 17, 21, 22]]
distress_post = [f"Pre_SSSQ_{i}_post" for i in [1, 3, 4, 6, 7, 8, 9, 10]]
worry_post = [f"Pre_SSSQ_{i}_post" for i in [14, 15, 16, 18, 19, 20, 23, 24]]

'''
Here we calculate the mean for each person for each scale
combined_data[engagement_pre] stands for the column in the dataset
.mean obviously calculates the mean 
and axis = 1 means, that the mean will be calculated rowwise (so per person); axis = 0 would be
a calculation columnwise so per variable :) 
'''
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

#for each scale and total:
for scale in ['engagement', 'distress', 'worry', 'total']:
    print(f"Test für {scale}")

    for group in ['cg', 'eg']: #here we get one loop for condition CG and one for EG
        print(f"Analyse innerhalb der Gruppe {group.upper()}")
        group_data = combined_data[combined_data['Group'] == group] # get the group that is defined in the for loop already

        '''
        here we chose the right scales, e.g. engagement_pre; scale is defined in the outer for loop 
        e.g. for scale in engagement, which means scale would get the value engagement 
        '''
        pre_col = f"{scale}_pre"
        post_col = f"{scale}_post"

        #here we look for the already calculated diff scale in the dataset for the normal distribution test
        diff = combined_data[f"{scale}_diff"]

        # Shapiro-Wilk-Test for normal distribution
        stat, p_value = shapiro(diff) #here the test is calculated
        print(f"Shapiro-Wilk-Test für {scale} (Gruppe {group.upper()}):") #this is just output of our results
        print(f"T-Wert: {stat:.3f}, P-Wert: {p_value:.3f}")
        normal_distribution = p_value >= 0.05 #and this means: if p is bigger than 0.05, normal_distribution will get the value TRUE

        # test decision based on normal distribution
        if normal_distribution:
            t_stat, p_value = ttest_rel(group_data[pre_col], group_data[post_col]) #in case of normal distribution we calculate a pairwise t test
            test_name = "t-Test (paired)"
        else:
            t_stat, p_value = wilcoxon(group_data[pre_col], group_data[post_col]) #and in case of no n.d. we calculate the non-parametric test
            test_name = "Wilcoxon-Test (paired)"

            #test for direction
            w_inc, p_inc = wilcoxon(group_data[pre_col], group_data[post_col],
                                    alternative='greater')  # Test für Anstieg
            w_dec, p_dec = wilcoxon(group_data[pre_col], group_data[post_col], alternative='less')  # Test für Abnahme

            if p_inc < 0.05:
                direction = "Anstieg"
            elif p_dec < 0.05:
                direction = "Abnahme"
            else:
                direction = "Keine signifikante Richtung"
            print(direction)

        print(f"{test_name} für {scale} (Gruppe {group.upper()}):") #here again just output
        print(f"T-Wert: {t_stat:.3f}, P-Wert: {p_value:.3f}\n") #and here the t-value is given with 3 decimals, the p value as well

        if p_value < 0.05:
            print(f"Signifikante Unterschiede für {scale} (p < 0.05).\n")
        else:
            print(f"Keine signifikanten Unterschiede für {scale} (p > 0.05).\n")

'''
and now we look at the group differences 
the for loop does the same as the other one
'''
for scale in ['engagement_diff', 'distress_diff', 'worry_diff', 'total_diff']:
    print(f"Vergleich zwischen Gruppen für {scale}")

    # here we get the two groups based on the scale, e.g. control group of the engagement scale
    cg_data = combined_data[combined_data['Group'] == 'cg'][scale]
    eg_data = combined_data[combined_data['Group'] == 'eg'][scale]

    # here both groups are tested for normal distribution
    cg_normal = shapiro(cg_data).pvalue >= 0.05
    eg_normal = shapiro(eg_data).pvalue >= 0.05

    # and if both tests are normally distributed we can calculate either the t-test or man-whitney-u
    if cg_normal and eg_normal:
        t_stat, p_value = ttest_ind(cg_data, eg_data)
        test_name = "t-Test"
    else:
        t_stat, p_value = mannwhitneyu(cg_data, eg_data)
        test_name = "Mann-Whitney-U-Test"

    # here again the results are printed
    print(f"{test_name} für {scale}:")
    print(f"T-Wert: {t_stat:.3f}, P-Wert: {p_value:.3f}")

    if p_value < 0.05:
        print(f"Signifikante Unterschiede zwischen CG und EG für {scale} (p < 0.05).\n")
    else:
        print(f"Keine signifikanten Unterschiede zwischen CG und EG für {scale} (p > 0.05).\n")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn style
sns.set_style("whitegrid")

# Farben für bessere Unterscheidung
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# 1. Plot: Pre/Post Differences in the Experimental Group as Line Plot with Standard Deviation
eg_data = combined_data[combined_data['Group'] == 'eg']

fig, ax = plt.subplots(figsize=(8, 6))
scales = ['engagement', 'distress', 'worry', 'total']

time_points = ['Pre', 'Post']
means = {scale: [eg_data[f"{scale}_pre"].mean(), eg_data[f"{scale}_post"].mean()] for scale in scales}
sds = {scale: [eg_data[f"{scale}_pre"].std(), eg_data[f"{scale}_post"].std()] for scale in scales}

x_labels = np.arange(len(time_points))
for i, scale in enumerate(scales):
    ax.plot(x_labels, means[scale], marker='o', linestyle='-', label=scale, color=colors[i])
    ax.errorbar(x_labels, means[scale], yerr=sds[scale], fmt='o', color=colors[i], capsize=5)

ax.set_xticks(x_labels)
ax.set_xticklabels(time_points)
ax.set_ylabel("Mean ± SD")
ax.set_title("Pre/Post Differences in the Experimental Group")
ax.legend()
plt.show()

# 2. Multiple Plots for Differences between CG and EG
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, scale in enumerate(['engagement_diff', 'distress_diff', 'worry_diff', 'total_diff']):
    sns.boxplot(data=combined_data, x='Group', y=scale, ax=axes[i], hue='Group', palette=colors[:2], legend=False, showfliers=False)
    axes[i].set_title(f"Comparison CG vs. EG for {scale.replace('_diff', '')}")
    axes[i].set_ylabel("Difference")
    axes[i].set_xlabel("Group")

plt.tight_layout()
plt.show()
