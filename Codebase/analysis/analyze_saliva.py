"""
Description: the goal of this script is to statistically analyze the data of alpha-amylase and cortisol measured over four timepoints within a TSST.
                The study subjects either belong to the experimental group (--> stressful TSST) or the control group (--> friendly TSST-version)
                With this analysis we first want to find out whether the stressful TSST really causes objective stress within the study subjects measured by
                alpha-amylase and cortisol as quantitative metrics. This can be treated as a manipulation check for further analysis. 
                In addition to that we want to find out whether the stress-reaction (quantified in form of the amylase increase / max. cortisol increase) 
                significantly differs between
                1) experimental group and control group
                2) female and male within the experimental group
Author: Lukas Bönsel
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))) # for relative imports

import biopsykit as bp
from fau_colors import cmaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from paths import STRUCTURED_DATA_PATH, PLOT_PATH

SIGNIFICANCE_LVL = 0.05

# dataframe cleaning and modification
def remove_outliers_by_IQR(df, feature: str):
    """
    this method removes outliers in the specified feature of the given dataframe based on the Interquartile Range (IQR),
    i.e. the outlier value for this feature is replaced by NaN

    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data
        feature (str): feature for which to remove outliers

    Returns:
        pandas.core.frame.DataFrame: modified dataframe of structured data (with outliers being removed / replaced by NaN)
    """
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    # define lower and upper bounds for outlier detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # replace outliers by NaN
    df[feature] = df[feature].apply(lambda x: np.nan if x < lower_bound or x > upper_bound else x)
    return df

def clean_dataframe(df, features: list[str] = None):
    """
    this method removes rows in the dataframe for which the group assignment is missing and optionally
    removes outliers based on the IQR for all specified features --> see remove_outliers_by_IQR() for more details

    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data
        features (list[str], optional): list of features for which to perform outlier rejection. Default is None.

    Returns:
        pandas.core.frame.DataFrame: cleaned dataframe of structured data
    """
    df_cleaned = df[(df["Group"] == "cg") | (df["Group"] == "eg")]

    if features:
        for feature in features:
            df_cleaned = remove_outliers_by_IQR(df_cleaned, feature=feature)

    return df_cleaned

def calc_increase_amylase(df):
    """
    this method calculates the amylase increase by:
    amylase sample value after presentation (03) - amylase sample value before presentation (02)
    
    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data

    Returns:
        pandas.core.frame.DataFrame: dataframe of structured data with a new column added for the amylase increase
    """
    subjects = []
    for _, row in df.iterrows():
        vp_data = row.to_dict()
        if np.isnan(vp_data["Amylase Sample 02 (U/ml)"]) or np.isnan(vp_data["Amylase Sample 03 (U/ml)"]):
            amylase_increase = np.nan
        else:
            amylase_increase = vp_data["Amylase Sample 03 (U/ml)"] - vp_data["Amylase Sample 02 (U/ml)"] 
        vp_data["Amylase increase (U/ml)"] = amylase_increase
        subjects.append(vp_data)

    df = pd.DataFrame(subjects)
    return df

def calc_max_increase_cortisol(df):
    """
    this method calculates the maximum cortisol increase by:
    maximum cortisol sample value (considering timepoints 2,3,4) - cortisol sample value at the very beginning of the TSST (timepoint 1)
    
    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data

    Returns:
        pandas.core.frame.DataFrame: dataframe of structured data with a new column added for the maximum cortisol increase
    """
    subjects = []
    for _, row in df.iterrows():
        vp_data = row.to_dict()
        if np.isnan(vp_data["Cortisol Sample 01 (nmol/l)"]) or np.isnan(vp_data["Cortisol Sample 02 (nmol/l)"]) or np.isnan(vp_data["Cortisol Sample 03 (nmol/l)"]) or np.isnan(vp_data["Cortisol Sample 04 (nmol/l)"]):
            cortisol_max_increase = np.nan
        else:
            cortisol_max_increase = max(vp_data["Cortisol Sample 02 (nmol/l)"], vp_data["Cortisol Sample 03 (nmol/l)"], 
                                        vp_data["Cortisol Sample 04 (nmol/l)"]) - vp_data["Cortisol Sample 01 (nmol/l)"]
        vp_data["Cortisol max increase (nmol/l)"] = cortisol_max_increase
        subjects.append(vp_data)

    df = pd.DataFrame(subjects)
    return df

# plotting methods
def setup_plotting_style():
    """
    defines the general plotting style to keep consistency within the analysis plots
    --> to be extended!
    """
    sns.set_palette(cmaps.faculties_light)

def plot_distribution_as_histogram(df, feature: str):
    """
    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data
        feature (str): feature for which we want to plot the distribution as a histogram
    """
    plt.figure(figsize=(10,6))
    sns.histplot(data=df[feature], bins=10, kde=False)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {feature}")
    plt.show()

def create_box_stripplot_plot(df, feature: str, group: str, show: bool = True, save_path: str = None):
    """
    this method creates a boxplot overlayed by a stripplot comparing two groups defined by group in a certain specified feature

    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data
        feature (str): feature one is interested in
        group (str): group used for comparison
        show (bool, optional): if True, the plot is shown on the screen. Default is True.
        save_path (str, optional): If specified the plot is saved as .png at the given path. Defaults to None.
    """
    if group == "Gender":
        df = df[df["Group"] == "eg"] # extract the experimental group

    # clean dataframe group wise
    if group == "Group":
        df_1 = remove_outliers_by_IQR(df=df[df[group] == "eg"], feature=feature)
        df_2 = remove_outliers_by_IQR(df=df[df[group] == "cg"], feature=feature)
    else: # group == "Gender"
        df_1 = remove_outliers_by_IQR(df=df[df[group] == "female"], feature=feature)
        df_2 = remove_outliers_by_IQR(df=df[df[group] == "male"], feature=feature)

    # concatenate dataframes
    df = pd.concat([df_1, df_2], axis=0, ignore_index=True)

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=group, y=feature, data=df, width=0.5, showfliers=False)
    sns.stripplot(x=group, y=feature, data=df, color="black", alpha=0.8, jitter=True)  # add jitter for better visibility
    plt.title("Boxplot with overlayed samples")
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

def create_line_plot(df, feature: str, group: str, show: bool = True, save_path: str = None):
    """
    this method creates a time series lineplot for the specified feature

    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data
        feature (str): feature for which to create the lineplot. Can be one of the following: ["amylase", "cortisol"]
        group (str): group used for group comparison, can be one of the following ["Group", "Gender"]
        --> in case of group="Gender" it compares the feature statistics between "male" and "female" WITHIN the experimental group
        show (bool, optional): if True, the plot is shown on the screen. Default is True.
        save_path (str, optional): if specified: path where to save the plot as .png. Defaults to None.
    """
    if group == "Gender":
        df = df[df["Group"] == 'eg'] # just take the experimental group

    # clean dataframe group wise
    if feature == "amylase":
        features_to_clean = ["Amylase Sample 01 (U/ml)", "Amylase Sample 02 (U/ml)", "Amylase Sample 03 (U/ml)", "Amylase Sample 04 (U/ml)"]
    else: # feature == "cortisol"
        features_to_clean = ["Cortisol Sample 01 (nmol/l)", "Cortisol Sample 02 (nmol/l)", "Cortisol Sample 03 (nmol/l)", "Cortisol Sample 04 (nmol/l)"]

    if group == "Group":
        df_1 = clean_dataframe(df=df[df[group] == "eg"], features=features_to_clean)
        df_2 = clean_dataframe(df=df[df[group] == "cg"], features=features_to_clean)
    else: # group == "Gender"
        df_1 = clean_dataframe(df=df[df[group] == "male"], features=features_to_clean)
        df_2 = clean_dataframe(df=df[df[group] == "female"], features=features_to_clean)

    # concatenate dataframes
    df = pd.concat([df_1, df_2], axis=0, ignore_index=True)

    # extract relevant features from the given dataframe
    if feature == "amylase":
        columns_to_extract = ["VP", group, "Amylase Sample 01 (U/ml)", "Amylase Sample 02 (U/ml)", "Amylase Sample 03 (U/ml)", "Amylase Sample 04 (U/ml)"]
    else: # cortisol
        columns_to_extract = ["VP", group, "Cortisol Sample 01 (nmol/l)", "Cortisol Sample 02 (nmol/l)", "Cortisol Sample 03 (nmol/l)", "Cortisol Sample 04 (nmol/l)"]

    df_extracted = df[columns_to_extract]
    # rename columns
    df_extracted.columns = ["VP", group, "01", "02", "03", "04"]
    # convert dataframe into long format
    if feature == "amylase":
        value_name = "Amylase (U/ml)"
    else: # cortisol
        value_name = "Cortisol (nmol/l)"
    df_extracted_long = df_extracted.melt(id_vars=["VP", group], var_name="sample", value_name=value_name)

    _, ax = plt.subplots(figsize=(8,6))
    bp.plotting.lineplot(data=df_extracted_long, x="sample", y=value_name, hue=group, ax=ax)
    plt.title(f"Time series lineplot with standard deviation for {feature}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

# helpers for statistics
def check_for_normality(distribution: pd.core.series.Series, qq_plot: bool = False):
    """
    Args:
        distribution (pd.core.series.Series): distribution for which to check for normality
        qq_plot (bool, optional): If True, a q-q-plot of the given distribution is plotted. Default is False.

    Returns:
        bool: True if normally distributed, else False
    """
    from scipy.stats import shapiro
    # --> H0: the distribution follows a normal distribution

    if qq_plot:
        fig, ax = plt.subplots(figsize=(8,6))
        stats.probplot(distribution, dist="norm", plot=ax)
        plt.title("Q-Q plot for the given distribution")
        plt.show()

    stat, p_value = shapiro(distribution)
    if p_value < SIGNIFICANCE_LVL:
        # reject H0
        print(f"p-value: {p_value}\nThe distribution does not follow a normal distribution!")
        return False
    else:
       # failed to reject H0
       print(f"p-value: {p_value}\nThe distribution follows a normal distribution!")
       return True

def check_for_variance_homogeneity(distribution1: pd.core.series.Series, distribution2: pd.core.series.Series):
    """
    Args:
        distribution1 (pd.core.series.Series): distribution for which to check for variance homogeneity with the other distribution
        distribution2 (pd.core.series.Series): distribution for which to check for variance homogeneity with the other distribution

    Returns:
        bool: True if variance homogeneity is given, else False
    """
    from scipy.stats import levene
    # --> H0: variances are the same over the groups

    stat, p_value = levene(distribution1, distribution2)
    if p_value < SIGNIFICANCE_LVL:
        # reject H0
        print(f"p-value: {p_value}\nThe variances are significantly different between the two distributions!")
        return False
    else:
        # failed to reject H0
        print(f"p-value: {p_value}\nThe variances are NOT significantly different and thus homogeneous between the two distributions!")
        return True

def prove_t_test_conditions(distribution1: pd.core.series.Series, distribution2: pd.core.series.Series, group: str):
    """
    proves the conditions that are necessary to be fullfilled for a t-test

    Args:
        distribution1 (pd.core.series.Series): distribution for which to check for normality
        distribution2 (pd.core.series.Series): distribution for which to check for normality
        --> additionally check for variance homogeneity in between the two distributions
        group (str): group in which the two given distributions differ

    Returns:
        tuple: boolean values for the different checks of normality_1, normality_2, variance_homogeneity
    """
    # 1) check for normality
    print(f"Normality check for {group}:")
    normality_1 = check_for_normality(distribution=distribution1)
    print("Distribution 1 is normally distributed: ", normality_1)
    normality_2 = check_for_normality(distribution=distribution2)
    print("Distribution 2 is normally distributed: ", normality_2)

    # 2) check for variance homogeneity
    print(f"Variance homogeneity check for {group}:")
    variance_homogeneity = check_for_variance_homogeneity(distribution1=distribution1, distribution2=distribution2)
    print("There is a variance homogeneity in between the distributions: ", variance_homogeneity)

    return normality_1, normality_2, variance_homogeneity

# statistical tests
def compare_means_by_t_test(df, feature: str, group: str):
    """
    this method performs an independent samples t-test on the specified feature

    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data
        feature (str): feature for which to perform a t-test on
        group (str): group used for comparison, can be one of the following ["Group", "Gender"]
        --> in case of group="Gender" it compares the feature statistics between "male" and "female" WITHIN the experimental group
    
    Returns:
        tuple: t_stat, p_value
    """
    # null hypothesis (H0): There is no significant difference in the means of the two independent groups being compared
    
    if group == "Group":
        group_1 = df[df[group] == 'eg'][feature].dropna() # missing values are dropped
        group_2 = df[df[group] == 'cg'][feature].dropna()
    else: # group == "Gender"
        df = df[df["Group"] == 'eg'] # just take the experimental group
        group_1 = df[df[group] == 'male'][feature].dropna()
        group_2 = df[df[group] == 'female'][feature].dropna()

    # remove outliers in both groups separately 
    group_1 = remove_outliers_by_IQR(df=pd.DataFrame(group_1), feature=feature)[feature].dropna()
    group_2 = remove_outliers_by_IQR(df=pd.DataFrame(group_2), feature=feature)[feature].dropna()
    # print size of groups
    print(f"Group 1 size = {group_1.size}")
    print(f"Group 2 size = {group_2.size}")

    # test conditions of t-test before proceed
    normality_1, normality_2, variance_homogeneity = prove_t_test_conditions(distribution1=group_1, distribution2=group_2, group=group)
    if all((normality_1, normality_2, variance_homogeneity)):
        print("All t-test conditions are fullfilled! You can proceed!")
        # perform the t-test
        t_stat, p_value = stats.ttest_ind(group_1, group_2, equal_var=True)  # Set equal_var=False if variances are unequal
    elif (normality_1, normality_2) and not variance_homogeneity:
        print("Variance homogeneity is not given! You can though perform the t-test with equal_var=False!")
        # perform the t-test with equal_var=False
        t_stat, p_value = stats.ttest_ind(group_1, group_2, equal_var=False)
    else:
        print("Normality is not given for at least one group! Transform your data first!")
        t_stat, p_value = np.nan, np.nan

    return t_stat, p_value

def check_for_significant_change_in_feature(df, feature: str):
    """
    this method performs a paired t-test (also known as a dependent samples t-test) to compare levels of the specified feature
    before and after the TSST within the same subjects within the experimental group

    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data
        feature (str): feature for which to perform a paired t-test on. Can be one of the following ["amylase" or "cortisol"]

    Returns:
        tuple: t_stat, p_value
    """
    # H0: There is no significant difference in the features levels before and after the TSST

    # get experimental group
    df = df[df["Group"] == 'eg']
    if feature == "amylase":
        # take 2nd and 3rd timepoint
        pre_feature = "Amylase Sample 02 (U/ml)"
        post_feature = "Amylase Sample 03 (U/ml)"
    else: # cortisol
        # take 2nd and 4th timepoint
        pre_feature = "Cortisol Sample 02 (nmol/l)"
        post_feature = "Cortisol Sample 04 (nmol/l)"

    # perform pair-wise deletion of study subjects if either pre_feature or post_feature is missing
    df = df.dropna(subset=[pre_feature, post_feature])
    pre = df[pre_feature]
    post = df[post_feature]

    # check for normality of differences
    diff = post - pre
    normality = check_for_normality(distribution=diff, qq_plot=False)
    if not normality:
        print("The differences are not normally distributed! But we can though use the paired t-test!")
    
    # perform a pairwise t-test
    t_stat, p_value = stats.ttest_rel(pre, post)

    return t_stat, p_value

def compare_groups_by_whitney_u_test(df, feature: str, group: str, plot_distribution: bool = False):
    """
    this method performs a Wilcoxon-Mann-Whitney-Test (non-parametric) on the specified feature between the groups specified by group

    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data
        feature (str): feature for which to perform the test on
        group (str): group used for group comparison, can be one of the following ["Group", "Gender"]
        --> in case of group="Gender" it compares the feature statistics between "male" and "female" WITHIN the experimental group
        plot_distribution (bool): Defines whether the feature distribution over groups should be plotted or not. Default is False.
    
    Returns:
        tuple: stat, p_value
    """
    # null hypothesis (H0): the distributions of the two independent groups being compared are similar:
    # the probability for a randomly chosen sample from one population being larger or smaller compared to a randomly chosen sample
    # from the other population is the same
    # --> population from which each sample is drawn has the same shape, central tendency, and spread (e.g., medians, distribution form)
    from scipy.stats import mannwhitneyu

    if group == "Group":
        group_1 = df[df[group] == 'eg'][feature].dropna() # missing values are dropped
        group_2 = df[df[group] == 'cg'][feature].dropna()
    else: # group == "Gender"
        df = df[df["Group"] == 'eg'] # just take the experimental group
        group_1 = df[df[group] == 'male'][feature].dropna()
        group_2 = df[df[group] == 'female'][feature].dropna()

    # remove outliers in both groups separately 
    group_1 = remove_outliers_by_IQR(df=pd.DataFrame(group_1), feature=feature)[feature].dropna()
    group_2 = remove_outliers_by_IQR(df=pd.DataFrame(group_2), feature=feature)[feature].dropna()
    # print size of groups
    print(f"Group 1 size = {group_1.size}")
    print(f"Group 2 size = {group_2.size}")

    if plot_distribution:
        # plot histograms to get an overview of the distributions
        if group == "Group":
            num_bins = 5
        else:
            num_bins = 3
        plt.figure(figsize=(10,6))
        plt.hist(group_1, bins=num_bins, alpha=0.5, label="eg" if group == "Group" else "male", color='blue')
        plt.hist(group_2, bins=num_bins, alpha=0.5, label="cg" if group == "Group" else "female", color='orange')
        plt.legend()
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {feature} over {"condition (eg, cg)" if group=="Group" else "gender within the experimental group (eg)"}")
        plt.show()

    stat, p_value = mannwhitneyu(group_1, group_2)
    return stat, p_value


def main():
    setup_plotting_style()

    df = pd.read_excel(STRUCTURED_DATA_PATH)
    # remove study subjects for which the group assignment is missing
    df = clean_dataframe(df)

    # check for study subjects with significantly high cortisol level at time of arrival
    plot_distribution_as_histogram(df, feature='Cortisol Sample 01 (nmol/l)')
    # --> no outlier

    # for the paired t-test we analyse "Amylase Sample 02 (U/ml)", "Amylase Sample 03 (U/ml)",
    # "Cortisol Sample 02 (nmol/l)", "Cortisol Sample 04 (nmol/l)" withinin the experimental group
    # --> check for outliers and replace by NaN
    features_to_clean = ["Amylase Sample 02 (U/ml)", "Amylase Sample 03 (U/ml)",
                         "Cortisol Sample 02 (nmol/l)", "Cortisol Sample 04 (nmol/l)"]
    # extract only the experimental group from the dataframe
    df_eg = df[df["Group"] == "eg"]
    df_eg_cleaned = clean_dataframe(df_eg, features=features_to_clean)

    # perform a pairwise t-test to test whether the TSST causes stress in the experimental group
    # --> this is a manipulation check for the successive statistical tests
    features_paired_ttest = ["amylase", "cortisol"]
    for feature in features_paired_ttest:
        print(f"\nPerforming a paired t-test for the feature {feature} within the experimental group")
        t_stat, p_value = check_for_significant_change_in_feature(df=df_eg_cleaned, feature=feature)
        if not np.isnan(t_stat) and not np.isnan(p_value):
            print(f"Feature: {feature}\np-value: {p_value}\nt-stat: {t_stat}")
            if p_value < SIGNIFICANCE_LVL:
                # reject the null hypothesis
                print(f"There is a significant difference in the {feature} levels before and after the TSST!")
            else:
                # failed to reject the null hypothesis
                print(f"There is NO significant difference in the {feature} levels before and after the TSST!")
    
    # in order to calculate the amylase increase and the max. cortisol increase we need these features
    features_to_clean = ["Amylase Sample 02 (U/ml)", "Amylase Sample 03 (U/ml)",
                         "Cortisol Sample 01 (nmol/l)", "Cortisol Sample 02 (nmol/l)",
                         "Cortisol Sample 03 (nmol/l)", "Cortisol Sample 04 (nmol/l)"]
    # --> perform outlier rejection on total dataframe df (including cg and eg)
    df_total_cleaned = clean_dataframe(df, features=features_to_clean)
    
    # calculate new features
    df_total_cleaned = calc_increase_amylase(df=df_total_cleaned)
    df_total_cleaned = calc_max_increase_cortisol(df=df_total_cleaned)

    # try to perform a t-test in which requirement conditions are checked
    test_groups = ["Group", "Gender"]
    features_to_test = ["Amylase increase (U/ml)", "Cortisol max increase (nmol/l)"]
    for group in test_groups:
        for feature in features_to_test:
            print(f"\nTrying to perform a t-test for the feature {feature} distinguishing between {group}")
            t_stat, p_value = compare_means_by_t_test(df=df_total_cleaned, feature=feature, group=group)
            if not np.isnan(t_stat) and not np.isnan(p_value):
                print(f"Group: {group}\nFeature: {feature}\np-value: {p_value}\nt-stat: {t_stat}")
                if p_value < SIGNIFICANCE_LVL:
                    # reject the null hypothesis
                    print(f"There is a significant difference in the means of {feature} between the two independent groups which differ in {"condition (eg, cg)" if group=="Group" else "gender within the experimental group (eg)"}")
                else:
                    # failed to reject the null hypothesis
                    print(f"There is NO significant difference in the means of {feature} between the two independent groups which differ in {"condition (eg, cg)" if group=="Group" else "gender within the experimental group (eg)"}")

    # as the t-test failed for Cortisol with Group Gender use another test which does not require normality
    # perform a Mann-Whitney U test --> non-parametric counterpart to t-test
    features_to_test = ["Cortisol max increase (nmol/l)"]
    test_groups = ["Gender"]
    for group in test_groups:
        for feature in features_to_test:
            print(f"\nPerforming a Wilcoxon-Mann-Whitney-test for the feature {feature} distinguishing between {group}")
            stat, p_value = compare_groups_by_whitney_u_test(df=df_total_cleaned, feature=feature, group=group, plot_distribution=True)
            print(f"Group: {group}\nFeature: {feature}\np-value: {p_value}\nstat: {stat}")
            if p_value < SIGNIFICANCE_LVL:
                # reject the null hypothesis
                print(f"There is a significant difference in the central tendencies (such as medians) or distributions of {feature} between the two independent groups which differ in {"condition (eg, cg)" if group=="Group" else "gender within the experimental group (eg)"}")
            else:
                # failed to reject the null hypothesis
                print(f"There is NO significant difference in the central tendencies (such as medians) or distributions of {feature} between the two independent groups which differ in {"condition (eg, cg)" if group=="Group" else "gender within the experimental group (eg)"}")
        
    # plotting
    # additionally clean for the features "Amylase Sample 01 (U/ml)", "Amylase Sample 04 (U/ml)" for which we have not cleaned yet in df_total_cleaned
    features_to_clean = ["Amylase Sample 01 (U/ml)", "Amylase Sample 04 (U/ml)"]
    df_total_cleaned = clean_dataframe(df_total_cleaned, features=features_to_clean)
    # --> no outlier detection within the groups but on the total dataframe (including cg and eg)

    # create line plot for amylase and cortisol
    # --> with group distinction
    create_line_plot(df_total_cleaned, feature="amylase", group="Group", show=False, save_path=os.path.join(PLOT_PATH, "amylase_group_lineplot.png"))
    create_line_plot(df_total_cleaned, feature="cortisol", group="Group", show=False, save_path=os.path.join(PLOT_PATH, "cortisiol_group_lineplot.png"))
    # --> with gender distinction within experimental group
    create_line_plot(df_total_cleaned, feature="amylase", group="Gender", show=False, save_path=os.path.join(PLOT_PATH, "amylase_gender_lineplot.png"))
    create_line_plot(df_total_cleaned, feature="cortisol", group="Gender", show=False, save_path=os.path.join(PLOT_PATH, "cortisiol_gender_lineplot.png"))

    # create box plots to get a better feeling for the distributions of amylase increase and max. cortisol increase
    # --> with group distinction
    create_box_stripplot_plot(df_total_cleaned, feature="Amylase increase (U/ml)", group="Group", show=False, save_path=os.path.join(PLOT_PATH, "amylase_increase_group_boxplot.png"))
    create_box_stripplot_plot(df_total_cleaned, feature="Cortisol max increase (nmol/l)", group="Group", show=False, save_path=os.path.join(PLOT_PATH, "cortisol_increase_group_boxplot.png"))
    # --> with gender distinction within experimental group
    create_box_stripplot_plot(df_total_cleaned, feature="Amylase increase (U/ml)", group="Gender", show=False, save_path=os.path.join(PLOT_PATH, "amylase_increase_gender_boxplot.png"))
    create_box_stripplot_plot(df_total_cleaned, feature="Cortisol max increase (nmol/l)", group="Gender", show=False, save_path=os.path.join(PLOT_PATH, "cortisol_increase_gender_boxplot.png"))

    df.to_excel(os.path.join(os.path.dirname(STRUCTURED_DATA_PATH), "structured_data_initially_cleaned.xlsx"))
    df_total_cleaned.to_excel(os.path.join(os.path.dirname(STRUCTURED_DATA_PATH), "structured_data_with_features_cleaned.xlsx"))
    sys.exit(0)

if __name__ == "__main__":
    main()