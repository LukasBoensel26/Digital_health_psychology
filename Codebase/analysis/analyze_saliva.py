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

def setup_plotting_style():
    """
    defines the general plotting style to keep consistency within the analysis plots
    --> to be extended!
    """

    sns.set_palette(cmaps.faculties_light)

def remove_outliers_by_IQR(df, feature: str):
    """
    this method REMOVES outliers in the specified feature of the given dataframe based on the Interquartile Range (IQR),
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

def clean_dataframe(df, features: list[str]):
    """
    this method removes rows in the dataframe for which the group assignment is missing and 
    removes outliers based on the IQR for all specified features --> see remove_outliers_by_IQR() for more details

    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data
        features (list[str]): list of features for which to perform outlier rejection

    Returns:
        pandas.core.frame.DataFrame: cleaned dataframe of structured data
    """
    df_cleaned = df[(df["Group"] == "cg") | (df["Group"] == "eg")]

    for feature in features:
        df_cleaned = remove_outliers_by_IQR(df_cleaned, feature=feature)

    return df_cleaned

def calc_increase_amylase(df):
    """
    this method calculates the amylase increase by:
    amylase sample value after presentation - amylase sample value before presentation
    
    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data

    Returns:
        pandas.core.frame.DataFrame: dataframe of structured data with a new column added for the amylase increase
    """
    subjects = []
    for _, row in df.iterrows():
        vp_data = row.to_dict()
        if vp_data["Amylase Sample 02 (U/ml)"] and vp_data["Amylase Sample 03 (U/ml)"]:
            amylase_increase = vp_data["Amylase Sample 03 (U/ml)"] - vp_data["Amylase Sample 02 (U/ml)"]
        else:
            amylase_increase = np.nan
        vp_data["Amylase increase (U/ml)"] = amylase_increase
        subjects.append(vp_data)

    df = pd.DataFrame(subjects)
    return df

def calc_max_increase_cortisol(df):
    """
    this method calculates the maximum cortisol increase by:
    maximum cortisol sample value (considering timepoints 2,3,4) - cortisol sample value at the very beginning of the TSST
    
    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data

    Returns:
        pandas.core.frame.DataFrame: dataframe of structured data with a new column added for the maximum cortisol increase
    """
    subjects = []
    for _, row in df.iterrows():
        vp_data = row.to_dict()
        if vp_data["Cortisol Sample 01 (nmol/l)"] and vp_data["Cortisol Sample 02 (nmol/l)"] and vp_data["Cortisol Sample 03 (nmol/l)"] and vp_data["Cortisol Sample 04 (nmol/l)"]:
            cortisol_max_increase = max(vp_data["Cortisol Sample 02 (nmol/l)"], vp_data["Cortisol Sample 03 (nmol/l)"], 
                                        vp_data["Cortisol Sample 04 (nmol/l)"]) - vp_data["Cortisol Sample 01 (nmol/l)"]
        else:
            cortisol_max_increase = np.nan
        vp_data["Cortisol max increase (nmol/l)"] = cortisol_max_increase
        subjects.append(vp_data)

    df = pd.DataFrame(subjects)
    return df

def create_box_stripplot_plot(df, feature: str, hue: str = None, show: bool = True, save_path: str = None):
    """
    this method creates a boxplot overlayed by a stripplot comparing experimental group and control group in a certain specified feature

    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data
        feature (str): feature one is interested in
        hue (str, optional): If specified the plot distinguishes between different groups or categories in your data. Defaults to None.
        show (bool, optional): if True, the plot is shown on the screen. Defaults is True.
        save_path (str, optional): If specified the plot is saved as .png at the given path. Defaults to None.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Group', y=feature, data=df, hue=hue, width=0.5)
    sns.stripplot(x='Group', y=feature, data=df, color="black", alpha=0.8, jitter=True)  # Jitter for better visibility
    plt.title('Boxplot with overlayed samples')
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

def create_line_plot(df, feature: str, show: bool = True, save_path: str = None):
    """
    this method creates a time series lineplot for the specified feature

    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data
        feature (str): feature for which to create the lineplot
        show (bool, optional): if True, the plot is shown on the screen. Defaults is True.
        save_path (str, optional): if specified: path where to save the plot as .png. Defaults to None.
    """
    # extract relevant features from the passed dataframe
    if feature == "amylase":
        columns_to_extract = ["VP", "Group", "Amylase Sample 01 (U/ml)", "Amylase Sample 02 (U/ml)", "Amylase Sample 03 (U/ml)", "Amylase Sample 04 (U/ml)"]
    else: # cortisol
        columns_to_extract = ["VP", "Group", "Cortisol Sample 01 (nmol/l)", "Cortisol Sample 02 (nmol/l)", "Cortisol Sample 03 (nmol/l)", "Cortisol Sample 04 (nmol/l)"]

    df_extracted = df[columns_to_extract]
    # rename columns
    df_extracted.columns = ["VP", "Group", "01", "02", "03", "04"]
    # convert dataframe into long format
    if feature == "amylase":
        value_name = "Amylase (U/ml)"
    else: # cortisol
        value_name = "Cortisol (nmol/l)"
    df_extracted_long = df_extracted.melt(id_vars=["VP", "Group"], var_name="sample", value_name=value_name)

    _, ax = plt.subplots(figsize=(8,6))
    bp.plotting.lineplot(data=df_extracted_long, x="sample", y=value_name, hue="Group", ax=ax)
    plt.title(f"Time series lineplot with standard deviation for {feature}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

def check_for_normality(df, feature: str):
    from scipy.stats import shapiro

    group_stressful = df[df['Group'] == 'eg'][feature].dropna()
    group_friendly = df[df['Group'] == 'cg'][feature].dropna()

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    stats.probplot(group_stressful, dist="norm", plot=ax1)
    stats.probplot(group_friendly, dist="norm", plot=ax2)
    ax1.set_title("Q-Q plot for the experimental group (eg)")
    ax2.set_title("Q-Q plot for the control group (cg)")
    plt.show()

    stat_eg, p_value_eg = shapiro(group_stressful)
    stat_cg, p_value_cg = shapiro(group_friendly)
    print("eg: ", p_value_eg)
    print("cg: ", p_value_cg)
    if p_value_eg < 0.05:
        print(f"{feature} does not follow a normal distribution for the experimental group!")
    else:
       print(f"{feature} follows a normal distribution for the experimental group!")
    if p_value_cg < 0.05:
        print(f"{feature} does not follow a normal distribution for the control group!")
    else:
       print(f"{feature} follows a normal distribution for the control group!") 

def check_variance_homogeneity(df, feature: str):
    from scipy.stats import levene

    group_stressful = df[df['Group'] == 'eg'][feature].dropna()
    group_friendly = df[df['Group'] == 'cg'][feature].dropna()

    stat, p_value = levene(group_stressful, group_friendly)
    print(p_value)
    if p_value < 0.05:
        print(f"For {feature} the variances are significantly different between experimental group and control group!")
    else:
       print(f"For {feature} the variances are homogeneous between experimental group and control group!")

def compare_means(df, feature: str):
    """
    this method performs a t-test on the specified feature

    Args:
        df (pandas.core.frame.DataFrame): dataframe of structured data
        feature (str): feature for which to perform a t-test on
    
    Returns:
        tuple: t_stat, p_value
    """
    # null hypothesis (H0): There is no significant difference in the mean feature increase between the two groups.
    # --> compare means
    # --> perform t-test

    # separate the data into two groups and remove missing values --> TODO: maybe implement other strategy to deal with missing values
    group_stressful = df[df['Group'] == 'eg'][feature].dropna()
    group_friendly = df[df['Group'] == 'cg'][feature].dropna()

    # perform the t-test
    t_stat, p_value = stats.ttest_ind(group_stressful, group_friendly, equal_var=True)  # Set equal_var=False if variances are unequal

    return t_stat, p_value
    
def main():
    setup_plotting_style()
    significance_lvl = 0.05

    df = pd.read_excel(STRUCTURED_DATA_PATH)
    df = calc_increase_amylase(df)
    df = calc_max_increase_cortisol(df)

    # clean dataframe
    features_to_clean = ["Amylase increase (U/ml)", "Cortisol max increase (nmol/l)"]
    df_cleaned = clean_dataframe(df, features=features_to_clean)

    # 1) check for normality in the respective features
    print("Normality check:")
    check_for_normality(df=df_cleaned, feature="Amylase increase (U/ml)")

    # log transform with shift to try to get normality --> just relevant for cortisol
    cortisol = df_cleaned["Cortisol max increase (nmol/l)"]
    cortisol_log = np.log(cortisol - cortisol.min() + 1)
    df_cleaned["Cortisol max increase (nmol/l)"] = cortisol_log
    check_for_normality(df=df_cleaned, feature="Cortisol max increase (nmol/l)")

    # 2) check for variance homogeneity in the respective features
    print("Variance homogeneity check:")
    check_variance_homogeneity(df=df_cleaned, feature="Amylase increase (U/ml)")
    check_variance_homogeneity(df=df_cleaned, feature="Cortisol max increase (nmol/l)")

    # independence check --> DONE
    # normality check --> DONE
    # variance homogeneity check --> DONE
    # --> perform t-test

    # plotting
    # before cleaning
    # create_box_stripplot_plot(df, feature="Amylase increase (U/ml)", hue="Gender", show=False)
    # create_box_stripplot_plot(df, feature="Cortisol max increase (nmol/l)", hue="Gender", show=False)
    # # after cleaning
    # create_box_stripplot_plot(df_cleaned, feature="Amylase increase (U/ml)", hue="Gender", show=False, save_path=os.path.join(PLOT_PATH, "amylase_boxplot.png"))
    # create_box_stripplot_plot(df_cleaned, feature="Cortisol max increase (nmol/l)", hue="Gender", show=False, save_path=os.path.join(PLOT_PATH, "cortisol_boxplot.png"))

    # # create line plot for amylase and cortisol
    # create_line_plot(df_cleaned, feature="amylase", show=False, save_path=os.path.join(PLOT_PATH, "amylase_lineplot.png"))
    # create_line_plot(df_cleaned, feature="cortisol", show=False, save_path=os.path.join(PLOT_PATH, "cortisol_lineplot.png"))

    # perform the statistical tests
    _, p_value = compare_means(df_cleaned, feature="Amylase increase (U/ml)")
    print(f"Amylase p-value: {p_value}")
    if p_value < significance_lvl:
        # reject the null hypothesis
        print(f"There is a significant difference in the mean amylase increase between the two groups")
    else:
        # failed to reject the null hypothesis
        print(f"There is no significant difference in the mean amylase increase between the two groups")

    _, p_value = compare_means(df_cleaned, feature="Cortisol max increase (nmol/l)")
    print(f"Cortisol p-value: {p_value}")
    if p_value < significance_lvl:
        # reject the null hypothesis
        print(f"There is a significant difference in the mean maximum cortisol increase between the two groups")
    else:
        # failed to reject the null hypothesis
        print(f"There is no significant difference in the mean maximum cortisol increase between the two groups")

    # df.to_excel(os.path.join(os.path.dirname(STRUCTURED_DATA_PATH), "structured_data_with_features.xlsx"))
    # df_cleaned.to_excel(os.path.join(os.path.dirname(STRUCTURED_DATA_PATH), "structured_data_with_features_cleaned.xlsx"))
    sys.exit(0)

if __name__ == "__main__":
    main()