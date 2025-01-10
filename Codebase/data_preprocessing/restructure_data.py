import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))) # for relative imports

import pandas as pd
import re

from paths import DATAFRAMES_PATH, QUESTIONNAIRE_POST_PATH, QUESTIONNAIRE_PRE_PATH, AMYLASE_PATH, CORTISOL_PATH, STUDY_LIST_PATH


def convert_df_saliva_to_wide_format(df):
    """
    this method converts the given saliva dataframe from long to wide format

    Args:
        df (pandas.core.frame.DataFrame): saliva dataframe in long format

    Returns:
        pandas.core.frame.DataFrame: saliva dataframe in wide format
    """
    # regex pattern to extract ids after VP and S
    pattern = r"(?<=VP_)(\d+).*?(?<=S)(\d+)"
    subjects = []

    for _, row in df.iterrows():
        id = row["Sample ID"]
        match = re.search(pattern, id)
        vp_code = f"VP{int(match.group(1))}"
        sample_id = int(match.group(2))
        
        vps_total = [s["VP"] for s in subjects]
        if not (vp_code in vps_total):
            # new subject
            sub = {"VP": vp_code, "Sample 0" + str(sample_id): row[df.columns[1]]}
            subjects.append(sub)
        else:
            subject_idx = [idx for idx, s in enumerate(subjects) if s.get('VP') == vp_code][0]
            subjects[subject_idx]["Sample 0" + str(sample_id)] = row[df.columns[1]]
    
    df = pd.DataFrame(subjects)
    return df

def process_questionnaire(subjects: list[dict], file_path: str):
    """
    Args:
        subjects (list[dict]): list of dicts, each dict contains study data for a particular study subject
        file_path (str): path to questionnaire (pre/post) dataframe

    Returns:
        list[dict]: subjects
    """
    # check whether we have PRE or POST questionnaire
    pre = "pre" in os.path.basename(file_path)

    df_questionare = pd.read_excel(file_path)
    if not pre:
        # replace column names in case of POST questionnaire
        df_questionare.columns = [col.replace("Pre", "Post") for col in df_questionare.columns]

    SSSQ_columns = [col for col in df_questionare.columns.tolist() if "SSSQ" in col]
    relevant_col = ["VPN_Kennung"] + SSSQ_columns
    for _, row in df_questionare[relevant_col].iterrows():
        vp_code = row["VPN_Kennung"]
        vps_total = [s["VP"] for s in subjects]
        codes_total = [s["Code"] for s in subjects]
        if not (vp_code in vps_total or vp_code in codes_total):
            # new subject --> add to list
            sub = row.to_dict()
            if sub['VPN_Kennung'].startswith("VP"):
                sub['VP'] = sub.pop('VPN_Kennung')
            else:
                sub['Code'] = sub.pop('VPN_Kennung')
                sub["VP"] = None
            subjects.append(sub)
        else:
            # subject already in list of subjects
            sub = row.to_dict()
            vp_code = sub.pop("VPN_Kennung")
            # append to respective entry
            subject_idx = [idx for idx, s in enumerate(subjects) if s.get('VP') == vp_code or s.get('Code') == vp_code][0]
            subjects[subject_idx].update(sub)

    return subjects

def process_saliva(subjects: list[dict], file_path: str):
    """
    Args:
        subjects (list[dict]): list of dicts, each dict contains study data for a particular study subject
        file_path (str): path to saliva (amylase/cortisol) dataframe

    Returns:
        list[dict]: subjects
    """
    # check whether we have amylase or cortisol
    amylase = "amylase" in os.path.basename(file_path)

    df_saliva = pd.read_excel(file_path)
    if amylase:
        df_saliva = df_saliva[["Unnamed: 1", "Unnamed: 2"]].iloc[2:]
        df_saliva.columns = ["Sample ID", "Amylase (U/ml)"]
    else:
        df_saliva = df_saliva[["Unnamed: 1", "Unnamed: 2"]].iloc[3:]
        df_saliva.columns = ["Sample ID", "Cortisol (nmol/l)"]

    df_saliva = convert_df_saliva_to_wide_format(df_saliva)
    if amylase:
        df_saliva.columns = ["Amylase " + col + " (U/ml)" if "sample" in col.lower() else col for col in df_saliva.columns]
    else:
        df_saliva.columns = ["Cortisol " + col + " (nmol/l)" if "sample" in col.lower() else col for col in df_saliva.columns]

    # add to subjects
    for _, row in df_saliva.iterrows():
        vp_code = row["VP"]
        vps_total = [s["VP"] for s in subjects]
        codes_total = [s["Code"] for s in subjects]
        if not (vp_code in vps_total or vp_code in codes_total):
            # new subject --> add to list
            sub = row.to_dict()
            subjects.append(sub)
        else:
            # subject already in list of subjects
            sub = row.to_dict()
            vp_code = sub.pop("VP")
            # append to respective entry
            subject_idx = [idx for idx, s in enumerate(subjects) if s.get('VP') == vp_code or s.get('Code') == vp_code][0]
            subjects[subject_idx].update(sub)
    
    return subjects  

def create_total_dataframe():
    """
    method to create a global dataframe summarizing all study data (questionnaires, saliva, ecg,...) in one place

    Returns:
        pandas.core.frame.DataFrame: dataframe with all study data
    """
    # start with StudyList
    df_study_list = pd.read_excel(STUDY_LIST_PATH)
    df_total = df_study_list[["VP", "Code", "Group", "Gender"]]
    subjects = []
    for _, row in df_total.iterrows():
        subjects.append(row.to_dict())

    # process questionnaires
    subjects = process_questionnaire(subjects, file_path=QUESTIONNAIRE_PRE_PATH)
    subjects = process_questionnaire(subjects, file_path=QUESTIONNAIRE_POST_PATH)
    
    # process saliva
    subjects = process_saliva(subjects, file_path=AMYLASE_PATH)
    subjects = process_saliva(subjects, file_path=CORTISOL_PATH)

    # convert list of subjects to dataframe
    df = pd.DataFrame(subjects)
    return df

def main():
    df = create_total_dataframe()
    df.to_pickle(os.path.join(DATAFRAMES_PATH, "Output", "structured_data.pkl"))
    df.to_excel(os.path.join(DATAFRAMES_PATH, "Output", "structured_data.xlsx"))

    sys.exit(0)

if __name__ == "__main__":
    main()