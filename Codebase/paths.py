import os

# adapt USER_DIR
USER_DIR = "C:\\Users\\lukas\\Uni Luki\\Digital_health\\Digital_health_psychology\\" # path to repo

DATAFRAMES_PATH = os.path.join(USER_DIR, "Dataframes")
QUESTIONNAIRE_PRE_PATH = os.path.join(DATAFRAMES_PATH, "Questionnaires", "SSSQ_pre.xlsx")
QUESTIONNAIRE_POST_PATH = os.path.join(DATAFRAMES_PATH, "Questionnaires", "SSSQ_post.xlsx")
AMYLASE_PATH = os.path.join(DATAFRAMES_PATH, "Saliva", "amylase.xlsx")
CORTISOL_PATH = os.path.join(DATAFRAMES_PATH, "Saliva", "cortisol.xlsx")
STUDY_LIST_PATH = os.path.join(DATAFRAMES_PATH, "StudyList.xlsx")

STRUCTURED_DATA_PATH = os.path.join(DATAFRAMES_PATH, "Output", "structured_data.xlsx")
ECG_PATH = os.path.join(USER_DIR, "ECG_data")
PLOT_PATH = os.path.join(USER_DIR, "Plots")