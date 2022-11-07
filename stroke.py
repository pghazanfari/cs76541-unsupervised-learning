import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def safe_log(df):
    if df.min() <= 0.0:
        return np.log(df + np.abs(df.min()) + 1)
    else:
        return np.log(df)

def load_dataset(rng=104582):
    # Load data
    # Source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')

    # Convert Categorical Labels
    df['is_male'] = np.where(df['gender'] == 'Male', 1, 0)
    df['ever_married'] = np.where(df['ever_married'] == 'Yes', 1, 0)
    df['is_rural_residence'] = np.where(df['Residence_type'] == 'Rural', 1, 0)

    # Private = all of these are 0
    df['work_type_Self_employed'] = np.where(df['work_type'] == 'Self-employed', 1, 0)
    df['work_type_children'] = np.where(df['work_type'] == 'children', 1, 0)
    df['work_type_Govt_job'] = np.where(df['work_type'] == 'Govt_job', 1, 0)
    df['work_type_Never_worked'] = np.where(df['work_type'] == 'Never_worked', 1, 0)

    # never smoked = all of these are 0
    df['smoking_status_unknown'] = np.where(df['smoking_status'] == 'Unknown', 1, 0)
    df['smoking_status_formerly_smoked'] = np.where(df['smoking_status'] == 'formerly smoked', 1, 0)
    df['smoking_status_smokes'] = np.where(df['smoking_status'] == 'smokes', 1, 0)

    df = df.drop(columns=['id', 'gender', 'Residence_type', 'work_type', 'smoking_status'])

    df_X = df.drop(columns=['stroke'])
    df_y = df['stroke']

    imp = SimpleImputer(strategy='median')

    imp.fit(df_X)
    df_X = pd.DataFrame(imp.transform(df_X), columns=df_X.columns)

    df_X['bmi'] = safe_log(df_X['bmi'])

    df_X = df_X.drop(columns=['avg_glucose_level'])

    scaler = StandardScaler()
    df_X = pd.DataFrame(scaler.fit_transform(df_X, df_y), columns = df_X.columns)

    over_sampler = SMOTE(random_state=rng)
    os_df_X, os_df_y = over_sampler.fit_resample(df_X, df_y)

    # Returns the oversampled and the original dataset
    return os_df_X, os_df_y, df_X, df_y