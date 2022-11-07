import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def safe_log(df):
    if df.min() <= 0.0:
        return np.log(df + np.abs(df.min()) + 1)
    else:
        return np.log(df)

def load_dataset():
    df = pd.read_csv('heart.csv')

    # Convert Categorical Labels
    df['IsMale'] = np.where(df['Sex'] == 'M', 1, 0)
    df['HasExerciseAngina'] =  np.where(df['ExerciseAngina'] == 'Y', 1, 0)

    df['ChestPainType_TA'] = np.where(df['ChestPainType'] == 'TA', 1, 0)
    df['ChestPainType_ATA'] = np.where(df['ChestPainType'] == 'ATA', 1, 0)
    df['ChestPainType_NAP'] = np.where(df['ChestPainType'] == 'NAP', 1, 0)
    #df['ChestPainType_ASY'] = np.where(df['ChestPainType'] == 'ASY', 1, 0)

    #df['RestingECG_Normal'] = np.where(df['RestingECG'] == 'Normal', 1, 0)
    df['RestingECG_ST'] = np.where(df['RestingECG'] == 'ST', 1, 0)
    df['RestingECG_LVH'] = np.where(df['RestingECG'] == 'LVH', 1, 0)

    #df['IsSTSlope_Flat'] =  np.where(df['ST_Slope'] == 'Flat', 1, 0)
    df['IsSTSlope_Up'] = np.where(df['ST_Slope'] == 'Up', 1, 0)
    df['IsSTSlope_Down'] = np.where(df['ST_Slope'] == 'Down', 1, 0)

    df = df.drop(columns=['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope'])
    
    real_cols = ['RestingBP', 'Cholesterol', 'Oldpeak']

    for i, name in enumerate(real_cols):
        df[name] = safe_log(df[name])

    df_X = df.drop(columns=['HeartDisease'])
    df_y = df['HeartDisease']

    scaler = StandardScaler()
    df_X_orig = df_X
    df_X = pd.DataFrame(scaler.fit_transform(df_X, df_y), columns = df_X.columns)

    #categorical_cols = [c for c in df_X.columns if c not in real_cols]
    #df_X[categorical_cols] = df_X_orig[categorical_cols]

    return df_X, df_y



