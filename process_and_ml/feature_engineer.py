import pandas as pd
import numpy as np


def fs_new_cols(df: pd.DataFrame):
    df['WOHNLAGE'].replace(0, np.nan)

    # Saving values on lists to create new columns from CAMEO_INTL_2015.
    values_list_df = list(df['CAMEO_DEUINTL_2015'].values)

    # Creating columns from CAMEO_INTL_2015 column.

    df['WEALTH'] = [value if pd.isnull(value) else int(str(value)[0]) for value in values_list_df]
    df['LIFE_AGE'] = [value if pd.isnull(value) else int(str(value)[1]) for value in values_list_df]

    # Feature Engineer on PLZ8_BAUMAX.
    df['PLZ8_BAUMAX_FAMILY'] = np.where(df['PLZ8_BAUMAX'] == 5, 0, df['PLZ8_BAUMAX'])
    df['PLZ8_BAUMAX_BUSSINESS'] = np.where(df['PLZ8_BAUMAX'] == 5, 1,
                                           np.where(df['PLZ8_BAUMAX'].isnull(), df['PLZ8_BAUMAX'], 0))

    # Drop After Feature Engineer
    df.drop(['CAMEO_DEUINTL_2015', 'PLZ8_BAUMAX'], axis=1, inplace=True)
    print("Created WEALTH and LIFE_AGE column from 'CAMEO_DEUINTL_2015', original column was droped")
    print("Created 'PLZ8_BAUMAX_BUSSINESS' and 'PLZ8_BAUMAX_FAMILY' from 'PLZ8_BAUMAX_FAMILY', original column was "
          "droped.")
    return df


def fs_generation(df):
    generations = {0: [1, 2],  # 40s
                   1: [3, 4],  # 50s
                   2: [5, 6, 7],  # 60s
                   3: [8, 9],  # 70s
                   4: [10, 11, 12, 13],  # 80s
                   5: [14, 15]}  # 90s

    def classify_generation(value):
        try:
            for key, values in generations.items():
                if value in values:
                    return key
        # In case value is NaN
        except ValueError:
            return np.nan

    # Movement
    mainstream = [1, 3, 5, 8, 10, 12, 14]

    def classify_movement(value):
        try:
            if value in mainstream:
                return 1
            else:
                return 0
        # In case value is NaN
        except ValueError:
            return np.nan

    # Engineer generation column
    df['PRAEGENDE_JUGENDJAHRE_GEN'] = df['PRAEGENDE_JUGENDJAHRE'].apply(classify_generation)
    # azdias.loc[:,'PRAEGENDE_JUGENDJAHRE_GEN'] = azdias['PRAEGENDE_JUGENDJAHRE'].apply(classify_generation)

    # Engineer movement column
    df['PRAEGENDE_JUGENDJAHRE_MOV'] = df['PRAEGENDE_JUGENDJAHRE'].apply(classify_movement)
    # azdias_new.loc[:,'PRAEGENDE_JUGENDJAHRE_MOV'] = azdias_new['PRAEGENDE_JUGENDJAHRE'].apply(classify_movement)

    df.drop('PRAEGENDE_JUGENDJAHRE', axis=1, inplace=True)
    print(
        "Created 'PRAEGENDE_JUGENDJAHRE_GEN' and 'PRAEGENDE_JUGENDJAHRE_MOV' from 'PRAEGENDE_JUGENDJAHRE', original column was droped.")
    return df


def standardize_binary_columns(df):
    print("Working on Binary columns.")
    df['OST_WEST_KZ'].replace(['O', 'W'], [0, 1], inplace=True)
    df['VERS_TYP'].replace([2.0, 1.0], [1, 0], inplace=True)
    df['ANREDE_KZ'].replace([2, 1], [1, 0], inplace=True)
    return df


def correlated_columns_to_drop(df, min_corr_level=0.95):
    """Drop columns based on high correlated columns.

    Args:
        df (pd.DataFrame): Dataframe with columns to check correlation.
        min_corr_level (float, optional): Minimum correlation to choose to drop. Defaults to 0.95.

    Returns:
        List[List[str], pd.DataFrame]: List with columns dropped, and dataframe without this columns.
    """
    print("Removing Correlated columns.")
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than min_corr_level
    to_drop = [column for column in upper.columns if any(upper[column] > min_corr_level)]

    df_with_no_corr = df.drop(to_drop, axis=1)

    return [to_drop, df_with_no_corr]


def confirm_equal_columns_dataframe(df_1, df_2):
    return set(df_1).difference(df_2)


def remove_kba(df):
    print("Removing KBA05 columns")
    kba_cols = df.columns[df.columns.str.startswith('KBA05')]
    df.drop(list(kba_cols), axis='columns', inplace=True)
    return df


def categorize_columns(df):
    print("Get_Dummies with pandas")
    CATEGORICAL_COLUMNS = df.select_dtypes(include=['category', 'object']).columns.to_list()
    return pd.get_dummies(df, columns=CATEGORICAL_COLUMNS)


def fs_pipeline_stage_one(df):
    df = fs_new_cols(df)
    df = fs_generation(df)
    df = standardize_binary_columns(df)
    corr_result = correlated_columns_to_drop(df)
    df = remove_kba(corr_result[1])
    df = categorize_columns(df)
    return df
