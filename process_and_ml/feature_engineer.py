import pandas as pd
from numba import np
import dask.dataframe as dd


def fs_for_cat_part_one(df: pd.DataFrame):
    df['WOHNLAGE'].replace(0, np.nan)

    # Saving values on lists to create new columns from CAMEO_INTL_2015.
    values_list_df = list(df['CAMEO_DEUINTL_2015'].values)

    # Creating columns from CAMEO_INTL_2015 column.

    df['WEALTH'] = [value if pd.isnull(value) else int(str(value)[0]) for value in values_list_df]
    df['LIFE_AGE'] = [value if pd.isnull(value) else int(str(value)[1]) for value in values_list_df]

    # Drop Original column after.
    df.drop(['CAMEO_INTL_2015'], axis=1, inplace=True)

    # Feature Engineer on PLZ8_BAUMAX.
    df['PLZ8_BAUMAX_FAMILY'] = np.where(df['PLZ8_BAUMAX'] == 5, 0, df['PLZ8_BAUMAX'])
    df['PLZ8_BAUMAX_bussiness'] = np.where(df['PLZ8_BAUMAX'] == 5, 1,
                                           np.where(df['PLZ8_BAUMAX'].isnull(), df['PLZ8_BAUMAX'], 0))

    # Drop After Feature Engineer
    df.drop(['CAMEO_INTL_2015', 'PLZ8_BAUMAX_FAMILY'], axis=1, inplace=True)


def fs_for_cat_part_two(df):
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

    df.drop('PRAEGENDE_JUGENDJAHRE')


def standardize_binary_columns(df):
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

    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than min_corr_level
    to_drop = [column for column in upper.columns if any(upper[column] > min_corr_level)]

    return [to_drop, df.drop(to_drop, inplace=True)]


def confirm_equal_columns_dataframe(df_1, df_2):
    return set(df_1).difference(df_2)


def fs_pipeline_stage_one(df):
    ...
    # df = fs_pipeline_stage_one(df)
    # df =


def fs_for_cat_boost(df_population: pd.DataFrame, df_customer: pd.DataFrame) -> pd.DataFrame:
    """Function to concatenate dataframe in any size.
    
    Args:
        df_population: azdias dataframe, ll have new column as customer=0
        df_customer: customer dataframe, ll have new column as customer=1

    Returns:
        

    """""

    df_to_model = dd.concat(
        [dd.from_pandas(df_customer.assign(is_customer=1), npartitions=10),
         dd.from_pandas(df_population.assign(is_customer=0), npartitions=10)]
    )
    path = '../data/cat_dfs/df_concat_pop_and_customer.csv'
    df_to_model.to_csv(path, single_file=True)
    dataframe_concatenated = pd.read_csv(path)
    return dataframe_concatenated
    # TODO TERMINAR ESTA PIPELINE DE DADOS E FACILITAR A VIDA NA HORA DE TREINAR O MODELO
