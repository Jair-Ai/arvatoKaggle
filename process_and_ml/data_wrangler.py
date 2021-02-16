from typing import Dict, List, Optional
import logging
import json
import os
import pandas as pd
import numpy as np


def load_customers(path):
    customers = pd.read_csv(path, sep=';')
    customers = customers.rename(columns={'CAMEO_INTL_2015': 'CAMEO_DEUINTL_2015'})
    customers.drop(['PRODUCT_GROUP', 'CUSTOMER_GROUP', 'ONLINE_PURCHASE'], axis=1, inplace=True)
    return customers


def load_azdias(path):
    azdias = pd.read_csv(path, sep=';')
    azdias = azdias.rename(columns={'CAMEO_INTL_2015': 'CAMEO_DEUINTL_2015'})
    return azdias


def load_test_file(path, columns_to_keep):
    df_test = pd.read_csv(path, sep=';')
    df_test = df_test.rename(columns={'CAMEO_INTL_2015': 'CAMEO_DEUINTL_2015'})
    return df_test[columns_to_keep]


def drop_columns_nan(df, threshold: float = .2, drop: bool = True):
    azdias_nan_per = df.isnull().mean()
    drop_cols = df.columns[azdias_nan_per > threshold]
    if drop:
        df.drop(drop_cols, axis=1, inplace=True)
    return df


def pipeline_data_wrangler(df_test, path_dict, columns_to_keep, catboost: bool = False):
    # Step 1 drop columns that are not inicial analises
    df_test = df_test[columns_to_keep]


class CleanUp:

    def __init__(self, paths: List[str]):

        self.columns_not_in = {'azdias': [], 'customers': []}
        self.columns_to_drop = {'D19_LETZTER_KAUF_BRANCHE': 'Other columns name, no descriptions',
                                'EINGEFUEGT_AM': 'No information about, data as input', 'LNR': 'Client Number',
                                'CAMEO_DEUG_2015': 'Too Many Values'}
        self.paths: List[str] = paths
        self.df_info_columns: pd.DatataFrame = pd.DataFrame([])
        self.df_column_attributes: pd.DatataFrame = pd.DataFrame([])
        self.columns_with_info: np.array = np.array([])
        self.dict_to_nan: Dict[str, List[int, str]] = {}

    def load_info_dataframe(self, path):
        self.df_info_columns = pd.read_excel(path, engine='openpyxl')
        column_array = self.df_info_columns.iloc[0].values
        column_array[0] = 'to_drop'
        self.df_info_columns.drop('to_drop', axis=1, inplace=True)

    def load_attributes_dataframe(self, path):
        df_column_attributes = pd.read_excel(path, engine='openpyxl')
        column_array = df_column_attributes.iloc[0].values
        column_array[0] = 'to_drop'
        df_column_attributes.columns = column_array
        self.df_column_attributes.drop('to_drop', axis=1, inplace=True)

    def get_columns_with_info(self) -> np.array:
        """Function to discovery and drop columns without information.

        Returns:
            np.array: List with columns for clean up
        """

        columns_with_info = self.df_info_columns.Attribute.unique()
        columns_with_attributes = self.df_column_attributes.Attribute.unique()
        # Excluding nan column
        columns_with_info = np.delete(columns_with_info, 1)
        self.columns_with_info = list(np.unique(np.concatenate((columns_with_info, columns_with_attributes))))

        return self.columns_with_info

    def drop_columns_are_not_in_attribute_info(self, df):
        if not self.columns_with_info:
            logging.warning("You need to load get_columns_with_info first.")
            return pd.DataFrame([])
        else:
            columns_with_attributes = [col for col in self.columns_with_info if col in [df.columns]]
            df = df[columns_with_attributes]
        return df

    def load_nan_info(self, path):
        cod_nan = pd.read_csv(path)
        cod_nan.drop(cod_nan.columns[0], axis=1, inplace=True)
        cod_nan.set_index('Attribute', inplace=True)
        self.dict_to_nan = cod_nan.to_dict()['Value']
        self.dict_to_nan['CAMEO_DEU_2015'] = [-1, 'XX']
        self.dict_to_nan['CAMEO_DEUINTL_2015'] = [-1, 'XX']

    def set_unknown_value_as_nan(self, df: pd.DataFrame, path: Optional[str] = None):
        """

        Args:
            df:
            path:

        Returns:

        """
        # Load Pre Process dataframe with all nans
        columns_not_in = {'no_value_to_replace': []}
        if not self.dict_to_nan:
            self.load_nan_info(path)
        for key, value, in self.dict_to_nan.items():
            if key in df:
                df[key] = df[key].replace(value, np.nan)
            else:
                columns_not_in['_value_to_replace'].append(key)
        return df, columns_not_in

    def drop_initial_columns(self, df):
        return df.drop(self.columns_to_drop, axis=1, inplace=True)

    # def drop_customers_unique_columns(self):
    #     self.customers.drop(['PRODUCT_GROUP', 'CUSTOMER_GROUP', 'ONLINE_PURCHASE'], axis=1, inplace=True)

    def pipeline_clean_up(self, path_dict: Dict[str, str], dfs: List[pd.DataFrame]):
        """Pipeline function to make the first data wrangler, after that dataframe is ready for catboost

        Args:
            path_dict Dict[str,str]: Dict with all necessary path for pipeline, minimum key ['nan_info']
            dfs List[pd.DataFrame]: List with dataframes to clean_up

        Returns:

        """
        # 1 Get all columns on attr and info columns and drop
        # 2 Drop columns there are not in attr or info
        dfs_cleaner = dfs
        try:
            with open("../data/cleaned_data/columns_to_drop.json", "r") as read_file:
                columns_to_keep = json.load(read_file)
                self.load_nan_info(path_dict['nan_info'])
                for idx, df in enumerate(dfs_cleaner):
                    dfs_cleaner[idx] = dfs_cleaner[columns_to_keep]
                    dfs_cleaner[idx], _ = self.set_unknown_value_as_nan(df)
        except OSError as e:
            logging.warning("No file exist with columns, ll start cleanup pipeline")
            # 1 Get all columns on attr and info columns and drop
            self.load_info_dataframe(path_dict['info_df'])
            self.load_attributes_dataframe(path_dict['attr_df'])
            self.get_columns_with_info()
            # 2 Drop columns there are not in attr or info
            for df in dfs_cleaner:
                df = self.drop_columns_are_not_in_attribute_info(df)
            # 3 Get invalid values from attr_df and create invalid dataframe
            self.load_nan_info(path_dict['nan_info'])
            # 4 Iterate over dataframes and transform invalid on np.nan
            for idx, df in enumerate(dfs_cleaner):
                dfs_cleaner[idx], _ = self.set_unknown_value_as_nan(df)
                # 5 Drop columns with more than some % of nan, choose the threshold, default = 20%
                dfs_cleaner[idx] = drop_columns_nan(df)
            columns_to_save = {'columns_to_keep': [dfs_cleaner[0].columns]}
            with open("../data/cleaned_data/columns_to_drop.json", "w") as outfile:
                json.dump(columns_to_save, outfile)
        finally:
            return dfs_cleaner
