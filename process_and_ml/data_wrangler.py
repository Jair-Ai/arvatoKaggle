from typing import Dict, List, Optional
import logging
import json
import pandas as pd
import numpy as np

from models.constants import RANDOM_STATE


def load_customers(path, sample_ratio: Optional[float] = None):
    if sample_ratio:
        customers = pd.read_csv(path, sep=';').sample(frac=sample_ratio, random_state=RANDOM_STATE)
    else:
        customers = pd.read_csv(path, sep=';')
    customers = customers.rename(columns={'CAMEO_INTL_2015': 'CAMEO_DEUINTL_2015'})
    customers.drop(['PRODUCT_GROUP', 'CUSTOMER_GROUP', 'ONLINE_PURCHASE'], axis=1, inplace=True)
    return customers


def load_azdias(path, sample_ratio: Optional[float] = None):
    if sample_ratio:
        azdias = pd.read_csv(path, sep=';').sample(frac=sample_ratio, random_state=RANDOM_STATE)
    else:
        azdias = pd.read_csv(path, sep=';')
    azdias = azdias.rename(columns={'CAMEO_INTL_2015': 'CAMEO_DEUINTL_2015'})
    return azdias


def load_test_file(path):
    df_test = pd.read_csv(path, sep=';')
    df_test = df_test.rename(columns={'CAMEO_INTL_2015': 'CAMEO_DEUINTL_2015'})
    lnr_for_kaggle = df_test['LNR']
    df_test.drop('LNR', axis=1, inplace=True)
    return df_test, lnr_for_kaggle


def drop_columns_nan(df, threshold: float = .2, drop: bool = True):
    azdias_nan_per = df.isnull().mean()
    drop_cols = df.columns[azdias_nan_per > threshold]
    if drop:
        df.drop(drop_cols, axis=1, inplace=True)
    return df


class CleanUp:

    def __init__(self, paths: List[str]):
        """Class to clean up the dataframe for better analyses, need list of paths

        Args:
            paths:
        """

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
        self.df_info_columns.columns = column_array
        self.df_info_columns.drop('to_drop', axis=1, inplace=True)
        self.df_info_columns = self.df_info_columns.iloc[1:]

    def load_attributes_dataframe(self, path):
        self.df_column_attributes = pd.read_excel(path, engine='openpyxl')
        column_array = self.df_column_attributes.iloc[0].values
        column_array[0] = 'to_drop'
        self.df_column_attributes.columns = column_array
        self.df_column_attributes.drop('to_drop', axis=1, inplace=True)
        self.df_column_attributes = self.df_column_attributes.iloc[1:]

    def get_columns_with_info(self) -> np.array:
        """Function to discovery and drop columns without information.

        Returns:
            np.array: List with columns for clean up
        """

        columns_with_info = self.df_info_columns.Attribute.unique()
        columns_with_attributes = self.df_column_attributes.Attribute.unique()
        # Excluding nan column
        columns_with_attributes = np.delete(columns_with_attributes, 1)
        self.columns_with_info = list(np.unique(np.concatenate((columns_with_info, columns_with_attributes))))

        return self.columns_with_info

    def drop_columns_are_not_in_attribute_info(self, df):
        if not self.columns_with_info:
            logging.warning("You need to load get_columns_with_info first.")
            return pd.DataFrame([])
        else:
            columns_with_attributes = [col for col in self.columns_with_info if col in df.columns]
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
                columns_not_in['no_value_to_replace'].append(key)
        return df, columns_not_in

    def drop_initial_columns(self, df):
        return df.drop(self.columns_to_drop, axis=1, inplace=True)

    # def drop_customers_unique_columns(self):
    #     self.customers.drop(['PRODUCT_GROUP', 'CUSTOMER_GROUP', 'ONLINE_PURCHASE'], axis=1, inplace=True)

    def pipeline_clean_up(self, dfs: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, pd.DataFrame]:
        """Pipeline function to make the first data wrangler, after that dataframe is ready for catboost.

        Args:
            dfs (Dict[str,pd.DataFrame]): List with dataframes to clean_up.
            **kwargs: Arbitrary keyword arguments.
        Keyword Args:
            unknowns_df (pd.DataFrame): Dataframe with unknown variables of each columns.
            info_df (pd.DataFrame): Dataframe with column information.
            attr_df (pd.DataFrame): Dataframe with columns attributes.

        Returns:
            List[pd.Dataframe]: All Dataframes cleaned.

        """
        # 1 Get all columns on attr and info columns and drop

        dfs_cleaner = dfs
        self.load_nan_info(kwargs.get('unknowns_df'))
        if not self.columns_with_info:
            try:
                with open("./data/cleaned_data/columns_to_keep.json", "r") as read_file:
                    self.columns_with_info = json.load(read_file)['columns_to_keep']

            except OSError as e:
                logging.warning("\nNo file exist with columns, ll start cleanup pipeline")
                # 1 Get all columns on attr and info columns and drop
                self.load_info_dataframe(kwargs.get('info_df'))
                self.load_attributes_dataframe(kwargs.get('attr_df'))
                # 2 Get invalid values from attr_df and create invalid dataframe.
                self.get_columns_with_info()
                # 3 Drop columns there are not in attr or info.
                dfs_cleaner['azdias'] = self.drop_columns_are_not_in_attribute_info(dfs_cleaner['azdias'])
                # 4 Set unknown values as nan.
                dfs_cleaner['azdias'], _ = self.set_unknown_value_as_nan(dfs_cleaner['azdias'])
                # 5 Drop columns with more than some % of nan, choose the threshold, default = 20%.
                dfs_cleaner['azdias'] = drop_columns_nan(dfs_cleaner['azdias'])
                self.columns_with_info = dfs_cleaner['azdias'].columns
                columns_to_save = {'columns_to_keep': list(self.columns_with_info)}

                with open("./data/cleaned_data/columns_to_keep.json", "w") as outfile:
                    json.dump(columns_to_save, outfile)

        # 6 Iterate over dataframes and transform invalid on np.nan
        for key, value in dfs_cleaner.items():
            dfs_cleaner[key] = dfs_cleaner[key][self.columns_with_info]
            # 5 Drop columns with more than some % of nan, choose the threshold, default = 20%
            dfs_cleaner[key], _ = self.set_unknown_value_as_nan(dfs_cleaner[key])

        return dfs_cleaner
    # TODO: Test on test file and train file
    # TODO: Check the columns of cleaned azdias em compare with datawrangler.