
import pandas as pd
import numpy as np
from typing import Dict, List

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
        self.dict_to_nan: Dict[str, List[int, str]]

    def load_nan_info(self, path):
        cod_nan = pd.read_csv(path)
        cod_nan.drop(cod_nan.columns[0], axis=1, inplace=True)
        cod_nan.set_index('Attribute', inplace=True)
        self.dict_to_nan = cod_nan.to_dict()['Value']
        self.dict_to_nan['CAMEO_DEU_2015'] = [-1, 'XX']
        self.dict_to_nan['CAMEO_DEUINTL_2015'] = [-1, 'XX']

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
    
    def load_customers(self, path):
        customers = pd.read_csv(path, sep=';')
        customers = customers.rename(columns={'CAMEO_INTL_2015': 'CAMEO_DEUINTL_2015'})
        customers.drop(['PRODUCT_GROUP', 'CUSTOMER_GROUP', 'ONLINE_PURCHASE'], axis=1, inplace=True)
        return customers

    def load_azdias(self, path):
        azdias = pd.read_csv(path, sep=';')
        azdias = azdias.rename(columns={'CAMEO_INTL_2015': 'CAMEO_DEUINTL_2015'})
        return azdias
    
    def load_test_file(self, path):
        


    def get_columns_with_info(self, df) -> np.array:
        """Function to discovery and drop columns without information.

        Args:
            df (pd.DataFrame): Dataframe to clean up columns that are not in description.

        Returns:
            np.array: List with columns for clean up
        """

        columns_with_info = self.df_info_columns.Attribute.unique()
        columns_with_attributes = self.df_column_attributes.Attribute.unique()
        # Excluding nan column
        columns_with_info = np.delete(columns_with_info, 1)
        self.columns_with_info = np.unique(np.concatenate((columns_with_info, columns_with_attributes)))

        return self.columns_with_info

    def drop_columns_are_not_in_attribute_info(self, df):
        if not self.columns_with_info:
            print("You need to load get_columns_with_info first.")
            return pd.DataFrame([])
        else:
            columns_with_attributes = [col for col in self.columns_with_info if col  in [df.columns]]
            df = df[columns_with_attributes]
        return df

    def create_nan_dict(self, path, df: pd.DataFrame):
        # Load Pre Process dataframe with all nans
        columns_not_in = {'no_value_to_replace': []}
        if not self.dict_to_nan:
            self.load_nan_info()
        for key, value, in self.dict_to_nan.items():
            if key in df:
                df[key] = df[key].replace(value, np.nan)
            else:
                columns_not_in['_value_to_replace'].append(key)
        return df

    def drop_initial_columns(self, df):
        return df.drop(self.columns_to_drop, axis=1, inplace=True)

    # def drop_customers_unique_columns(self):
    #     self.customers.drop(['PRODUCT_GROUP', 'CUSTOMER_GROUP', 'ONLINE_PURCHASE'], axis=1, inplace=True)

    def all_columns_with_info(self, path_info: str, path_attr: str):
        self.load_attributes_dataframe(path_attr)
        columns_on_df_attributes = self.df_info_columns.Attribute.unique()

        self.load_info_dataframe(path_info)
        columns_on_df_info = self.df_info_columns.Attribute.unique()
        # Drop column with Nan on name
        columns_on_df_info = np.delete(columns_on_df_info, 1)
        # Concatenate columns in on single array
        self.columns_with_info = list[np.unique(np.concatenate((columns_on_df_info, columns_on_df_attributes)))] 

    def drop_columns_nan(self, threshold: float, drop: bool = True):
        azdias_nan_per = self.clean_azdias.isnull().mean()
        drop_cols = self.clean_azdias.columns[azdias_nan_per > threshold]
        if drop:
            self.clean_azdias.drop(drop_cols, axis=1, inplace=True)
            self.clean_customers.drop(drop_cols, axis=1, inplace=True)

    def pipeline_for_test():
        self.load_test_file

