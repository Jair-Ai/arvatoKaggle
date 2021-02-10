from typing import List, Optional
import glob
import os
import boto3
import sagemaker
import pandas as pd
import numpy as np


class WhereIs:

    def __init__(self, cloud: bool, bucket: Optional[str] = None, prefix: Optional[str] = None):
        self.cloud = cloud
        if self.cloud:
            self.sagemaker_session = sagemaker.Session()
            self.bucket = self.sagemaker_session.default_bucket()
            self.role = sagemaker.get_execution_role()
            self.s3_client = boto3.client('s3')
            self.object_list = self.s3_client.list_objects(Bucket=self.bucket)
            self.object_list_content = [df_file['Key'] for df_file in self.object_list['Contents'] if
                                        df_file['Key'][:-3] == 'csv' or df_file['Key'][:-3] == 'lsx']

        else:
            self.data_path = './data'
            self.object_list_content = [f for f_ in [glob.glob(self.data_path + file_types) for file_types in
                                                     ("/**/*.csv", "/**/*.xlsx")] for f in f_]

    @property
    def get_paths_list(self) -> List[str]:
        return self.object_list_content


# TODO: CREATING A FILE TO CLEAN UP DATA, TAKE ALL EXPERIMENTS
# TODO: SEPARATE CAMEO_DEUINTL_2015 IN 2 FEATURE, AGE, AND WEALTH
# TODO: INPUT NANS ON INVALID VALUES


class CleanUp:

    def __init__(self, azdias_df: str, df_customers: str, df_attributes: str, customers: bool):

        self.azdias = pd.read_csv(azdias_df, sep=';')
        self.customers = pd.read_csv(df_customers, sep=';')
        self.clean_customers = self.customers.copy
        self.clean_azdias = self.azdias.copy
        self.azdias_columns = self.azdias.columns
        self.customers_columns = self.customers
        self.columns_not_in = {'azdias': [], 'customers': []}
        self.columns_to_drop = {'D19_LETZTER_KAUF_BRANCHE': 'Other columns name, no descriptions',
                                'EINGEFUEGT_AM': 'No information about, data as input', 'LNR': 'Client Number',
                                'CAMEO_DEUG_2015': 'Too Many Values'}
        self.df_attr = df_attributes

    def create_nan_dict(self):
        cod_nan = pd.read_csv(self.df_attr)
        cod_nan.drop(cod_nan.columns[0], axis=1, inplace=True)
        cod_nan.set_index('Attribute', inplace=True)
        dict_to_nan = cod_nan.to_dict()['Value']
        dict_to_nan['CAMEO_DEU_2015'] = [-1, 'XX']
        dict_to_nan['CAMEO_INTL_2015'] = [-1, 'XX']

        columns_not_in = {'azdias': [], 'customers': []}
        for key, value, in dict_to_nan.items():
            if key in self.azdias_columns:
                self.clean_azdias[key] = self.clean_azdias[key].replace(value, np.nan)
            else:
                columns_not_in['azdias'].append(key)
            if key in self.customers_columns:
                self.clean_customers[key] = self.clean_customers[key].replace(value, np.nan)
            else:
                columns_not_in['customers'].append(key)

    def drop_initial_columns(self):
        self.clean_azdias.drop(self.columns_to_drop, axis=1, inplace=True)
        self.clean_customers.drop(self.columns_to_drop, axis=1, inplace=True)

    def drop_customers_unique_columns(self):
        self.customers.drop(['PRODUCT_GROUP', 'CUSTOMER_GROUP', 'ONLINE_PURCHASE'], axis=1, inplace=True)

    def drop_columns_nan(self, threshold: float, drop: bool = True):
        azdias_nan_per = self.clean_azdias.isnull().mean()
        drop_cols = self.clean_azdias.columns[azdias_nan_per > threshold]
        if drop:
            self.clean_azdias.drop(drop_cols, axis=1, inplace=True)
            self.clean_customers.drop(drop_cols, axis=1, inplace=True)

    def feature_engineer(self):

        self.clean_azdias['WOHNLAGE'].replace(0, np.nan)
        self.clean_customers['WOHNLAGE'].replace(0, np.nan)


        # Saving values on lists to create new columns from CAMEO_INTL_2015.
        values_list_azdias = list(self.clean_azdias['CAMEO_INTL_2015'].values)
        values_list_customers = list(self.clean_customers['CAMEO_INTL_2015'].values)

        # Creating columns from CAMEO_INTL_2015 column.

        self.clean_azdias['WEALTH'] = [value if pd.isnull(value) else int(str(value)[0]) for value in values_list_azdias]
        self.clean_azdias['LIFE_AGE'] = [value if pd.isnull(value) else int(str(value)[1]) for value in values_list_azdias]

        self.clean_customers['WEALTH'] = [value if pd.isnull(value) else int(str(value)[0]) for value in values_list_customers]
        self.clean_customers['LIFE_AGE'] = [value if pd.isnull(value) else int(str(value)[1]) for value in values_list_customers]

        # Drop Original column after.
        self.clean_azdias.drop(['CAMEO_INTL_2015'], axis=1, inplace=True)
        self.clean_customers.drop(['CAMEO_INTL_2015'], axis=1, inplace=True)

        # Feature Engineer on PLZ8_BAUMAX.
        self.clean_azdias['PLZ8_BAUMAX_FAMILY'] = np.where(self.clean_azdias['PLZ8_BAUMAX'] == 5, 0, self.clean_azdias['PLZ8_BAUMAX'])
        self.clean_azdias['PLZ8_BAUMAX_bussiness'] = np.where(self.clean_azdias['PLZ8_BAUMAX'] == 5, 1,
                                                        np.where(self.clean_azdias['PLZ8_BAUMAX'].isnull(), self.clean_azdias['PLZ8_BAUMAX'], 0))

        # Drop After Feature Engineer

    def normalize_categorical(self):
        self.clean_azdias['OST_WEST_KZ'].replace(['O', 'W'], [0, 1], inplace=True)
        self.clean_customers['OST_WEST_KZ'].replace(['O', 'W'], [0, 1], inplace=True)

    def confirm_equal_columns_dataframe(self):
        return self.clean_azdias.columns == self.clean_azdias

    def pipeline(self, threshold: float):
        self.drop_initial_columns()
        self.create_nan_dict()
        self.drop_columns_nan(threshold)
        self.feature_engineer()
        self.working_with_categorical_values()

