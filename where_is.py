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

    def __init__(self, paths: List[str]):

        self.columns_not_in = {'azdias': [], 'customers': []}
        self.columns_to_drop = {'D19_LETZTER_KAUF_BRANCHE': 'Other columns name, no descriptions',
                                'EINGEFUEGT_AM': 'No information about, data as input', 'LNR': 'Client Number',
                                'CAMEO_DEUG_2015': 'Too Many Values'}
        self.paths: List[str] = paths
        self.df_info_columns: pd.DatataFrame = pd.DataFrame([])
        self.df_column_attributes: pd.DatataFrame = pd.DataFrame([])
        self.columns_with_info: np.array = np.array([])

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

    def initial_clean(self, df) -> np.array:
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

    def load_customers(self, path):
        customers = pd.read_csv(path, sep=';')
        customers = customers.rename(columns={'CAMEO_INTL_2015': 'CAMEO_DEUINTL_2015'})
        customers.drop(['PRODUCT_GROUP', 'CUSTOMER_GROUP', 'ONLINE_PURCHASE'], axis=1, inplace=True)
        return customers

    def load_azdias(self, path):
        azdias = pd.read_csv(path, sep=';')
        azdias = azdias.rename(columns={'CAMEO_INTL_2015': 'CAMEO_DEUINTL_2015'})
        return azdias

    def create_nan_dict(self, path):
        #Load Pre Process dataframe with all nans
        cod_nan = pd.read_csv(path)
        cod_nan.drop(cod_nan.columns[0], axis=1, inplace=True)
        cod_nan.set_index('Attribute', inplace=True)
        dict_to_nan = cod_nan.to_dict()['Value']
        dict_to_nan['CAMEO_DEU_2015'] = [-1, 'XX']
        dict_to_nan['CAMEO_DEUINTL_2015'] = [-1, 'XX']
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
    
    def all_columns_with_info(self):
        self.df_column_attributes = self.load_attributes_dataframe()
        columns_on_df_attributes = self.df_info_columns.Attribute.unique()

        self.df_info_columns = self.load_info_dataframe()
        columns_on_df_info = self.df_info_columns.Attribute.unique()
        #drop column with Nan on name
        columns_on_df_info = np.delete(columns_on_df_info, 1)
        #Concatenate columns in on single array
        self.columns_with_info = np.unique(np.concatenate((columns_on_df_info, columns_on_df_attributes)))

    
    def drop_columns_are_not_in_attribute_info(self, df):
        if not self.columns_with_info:
            self.all_columns_with_info()

        return df.drop(self.columns_with_info, inplace=True)



    def drop_columns_nan(self, threshold: float, drop: bool = True):
        azdias_nan_per = self.clean_azdias.isnull().mean()
        drop_cols = self.clean_azdias.columns[azdias_nan_per > threshold]
        if drop:
            self.clean_azdias.drop(drop_cols, axis=1, inplace=True)
            self.clean_customers.drop(drop_cols, axis=1, inplace=True)

    def feature_engineer_part_one(self, df: pd.DataFrame):

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

    def normalize_categorical(self, df):
        df['OST_WEST_KZ'].replace(['O', 'W'], [0, 1], inplace=True)
        df['OST_WEST_KZ'].replace(['O', 'W'], [0, 1], inplace=True)
    
    def feature_engineer_part_two(self):
        generations = {0: [1, 2], # 40s
               1: [3, 4], # 50s
               2: [5, 6, 7], # 60s
               3: [8, 9], # 70s
               4: [10, 11, 12, 13], # 80s
               5:[14, 15]} # 90s

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
        azdias['PRAEGENDE_JUGENDJAHRE_GEN'] = azdias['PRAEGENDE_JUGENDJAHRE'].apply(classify_generation)
        #azdias.loc[:,'PRAEGENDE_JUGENDJAHRE_GEN'] = azdias['PRAEGENDE_JUGENDJAHRE'].apply(classify_generation)

        # Engineer movement column
        azdias['PRAEGENDE_JUGENDJAHRE_MOV'] = azdias['PRAEGENDE_JUGENDJAHRE'].apply(classify_movement)
        #azdias_new.loc[:,'PRAEGENDE_JUGENDJAHRE_MOV'] = azdias_new['PRAEGENDE_JUGENDJAHRE'].apply(classify_movement)
        

    def confirm_equal_columns_dataframe(self, df_1, df_2):
        if df_1.columns == df_2.columns:
            return []

        return self.clean_azdias.columns == self.clean_azdias

    def pipeline(self, threshold: float):
        self.drop_initial_columns()
        self.create_nan_dict()
        self.drop_columns_nan(threshold)
        self.feature_engineer()
        self.working_with_categorical_values()
