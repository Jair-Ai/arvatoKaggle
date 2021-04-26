from typing import List, Optional, Dict
import glob
import os
import boto3
import sagemaker


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
            self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '../data')
            self.object_list_content = [f for f_ in [glob.glob(self.data_path + file_types) for file_types in
                                                     ("/**/*.csv", "/**/*.xlsx")] for f in f_]

    @property
    def get_paths_list(self) -> List[str]:
        return self.object_list_content
