import sys 
import os
import re
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

from src.exception import CustomException
from src.logger import logging

class DataTransformationConfig:
     preprocessor_ob_file_path= os.path.join(r"C:\Fall 2022 courses\Project\Anamoly Detect\Multivariate-Time-Series-Anamoly-Detection",'artifacts')

class data_transform_function:
    # Function to handle null values
    def handle_null(self,df):
        try:  
            df=df.drop_duplicates()
            b=[]
            a=df.columns.values
            for i in a:
                if re.findall('date',str(i)):
                    pass
                else:
                    b.append(i)
                for i in b:
                    df[i] = df[i].ewm(halflife=4).mean()
            df=df.dropna()
            return df
        except:
            print("error in handle_null")

    def pca_DF(self,n,df,x,*args):
        try:
            pca = PCA(n_components=n)
            principalComponents = pca.fit_transform(x)
            principal_df = pd.DataFrame(data = principalComponents, columns = [*args])
            df=df.reset_index()
            df=pd.concat([df,principal_df], axis=1)
            df=df.set_index('date')
            return df
        except:
            print("error in PCA function")
    
    def df_log(self,*args):
        return np.log(*args)
    
    def df_sqrt(self,*args):
        return np.sqrt(*args)
    
    def df_difference(self,*args,phase):
        return (*args).shift(phase)/(*args)
    
    def df_difference_sqrt(self,,*args,phase):
        return np.sqrt(*args).shift(phase)/(*args)

    def df_sqrt_log(self,*args):
        return np.sqrt(np.log(*args))
    
    def df_log_sqrt(self,*args):
        return np.log(np.sqrt(*args))
    
    def 
    


class DataTransformation:
     def __init__(self):
         self.data_transformation_config=DataTransformationConfig()
         self.transformation_function=data_transform_function()
    
     def data_transformer_obj(self):
        '''
        This function si responsible for data trnasformation
        '''
          try:



          except:
               pass




