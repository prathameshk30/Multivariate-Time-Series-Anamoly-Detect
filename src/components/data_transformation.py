import sys 
import os
import re
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.exception import CustomException
from src.logger import logging

class DataTransformationConfig:
    preprocessor_ob_file_path= os.path.join(r"C:\Fall 2022 courses\Project\Anamoly Detect\Multivariate-Time-Series-Anamoly-Detection",'artifacts')

class transform_function:
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
    
    def df_log(self,*args):
        return np.log(*args)
    
    def df_sqrt(self,*args):
        return np.sqrt(*args)
    
    def df_difference(self,args,phase):
        return (args).shift(phase)/(args)
    
    def df_difference_sqrt(self,args,phase):
        return np.sqrt(args).shift(phase)/(args)

    def df_sqrt_log(self,*args):
        return np.sqrt(np.log(*args))
    
    def df_log_sqrt(self,*args):
        return np.log(np.sqrt(*args)) 
    
    def PCA_df(self,n,df,x,*args):
        try:
            pca=PCA(n_components=n)
            principalComponents = pca.fit_transform(x)
            principal_df = pd.DataFrame(data = principalComponents, columns = [*args])
            df=df.reset_index()
            df=pd.concat([df,principal_df], axis=1)
            df=df.set_index('date')
            return df
        except:
            print("error in PCA function")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.transform_function=transform_function()
    
class transform_df_O3(DataTransformation):
    def __init__(self):
        return self

    def fit(self,df):
        return self
    
    def transform(self,df):
        try:
            df=self.transform_function.handle_null(df)
            df['Aeroqual3']=self.transform_function.df_log(df['Aeroqual3'])
            return df
        except Exception as e:
            raise CustomException(e,sys)

class transform_df_PM(DataTransformation):
    def __init__(self):
        return self

    def fit(self,df):
        return self
    
    def transform(self,df):
        try:
            df=self.transform_function.handle_null(df)
            df['AirAssure1']=self.transform_function.df_log(df['AirAssure1'])
            df['AirAssure2']=self.transform_function.df_difference(df['AirAssure2'])
            df['AirAssure3']=self.transform_function.df_sqrt(df['AirAssure3'])
            df['OPC3']=self.transform_function.df_sqrt(df['OPC3'])
            df['Shinyei3']=self.transform_function.df_difference_sqrt(df['Shinyei3'],1)
            return df
        
        except Exception as e:
            raise CustomException(e,sys)
        
class transform_df_PM(DataTransformation):
    def __init__(self):
        return self

    def fit(self,df):
        return self
    
    def transform(self,df):
        try:
            df=self.transform_function.handle_null(df)
            df['AirAssure1']=self.transform_function.df_log(df['AirAssure1'])
            df['AirAssure2']=self.transform_function.df_difference(df['AirAssure2'])
            df['AirAssure3']=self.transform_function.df_sqrt(df['AirAssure3'])
            df['OPC3']=self.transform_function.df_sqrt(df['OPC3'])
            df['Shinyei3']=self.transform_function.df_difference_sqrt(df['Shinyei3'],1)
            return df
        
        except Exception as e:
            raise CustomException(e,sys)

class get_transformer_object:
    def initiate_data_transformation(self):
        try:
            root_path=r"C:\Fall 2022 courses\Project\Anamoly Detect\Multivariate-Time-Series-Anamoly-Detect\artifacts"
            df_O3=pd.read_pickle(os.path.join(root_path,"\df_O3.pkl"))
            df_PM=pd.read_pickle(os.path.join(root_path,"\df_PM.pkl"))
            df_PMO=pd.read_pickle(os.path.join(root_path,"\df_PMO.pkl"))
            df_O3_NO=pd.read_pickle(os.path.join(root_path,"\df_O3_NO.pkl"))
            df_hppcf=pd.read_pickle(os.path.join(root_path,"\df_hppcf.pkl"))

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_obj()

        except Exception as e:
            raise CustomException(e,sys)








