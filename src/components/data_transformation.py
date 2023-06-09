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
import warnings
warnings.filterwarnings("ignore")

from src.exception import CustomException
from src.logger import logging

class DataTransformationConfig:
    preprocessor_ob_file_path= os.path.join(r"C:\Fall 2022 courses\Project\Anamoly Detect\Multivariate-Time-Series-Anamoly-Detect\artifacts","preprocessed")
    root_path=os.path.join(r"C:\Fall 2022 courses\Project\Anamoly Detect\Multivariate-Time-Series-Anamoly-Detect",'artifacts')

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
    
class transform_df_O3():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.transform_function=transform_function()

    def fit(self,df):
        return self
    
    def transform(self,df):
        try:
            df=self.transform_function.handle_null(df)
            df['Aeroqual3']=self.transform_function.df_log(df['Aeroqual3'])
            df=self.transform_function.PCA_df(2,df,df.drop(columns=['SoC','TEMP','RHAMB']),['pc1','pc2'])
            return df
        except Exception as e:
            raise CustomException(e,sys)

class transform_df_PM():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.transform_function=transform_function()

    def fit(self,df):
        return self
    
    def transform(self,df):
        try:
            df=self.transform_function.handle_null(df)
            df['AirAssure1']=self.transform_function.df_log(df['AirAssure1'])
            df['AirAssure2']=self.transform_function.df_difference(df['AirAssure2'],1)
            df['AirAssure3']=self.transform_function.df_sqrt(df['AirAssure3'])
            df['OPC3']=self.transform_function.df_sqrt(df['OPC3'])
            df['Shinyei3']=self.transform_function.df_difference_sqrt(df['Shinyei3'],1)
            df=self.transform_function.PCA_df(2,df,df.dropna().drop(columns=['SoC','TEMP','RHAMB']),['pc1','pc2'])
            return df
        
        except Exception as e:
            raise CustomException(e,sys)
        
class transform_df_hppcf():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.transform_function=transform_function()

    def fit(self,df):
        return self
    
    def transform(self,df):
        try:
            df=self.transform_function.handle_null(df)
            df['Airbeam1']=self.transform_function.df_sqrt_log(df['Airbeam1'])
            df['Airbeam2']=self.transform_function.df_log(df['Airbeam2'])
            df['Airbeam3']=self.transform_function.df_log_sqrt(df['Airbeam3'])
            df=self.transform_function.PCA_df(2,df,df.drop(columns=['SoC','TEMP','RHAMB']),['pc1','pc2'])
            return df
        
        except Exception as e:
            raise CustomException(e,sys)
        
class transform_df_PMO():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.transform_function=transform_function()

    def fit(self,df):
        return self
    
    def transform(self,df):
        try:
            df=self.transform_function.handle_null(df)
            df=self.transform_function.PCA_df(2,df,df.drop(columns=['SoC','TEMP','RHAMB']),['pc1','pc2'])
            return df
        
        except Exception as e:
            raise CustomException(e,sys)

class transform_df_O3_NO():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.transform_function=transform_function()

    def fit(self,df):
        return self
    
    def transform(self,df):
        try:
            df=self.transform_function.handle_null(df)
            df=self.transform_function.PCA_df(2,df,df.drop(columns=['SOCP', 'TEMP', 'RHAMB', 'SoCO3', 'SoCNO2']),['pc1','pc2'])
            return df
        
        except Exception as e:
            raise CustomException(e,sys)

class get_transformer_object:
    def __init__(self):
        self.data_path=DataTransformation()
        self.transform_df_O3=transform_df_O3()
        self.transform_df_PM=transform_df_PM()
        self.transform_df_hppcf=transform_df_hppcf()
        self.transform_df_PMO=transform_df_PMO()
        self.transform_df_O3_NO=transform_df_O3_NO()

    def initiate_data_transformation(self):
        os.makedirs(self.data_path.data_transformation_config.preprocessor_ob_file_path , exist_ok=True)
        try:
            df_O3=pd.read_pickle(os.path.join(self.data_path.data_transformation_config.root_path,"df_O3.pkl"))
            df_PM=pd.read_pickle(os.path.join(self.data_path.data_transformation_config.root_path,"df_PM.pkl"))
            df_hppcf=pd.read_pickle(os.path.join(self.data_path.data_transformation_config.root_path,"df_hppcf.pkl"))
            df_PMO=pd.read_pickle(os.path.join(self.data_path.data_transformation_config.root_path,"df_PMO.pkl"))
            df_O3_NO=pd.read_pickle(os.path.join(self.data_path.data_transformation_config.root_path,"df_O3_NO.pkl"))
            

            logging.info("Read train and test data completed")

            #transforming the DF_O3
            self.transform_df_O3.fit(df_O3)
            df_O3=self.transform_df_O3.transform(df_O3)

            #transforming the DF_PM
            self.transform_df_PM.fit(df_PM)
            df_PM=self.transform_df_PM.transform(df_PM)

            #transforming the DF_hppcf
            self.transform_df_hppcf.fit(df_hppcf)
            df_hppcf=self.transform_df_hppcf.transform(df_hppcf)

            #transforming the DF_PMO
            self.transform_df_PMO.fit(df_PMO)
            df_PMO=self.transform_df_PMO.transform(df_PMO)

            #transforming the DF_O3_NO
            self.transform_df_O3_NO.fit(df_O3_NO)
            df_O3_NO=self.transform_df_O3_NO.transform(df_O3_NO)

            logging.info("Preparing to write preprocessed object")

            df_O3.to_pickle(os.path.join(self.data_path.data_transformation_config.preprocessor_ob_file_path,("df_O3.pkl")))
            df_hppcf.to_pickle(os.path.join(self.data_path.data_transformation_config.preprocessor_ob_file_path,("df_hppcf.pkl")))
            df_PM.to_pickle(os.path.join(self.data_path.data_transformation_config.preprocessor_ob_file_path,("df_PM.pkl")))
            df_PMO.to_pickle(os.path.join(self.data_path.data_transformation_config.preprocessor_ob_file_path,("df_PMO.pkl")))
            df_O3_NO.to_pickle(os.path.join(self.data_path.data_transformation_config.preprocessor_ob_file_path,("df_O3_NO.pkl")))

            logging.info("Obtaining preprocessing object")

            return (
                    pd.read_pickle(os.path.join(self.data_path.data_transformation_config.preprocessor_ob_file_path,("df_O3.pkl"))),
                    pd.read_pickle(os.path.join(self.data_path.data_transformation_config.preprocessor_ob_file_path,("df_hppcf.pkl"))),
                    pd.read_pickle(os.path.join(self.data_path.data_transformation_config.preprocessor_ob_file_path,("df_PM.pkl"))),
                    pd.read_pickle(os.path.join(self.data_path.data_transformation_config.preprocessor_ob_file_path,("df_PMO.pkl"))),
                    pd.read_pickle(os.path.join(self.data_path.data_transformation_config.preprocessor_ob_file_path,("df_O3_NO.pkl"))),
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=get_transformer_object()
    df_O3, df_hppcf, df_PM, df_PMO, df_O3_NO=obj.initiate_data_transformation()
    print(df_O3)








