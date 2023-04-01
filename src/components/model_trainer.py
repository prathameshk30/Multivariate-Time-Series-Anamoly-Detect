import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from pyod.models.pca import PCA as PCA_out
from pythresh.thresholds.dsn import DSN

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    model_path=os.path.join(r"C:\Fall 2022 courses\Project\Anamoly Detect\Multivariate-Time-Series-Anamoly-Detect\artifacts","model")
    root_path=os.path.join(r"C:\Fall 2022 courses\Project\Anamoly Detect\Multivariate-Time-Series-Anamoly-Detect",'artifacts')

class model_df_O3:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def fit(self,df):
        return self
    
    def transform(self,df):
        clf = PCA_out(n_components=2)
        clf.fit(df[['Aeroqual1','Aeroqual2','Aeroqual3']])
        scores = clf.decision_scores_
        thres = DSN(metric='BHT')
        labels = thres.eval(scores)
        a=pd.DataFrame(labels).astype(int)
        a=a.rename(columns={0:'Anamoly'})
        df=pd.concat([df.reset_index(),a], axis=1)
        df=df.set_index('date')

class model_df_O3_NO:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def fit(self,df):
        return self
    
    def transform(self,df):
        clf = PCA_out(n_components=2)
        clf.fit(df[["Cairclip2", "Cairclip3"]])
        scores = clf.decision_scores_
        thres = DSN(metric='BHT')
        labels = thres.eval(scores)
        a=pd.DataFrame(labels).astype(int)
        a=a.rename(columns={0:'Anamoly'})
        df=pd.concat([df.reset_index(),a], axis=1)
        df=df.set_index('date')

class model_df_PM:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def fit(self,df):
        return self
    
    def transform(self,df):
        clf = PCA_out(n_components=2)
        clf.fit(df[["AirAssure1",	"AirAssure2",	"AirAssure3" ,"OPC1", 	"OPC2"	,"OPC3"	,"Shinyei1"	,"Shinyei2"	,"Shinyei3"	,"Speck1"	,"Speck2"	,"Speck3"]])
        scores = clf.decision_scores_
        thres = DSN(metric='BHT')
        labels = thres.eval(scores)
        a=pd.DataFrame(labels).astype(int)
        a=a.rename(columns={0:'Anamoly'})
        df=pd.concat([df.reset_index(),a], axis=1)
        df=df.set_index('date')

class model_df_hppcf:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def fit(self,df):
        return self
    
    def transform(self,df):
        clf = PCA_out(n_components=2)
        clf.fit(df[["Airbeam1",	"Airbeam2"	,"Airbeam3" ,"Dylos1_x"	,"Dylos2_x"	,"Dylos3_x"	,"Dylos1_y"	,"Dylos2_y",	"Dylos3_y"	,"TZOA1"	,"TZOA2"	,"TZOA3"]])
        scores = clf.decision_scores_
        thres = DSN(metric='BHT')
        labels = thres.eval(scores)
        a=pd.DataFrame(labels).astype(int)
        a=a.rename(columns={0:'Anamoly'})
        df=pd.concat([df.reset_index(),a], axis=1)
        df=df.set_index('date')

class model_df_PMO:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def fit(self,df):
        return self
    
    def transform(self,df):
        clf = PCA_out(n_components=2)
        clf.fit(df[["OPC1",	"OPC2",	"OPC3"]])
        scores = clf.decision_scores_
        thres = DSN(metric='BHT')
        labels = thres.eval(scores)
        a=pd.DataFrame(labels).astype(int)
        a=a.rename(columns={0:'Anamoly'})
        df=pd.concat([df.reset_index(),a], axis=1)
        df=df.set_index('date')



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        os.makedirs(self.model_trainer_config.model_path , exist_ok=True)
        try:
            pass

        except Exception as e:
            raise CustomException(e,sys)






