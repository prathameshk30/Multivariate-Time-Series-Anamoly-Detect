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
        return df

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
        return df

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
        return df

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
        return df

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
        return df



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        self.model_df_O3=model_df_O3()
        self.model_df_PM=model_df_PM()
        self.model_df_O3_NO=model_df_O3_NO()
        self.model_df_hppcf=model_df_hppcf()
        self.model_df_PMO=model_df_PMO()

    def initiate_model_trainer(self,train_array,test_array):
        os.makedirs(self.model_trainer_config.model_path , exist_ok=True)
        try:
            df_O3=pd.read_pickle(os.path.join(self.model_trainer_config.root_path,"df_O3.pkl"))
            df_PM=pd.read_pickle(os.path.join(self.model_trainer_config.root_path,"df_PM.pkl"))
            df_hppcf=pd.read_pickle(os.path.join(self.model_trainer_config.root_path,"df_hppcf.pkl"))
            df_PMO=pd.read_pickle(os.path.join(self.model_trainer_config.root_path,"df_PMO.pkl"))
            df_O3_NO=pd.read_pickle(os.path.join(self.model_trainer_config.root_path,"df_O3_NO.pkl"))

            self.model_df_O3.fit(df_O3)
            df_O3=self.model_df_O3.transform(df_O3)

            self.model_df_O3.fit(df_PM)
            df_PM=self.model_df_PM.transform(df_PM)

            self.model_df_hppcf.fit(df_hppcf)
            df_hppcf=self.model_df_hppcf.transform(df_hppcf)

            self.model_df_PMO.fit(df_PMO)
            df_PMO=self.model_df_PMO.transform(df_PMO)

            self.model_df_O3_NO.fit(df_O3_NO)
            df_O3_NO=self.model_df_O3_NO.transform(df_O3_NO)

            logging.info("Preparing to write preprocessed object")

            df_O3.to_pickle(os.path.join(self.model_trainer_config.model_path,("df_O3.pkl")))
            df_hppcf.to_pickle(os.path.join(self.model_trainer_config.model_path,("df_hppcf.pkl")))
            df_PM.to_pickle(os.path.join(self.model_trainer_config.model_path,("df_PM.pkl")))
            df_PMO.to_pickle(os.path.join(self.model_trainer_config.model_path,("df_PMO.pkl")))
            df_O3_NO.to_pickle(os.path.join(self.model_trainer_config.model_path,("df_O3_NO.pkl")))

            logging.info("Obtaining preprocessing object")

            return (
                    pd.read_pickle(os.path.join(self.model_trainer_config.model_path,("df_O3.pkl"))),
                    pd.read_pickle(os.path.join(self.model_trainer_config.model_path,("df_hppcf.pkl"))),
                    pd.read_pickle(os.path.join(self.model_trainer_config.model_path,("df_PM.pkl"))),
                    pd.read_pickle(os.path.join(self.model_trainer_config.model_path,("df_PMO.pkl"))),
                    pd.read_pickle(os.path.join(self.model_trainer_config.model_path,("df_O3_NO.pkl"))),
            )


        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=ModelTrainer()
    df_O3, df_hppcf, df_PM, df_PMO, df_O3_NO=obj.initiate_model_trainer()
    print(df_O3)





