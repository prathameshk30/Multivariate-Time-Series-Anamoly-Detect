import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    data_path: str=os.path.join(r"C:\Fall 2022 courses\Project\Anamoly Detect\Multivariate-Time-Series-Anamoly-Detection",'artifacts')

class dataset:
      def df_to_datetime(self,df):
        try:
            try:
                df=df.rename(columns={'timestamp': 'date'})
                pd.to_datetime(df['date'])
                df=df.set_index('date')
                return df
            except:
                df=pd.to_datetime(df['date'])
                df=df.set_index('date')
                return df
        except:
            print("Error in df_to_datetime")


      def create_dataset(self,a):
        try:
            for i in range(len(a)):
                a[i]=self.df_to_datetime(a[i])
            for i in range(len(a)-1):
                a[0]=a[0].merge(a[i+1].iloc[:len(a[i+1]),[0,1,2]], how='inner', on='date')
            return a[0]
        except:
            print("Error in create_dataset")
        

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        self.dataset=dataset()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            os.makedirs(self.ingestion_config.data_path, exist_ok=True)

            xls=pd.ExcelFile(r"C:\Fall 2022 courses\Project\Dataset\Dataset\CAIRSENSE_DataFiles.xlsx")

            xls_list=xls.sheet_names
            df_list=[]
            for i in range(len(xls_list)):
                df_list.append(pd.read_excel(r"C:\Fall 2022 courses\Project\Dataset\Dataset\CAIRSENSE_DataFiles.xlsx", sheet_name=xls_list[i]))
            
            logging.info('Read the dataset as dataframe')

            # Creating the dataset for O3
            df_O3=df_list[1]
            df_O3=self.dataset.df_to_datetime(df_O3)

            # Creating the dataset for PM
            df_PM=[df_list[2], df_list[7],df_list[9], df_list[10]]
            df_PM=self.dataset.create_dataset(df_PM)
            
            # Creating the dataset for hppcf
            df_hppcf=[df_list[3], df_list[5], df_list[6], df_list[11]]
            df_hppcf=self.dataset.create_dataset(df_hppcf)
        
            # Creating the dataset for O3 & NO
            c=df_list[4]
            df_O3_NO=self.dataset.df_to_datetime(c)
            df_O3_NO=df_O3_NO.drop(columns=['Cairclip1'])
         
            # Creating the dataset for PM10
            df_PMO=df_list[8]
            df_PMO=self.dataset.df_to_datetime(df_PMO)

            df_O3.to_pickle(os.path.join(self.ingestion_config.data_path,("df_O3.pkl")))
            df_hppcf.to_pickle(os.path.join(self.ingestion_config.data_path,("df_hppcf.pkl")))
            df_PM.to_pickle(os.path.join(self.ingestion_config.data_path,("df_PM.pkl")))
            df_PMO.to_pickle(os.path.join(self.ingestion_config.data_path,("df_PMO.pkl")))
            df_O3_NO.to_pickle(os.path.join(self.ingestion_config.data_path,("df_O3_NO.pkl")))

            logging.info("Exported the Dataset as dataframe")

            return (
                    pd.read_pickle(os.path.join(self.ingestion_config.data_path,("df_O3.pkl"))),
                    pd.read_pickle(os.path.join(self.ingestion_config.data_path,("df_hppcf.pkl"))),
                    pd.read_pickle(os.path.join(self.ingestion_config.data_path,("df_PM.pkl"))),
                    pd.read_pickle(os.path.join(self.ingestion_config.data_path,("df_PMO.pkl"))),
                    pd.read_pickle(os.path.join(self.ingestion_config.data_path,("df_O3_NO.pkl"))),
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    df_O3, df_hppcf, df_PM, df_PMO, df_O3_NO=obj.initiate_data_ingestion()
    print(df_O3)

