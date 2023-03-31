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
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:






