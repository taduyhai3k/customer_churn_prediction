import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
import torch 
from torch.utils.data import Dataset, DataLoader 

class MCI_Dataset(Dataset):
    def __init__(self, root_path, data_path) -> None:
        super().__init__()
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    
    def __read_data__(self) -> None:
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        encode_columns = [col for col in df_raw.columns if df_raw[col].dtype == 'O'or df_raw[col].dtype == 'bool' ]    
        label_encoder = LabelEncoder()
        for col in encode_columns:
            df_raw[col] = label_encoder.fit_transform(df_raw[col])
        df_raw = df_raw.fillna(0)
        self.target = torch.tensor(df_raw[df_raw.columns[-1]].values, dtype = torch.float32)   
        df_raw = (df_raw - df_raw.mean()) / df_raw.std()
        self.features = torch.tensor(df_raw.drop(columns= df_raw.columns[-1]).values, dtype = torch.float32)
    
    def __len__(self):
        return len(self.target)     
    
    def __getitem__(self, index):
        return self.features[index], self.target[index]