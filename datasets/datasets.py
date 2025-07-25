import torch
from torch.utils.data import TensorDataset, DataLoader

#전처리된 X, y 데이터를 TensorDataset을 이용해 DataLoader로 변환
#단순한 데이터를 넘기는 것 이상의 학습 효율성과 유연성을 확보하기 위함(안정적이고 효율적으로 학습 데이터를 모델에 공급하기 위함)

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size = 32):
    X_train_tensor = torch.tensor(X_train.values, dtype = torch.float32) #학습용 Feature
    y_train_tensor = torch.tensor(y_train.values, dtype = torch.long) #학습용 Label
    X_test_tensor = torch.tensor(X_test.values, dtype = torch.float32) #테스트용 Feature
    y_test_tensor = torch.tensor(y_test.values, dtype = torch.long) #테스트용 Label

    #TensorDataset 생성
    train_datasets = TensorDataset(X_train_tensor, y_train_tensor)
    test_datasets = TensorDataset(X_test_tensor, y_test_tensor)

    #DataLoader 생성
    train_DataLoader = DataLoader(train_datasets, batch_size = batch_size, shuffle = True)
    test_DataLoader = DataLoader(test_datasets, batch_size = batch_size, shuffle = True)

    return train_DataLoader, test_DataLoader

