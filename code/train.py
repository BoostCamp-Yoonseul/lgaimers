import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import seed_everything, config_parser
from preprocess import make_train_data, data_scaling, label_encoding
from dataloader import CustomDataset
from model import LSTM

def validation(model, val_loader, criterion, device):
    model.eval()
    val_loss = []
    
    with torch.no_grad():
        for X, Y in tqdm(iter(val_loader)):
            X = X.to(device)
            Y = Y.to(device)
            
            output = model(X)
            loss = criterion(output, Y)
            
            val_loss.append(loss.item())
    return np.mean(val_loss)

def train(model, optimizer, train_loader, val_loader, device, config):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    best_loss = 9999999
    best_model = None
    
    for epoch in tqdm(range(1, config['EPOCHS']+1)):
        model.train()
        train_loss = []
        for X, Y in tqdm(iter(train_loader)):
            X = X.to(device)
            Y = Y.to(device)
            
            optimizer.zero_grad()
            
            output = model(X)
            loss = criterion(output, Y)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        
        val_loss = validation(model, val_loader, criterion, device)
        print(f'Epoch : {epoch} | Train Loss : {np.mean(train_loss):.5f} | Val Loss : [{val_loss:.5f}]')
        
        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model

    return best_model


if __name__ == "__main__":
    config = config_parser()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    seed_everything(config['SEED']) # Seed 고정

    # 데이터 전처리
    train_data = pd.read_csv('../data/train.csv').drop(columns=['ID', '제품'])
    _, _, train_data = data_scaling(train_data)
    train_data = label_encoding(train_data)

    train_input, train_target = make_train_data(train_data, train_size=config['TRAIN_WINDOW_SIZE'], predict_size=config['PREDICT_SIZE'])

    # Train / Validation Split
    data_len = len(train_input)
    val_input = train_input[-int(data_len*0.2):]
    val_target = train_target[-int(data_len*0.2):]
    train_input = train_input[:-int(data_len*0.2)]
    train_target = train_target[:-int(data_len*0.2)]

    train_dataset = CustomDataset(train_input, train_target)
    train_loader = DataLoader(train_dataset, batch_size = config['BATCH_SIZE'], shuffle=True, num_workers=0)

    val_dataset = CustomDataset(val_input, val_target)
    val_loader = DataLoader(val_dataset, batch_size = config['BATCH_SIZE'], shuffle=False, num_workers=0)

    model = LSTM()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = config["LEARNING_RATE"])
    model = train(model, optimizer, train_loader, val_loader, device, config)
    torch.save(model, "../output/result_model")