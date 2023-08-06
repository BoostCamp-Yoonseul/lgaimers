import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from dataloader import CustomDataset
from utils import seed_everything, config_parser
from preprocess import make_predict_data, data_scaling, label_encoding

def inference(model, test_loader, device):
    predictions = []
    
    with torch.no_grad():
        for X in tqdm(iter(test_loader)):
            X = X.to(device)
            output = model(X)
            # 모델 출력인 output을 CPU로 이동하고 numpy 배열로 변환
            output = output.cpu().numpy()
            predictions.extend(output)
    
    return np.array(predictions)

if __name__ =="__main__":
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config = config_parser()

    seed_everything(config['SEED']) 

    # 데이터 전처리
    train_data = pd.read_csv('../data/train.csv').drop(columns=['ID', '제품'])
    scale_max_dict, scale_min_dict, train_data = data_scaling(train_data)
    train_data = label_encoding(train_data)
    test_input = make_predict_data(train_data, train_size=config['TRAIN_WINDOW_SIZE'])

    test_dataset = CustomDataset(test_input, None)
    test_loader = DataLoader(test_dataset, batch_size = config['BATCH_SIZE'], shuffle=False, num_workers=0)

    model = torch.load("../output/result_model")
    model.eval()
    pred = inference(model, test_loader, device)

    # 추론 결과를 inverse scaling
    for idx in tqdm(range(len(pred))):
        pred[idx, :] = pred[idx, :] * (scale_max_dict[idx] - scale_min_dict[idx]) + scale_min_dict[idx]
        
    # 결과 후처리
    pred = np.round(pred, 0).astype(int)

    submit = pd.read_csv('../data/sample_submission.csv')
    submit.head()

    submit.iloc[:,1:] = pred
    submit.head()

    submit.to_csv('./20230906_submission_reappearance.csv', index=False)