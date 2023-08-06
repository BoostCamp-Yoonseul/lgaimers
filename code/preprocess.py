import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder

def data_scaling(data):
    # Data Scaling
    scale_max_dict = {}
    scale_min_dict = {}

    for idx in tqdm(range(len(data)), desc="Data Scaling: "):
        maxi = np.max(data.iloc[idx,4:])
        mini = np.min(data.iloc[idx,4:])
        
        if maxi == mini :
            data.iloc[idx,4:] = 0
        else:
            data.iloc[idx,4:] = (data.iloc[idx,4:] - mini) / (maxi - mini)
        
        scale_max_dict[idx] = maxi
        scale_min_dict[idx] = mini
    return scale_max_dict, scale_min_dict, data

def label_encoding(data):
    # Label Encoding
    label_encoder = LabelEncoder()
    categorical_columns = ['대분류', '중분류', '소분류', '브랜드']

    for col in categorical_columns:
        label_encoder.fit(data[col])
        data[col] = label_encoder.transform(data[col])

    return data

def make_train_data(data, train_size, predict_size):
    '''
    학습 기간 블럭, 예측 기간 블럭의 세트로 데이터를 생성
    data : 일별 판매량
    train_size : 학습에 활용할 기간
    predict_size : 추론할 기간
    '''
    num_rows = len(data)
    window_size = train_size + predict_size
    
    input_data = np.empty((num_rows * (len(data.columns) - window_size + 1), train_size, len(data.iloc[0, :4]) + 1))
    target_data = np.empty((num_rows * (len(data.columns) - window_size + 1), predict_size))
    
    for i in tqdm(range(num_rows), desc = "Making Train inputs: "):
        encode_info = np.array(data.iloc[i, :4])
        sales_data = np.array(data.iloc[i, 4:])
        
        for j in range(len(sales_data) - window_size + 1):
            window = sales_data[j : j + window_size]
            temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window[:train_size]))
            input_data[i * (len(data.columns) - window_size + 1) + j] = temp_data
            target_data[i * (len(data.columns) - window_size + 1) + j] = window[train_size:]
    
    return input_data, target_data

def make_predict_data(data, train_size):
    '''
    평가 데이터(Test Dataset)를 추론하기 위한 Input 데이터를 생성
    data : 일별 판매량
    train_size : 추론을 위해 필요한 일별 판매량 기간 (= 학습에 활용할 기간)
    '''
    num_rows = len(data)
    
    input_data = np.empty((num_rows, train_size, len(data.iloc[0, :4]) + 1))
    
    for i in tqdm(range(num_rows), desc="making predict inputs"):
        encode_info = np.array(data.iloc[i, :4])
        sales_data = np.array(data.iloc[i, -train_size:])
        
        window = sales_data[-train_size : ]
        temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window[:train_size]))
        input_data[i] = temp_data
    
    return input_data