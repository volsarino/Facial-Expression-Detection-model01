import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

EMOTION_LABEL={
    0:'angry',
    1:'disgust',
    2:'fear',
    3:'happy',
    4:'sad',
    5:'surprise',
    6:'neutral'
}

def setup_dir(base_path='./data'):
    for usage in ['train','test']:
        for emotion in EMOTION_LABEL.values():
            os.makedirs(os.path.join(base_path,usage,emotion),exist_ok=True)

def dataset(csv_path='fer2013.csv',output_base='./data'):
    if not os.path.exists(csv_path):
        print("csvファイルが見つかりません。")
        return 

    print("ファイル読み込み")
    df=pd.read_csv(csv_path)
    setup_dir(output_base)
    print("フォルダ保存")
    for idx,row in tqdm(df.iterrows(),total=len(df)):
        emotion_id=int(row['emotion'])
        emotion_name=EMOTION_LABEL[emotion_id]
        usage_type='train' if row['Usage']=='Training' else 'test'
        pixels=np.array(row['pixels'].split(),dtype='uint8').reshape(48,48)
        save_dir = os.path.join(output_base, usage_type, emotion_name)
        file_path = os.path.join(save_dir, f"{idx}.png")
        
        cv2.imwrite(file_path, pixels)

    print(f"\n変換完了 '{output_base}' ")

if __name__ == '__main__':
    dataset()
