import shutil
import ftplib
import pandas as pd
from time import sleep
from urllib import request
from contextlib import closing
from tqdm.notebook import tqdm
import os


def download(path, data_directory='data'):
    DATA_DIR = data_directory
    df = pd.read_csv(path)
    for class_id in df['class'].unique():
        print(class_id)
        os.makedirs(os.path.join(DATA_DIR, 'train', str(class_id)), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, 'valid', str(class_id)), exist_ok=True)
    for i, row in tqdm(df.iterrows()):
        url = os.path.join('ftp://83.149.249.48/dataset/dataset_fragments', str(row['dir']), str(row['id']))
        # print('** Path: {}  **'.format(url), end="\r", flush=True)
        dst_path = os.path.join(DATA_DIR, str(row['category']), str(row['dir']), str(row['id']) + '.jpg')
        print(dst_path)
        while True:
            try:
                with closing(request.urlopen(url)) as r:
                    with open(dst_path, 'wb') as f:
                        shutil.copyfileobj(r, f)
                break
            except Exception:
                print('retrying...')
                sleep(1)