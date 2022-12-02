import itertools
import os
import csv
import re
import pandas as pd

drop_columns = [0, 1, 2, 3, 8, 9, 22, 23]

for file_name in os.listdir('pose'):
    if not file_name.endswith('_new.csv'):
        continue
    match = re.match(r'(\w+)-pose_90ct_new\.csv', file_name)
    assert match
    clip_name = match.group(1)
    df = pd.read_csv(os.path.join('pose', file_name))
    df = df.drop(columns=df.columns[0]).rename(columns={"0": "frame"})
    df = df.drop(columns=[str(col + 1) for col in drop_columns])
    df['frame'] = df['frame'].astype(int)
    df = df.interpolate(limit_direction='both')
    df.to_csv(os.path.join('pose_clean', f'{clip_name}.csv'), index=False)
