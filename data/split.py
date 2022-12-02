import pandas as pd
import random

random.seed(42)

samples = pd.read_csv('samples.csv')
clips = samples['clip_name'].unique()

split = (2, 2)

test_clips = ['BC1ADPI', 'BC1LACA']
val_clips = ['BC1ANGA', 'BC1MAMA']

val_df = samples[samples['clip_name'].isin(val_clips)]
test_df = samples[samples['clip_name'].isin(test_clips)]
train_df = samples[~samples['clip_name'].isin(test_clips + val_clips)]

train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)