import itertools
import os
import csv
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--output-path', type=str, required=True)
args = parser.parse_args()

np.random.seed(args.seed)

num_samples = 2

typo_map = {
    'strech': 'stretch',
    'towardbookcase': 'toward bookcase'
}

first_5_frames_only = {}
normalize_word_map = {}


def normalize_words(str):
    normalized = []
    for word in re.split(r'(\W+)', str):
        parts = word.split('_')
        if len(parts) > 1 and parts[-1].isnumeric():
            parts.pop()
        parts = [typo_map.get(part, part) for part in parts]
        normalized_phrase = ' '.join(parts)
        normalized.append(normalized_phrase)
        if normalized_phrase != word:
            normalize_word_map[word] = normalized_phrase
    return ''.join(normalized)


motifs = defaultdict(list)
description_pattern = re.compile(r'(\w+)-motif_(\d+)(?:_\d+)?.csv')
for root, _, files in os.walk(os.path.join('description', 'SURF')):
    for file in files:
        match = re.match(description_pattern, file)
        if not match:
            print('invalid file', file)
            continue
        clip_name, motif_id = match.groups()
        motif_id = int(motif_id)
        motifs[clip_name].append((root, motif_id, file))

descriptions = []
for clip_name, files in motifs.items():
    files.sort()
    unique_id = 0
    for root, motif_id, file in files:
        with open(os.path.join(root, file), newline='') as csv_file:
            reader = csv.reader(csv_file)
            for i, row in enumerate(reader):
                if i % 2:  # skip list of frames
                    description = row[0]
                    if not description:
                        continue
                    if ';' in description:
                        description = description.split(';')[0]
                        first_5_frames_only[(
                            clip_name, motif_id, annotation_id)] = 0
                    annotation_id = i // 2
                    descriptions.append(
                        (clip_name, motif_id, annotation_id, normalize_words(description), unique_id))
                    unique_id += 1
descriptions_df = pd.DataFrame(descriptions, columns=[
                               'clip_name', 'motif_id', 'annotation_id', 'description', 'unique_id'])

clip_frames = defaultdict(list)
frame_pattern = re.compile(r'(\w+)-(\d+)-(\d+)-(\d+)')
for file_name in os.listdir('frame'):
    root, ext = os.path.splitext(file_name)
    if ext == '.csv':
        with open(os.path.join('frame', file_name), newline='') as csv_file:
            reader = csv.reader(csv_file)
            for frame_name in itertools.chain.from_iterable(reader):
                if not frame_name:
                    continue
                match = re.match(frame_pattern, frame_name)
                assert match
                clip_name, motif_id, annotation_id, frame = match.groups()
                motif_id, annotation_id, frame = int(
                    motif_id), int(annotation_id), int(frame)
                if (clip_name, motif_id, annotation_id) in first_5_frames_only:
                    if first_5_frames_only[(clip_name, motif_id, annotation_id)] >= 5:
                        continue
                    first_5_frames_only[(
                        clip_name, motif_id, annotation_id)] += 1
                sample_id = f'{clip_name}_{motif_id}_{annotation_id}_{frame}'
                clip_frames[(clip_name, motif_id, annotation_id)].append(frame)
# sampled_frames_df = frames_df.groupby(['clip_name', 'motif_id', 'annotation_id']).sample(n=2, replace=True)

unique_descriptions_df = descriptions_df.groupby('description').sample(n=1)

samples = []
for key, df in unique_descriptions_df.groupby(['clip_name', 'motif_id', 'annotation_id']):
    duplicated_samples = pd.concat([df] * num_samples, ignore_index=True)
    frames = pd.Series(clip_frames[key], name='frame')
    if len(frames) >= len(duplicated_samples):
        sampled_frames = frames.sample(n=len(duplicated_samples))
    else:
        sampled_frames = pd.concat([
            frames.sample(frac=1),
            frames.sample(n=(len(duplicated_samples) - len(frames)), replace=True)
        ], ignore_index=True)
        sampled_frames = sampled_frames.sample(frac=1)
    samples.append(duplicated_samples.assign(frame=sampled_frames.values))

merged = pd.concat(samples)
merged['sample_id'] = merged.apply(
    lambda row: f'{row["clip_name"]}_{row["unique_id"]}_{row["frame"]}', axis=1)
merged['image_name'] = merged.apply(
    lambda row: f'{row["clip_name"]}_{row["frame"]}.png', axis=1)
merged = merged.drop(columns='unique_id')
merged = merged.drop_duplicates()
for original, normalized in normalize_word_map.items():
    print(f'{original}\t->\t{normalized}')

merged.to_csv(os.path.join(args.output_path, 'samples.csv'), index=False)

clips = merged['clip_name'].unique()
np.random.shuffle(clips)
shuffled_clips = clips.tolist()

val_size = 300
test_size = 300

def take(n):
    clips = []
    total = 0
    while total < n:
        clip = shuffled_clips.pop()
        clip_df = merged[merged['clip_name'] == clip]
        clips.append(clip_df)
        total += len(clip_df)
    return pd.concat(clips, ignore_index=True)

val_df = take(val_size)
test_df = take(test_size)
train_df = merged[merged['clip_name'].isin(shuffled_clips)]

train_df.to_csv(os.path.join(args.output_path, 'train.csv'), index=False)
val_df.to_csv(os.path.join(args.output_path, 'val.csv'), index=False)
test_df.to_csv(os.path.join(args.output_path, 'test.csv'), index=False)