import json
import random
import tqdm
from collections import Counter

# NOTE: The dataset has been first enriched with metadata from SEC-EDGAR
# to figure out the year of submission for the original filings. This
# part is missing from the script.

# Parse original (augmented) dataset
categories = []
with open('ledgar.jsonl') as file:
    for line in tqdm.tqdm(file.readlines()):
        data = json.loads(line)
        categories.extend(data['labels'])

# Find the top-100 labels.
categories = set([label for label, count in Counter(categories).most_common()[:100]])


# Subsample examples labeled with one of the top-100 labels.
with open('ledgar_small.jsonl', 'w') as out_file:
    with open('ledgar.jsonl') as file:
        for line in tqdm.tqdm(file.readlines()):
            data = json.loads(line)
            if set(data['labels']).intersection(categories):
                labels = set(data['labels']).intersection(categories)
                if len(labels) == 1:
                    data['labels'] = sorted(list(labels))
                    data.pop('clause_types', None)
                    out_file.write(json.dumps(data)+'\n')


# Organize examples in clusters by year
years = []
samples = {year: [] for year in ['2016', '2017', '2018', '2019']}
with open('ledgar_small.jsonl') as file:
    for line in tqdm.tqdm(file.readlines()):
        data = json.loads(line)
        years.append(data['year'])
        data.pop('filer_cik', None)
        data.pop('filer_name', None)
        data.pop('filer_state', None)
        data.pop('filer_industry', None)
        samples[data['year']].append(data)


# Write final dataset 60k/10k/10k
random.seed(1)
with open('ledgar.jsonl', 'w') as file:
    final_samples = random.sample(samples['2016'], 30000)
    final_samples += random.sample(samples['2017'], 30000)
    for sample in final_samples:
        sample['data_type'] = 'train'
        file.write(json.dumps(sample) + '\n')
    final_samples = random.sample(samples['2018'], 10000)
    for sample in final_samples:
        sample['data_type'] = 'dev'
        file.write(json.dumps(sample) + '\n')
    final_samples = random.sample(samples['2019'], 10000)
    for sample in final_samples:
        sample['data_type'] = 'test'
        file.write(json.dumps(sample) + '\n')
