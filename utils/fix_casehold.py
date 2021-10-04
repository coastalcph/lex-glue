import re
import csv
import numpy as np
prompts = []
texts = []

with open('casehold_fixed.csv', "w", encoding="utf-8") as out_f:
    with open('casehold.csv', "r", encoding="utf-8") as f:
        for line in f.readlines():
            # Eliminate broken records
            if not re.match('\d', line) or not re.match('.+\d\n$', line):
                continue
            else:
                # Discard samples that are extremely long
                if len(line) < 5000:
                    out_f.write(line)

# Reload cleansed data and count text
with open('casehold_fixed.csv', "r", encoding="utf-8") as f:
    data = list(csv.reader(f))[1:]

for idx, sample in enumerate(data):
    for choice in sample[2:7]:
        texts.append(sample[1] + ' ' + choice)

# Compute approximate length per sample
t_lengths = [len(text.split()) for text in texts]

print(np.mean(t_lengths))
print(np.median(t_lengths))

