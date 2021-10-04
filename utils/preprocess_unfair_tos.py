import glob
import json
import re

filenames = glob.glob('/Users/rwg642/Downloads/ToS/sentences/*.txt')

data = {}
total_sentence_count = 0
companies = []
for filename in filenames:
    with open(filename) as file:
        company = filename.split('/')[-1].split('.')[0]
        data[f'{company}'] = []
        text = ''
        for line in file.readlines():
            total_sentence_count += 1
            data[f'{company}'].append(
                {'company': company, 'release_year': '-', 'labels': [], 'text': line.replace('-lrb-', '(').replace('-rrb-', ')')})
            text += line + ' '

        matches = re.findall('20[0-2][0-9]', text)
        if matches:
            date = matches[0]
        else:
            date = '-'
        companies.append((company, date))

print('All sentences: ', total_sentence_count)

annotated_sentences = 0
for label_type, label_name in zip(
        ['Labels_A', 'Labels_CH', 'Labels_CR', 'Labels_J', 'Labels_LAW', 'Labels_LTD', 'Labels_TER', 'Labels_USE'],
        ['Arbitration', 'Unilateral change', 'Content removal', 'Jurisdiction', 'Choice of law',
         'Limitation of liability', 'Unilateral termination', 'Contract by using']):
    filenames = glob.glob(f'/Users/rwg642/Downloads/ToS/{label_type}/*.txt')
    sentence_count = 0
    for filename in filenames:
        company = filename.split('/')[-1].split('.')[0]
        with open(filename) as file:
            for idx, line in enumerate(file.readlines()):
                if line == '1\n':
                    data[f'{company}'][idx]['labels'].append(label_name)
                    sentence_count += 1
                    annotated_sentences += 1

    print(f'{label_type}: ', sentence_count)


print('Unannotated: ', total_sentence_count - annotated_sentences)

companies = [('Tinder', '-'), ('Betterpoints_UK', '-'), ('Deliveroo', '-'), ('9gag', '-'), ('Booking', '-'),
             ('YouTube', '-'), ('Yahoo', '-'), ('TrueCaller', '-'), ('Skype', '2006'), ('WorldOfWarcraft', '2012'),
             ('Viber', '2013'), ('Microsoft', '2013'), ('Instagram', '2013'), ('Rovio', '2013'), ('Onavo', '2013'),
             ('Moves-app', '2014'), ('Syncme', '2014'), ('Google', '2014'), ('Facebook', '2015'), ('Vivino', '2015'),
             ('Atlas', '2015'), ('Dropbox', '2016'), ('musically', '2016'), ('Spotify', '2016'), ('Endomondo', '2016'),
             ('WhatsApp', '2016'), ('Zynga', '2016'), ('PokemonGo', '2016'), ('Masquerade', '2016'),
             ('Skyscanner', '2016'), ('Nintendo', '2017'), ('Airbnb', '2017'), ('Crowdtangle', '2017'),
             ('TripAdvisor', '2017'), ('Supercell', '2017'), ('Headspace', '2017'), ('Fitbit', '2017'),
             ('Vimeo', '2017'), ('Oculus', '2017'), ('LindenLab', '2017'), ('Academia', '2017'), ('Amazon', '2017'),
             ('Netflix', '2017'), ('Snap', '2017'), ('Twitter', '2017'), ('LinkedIn', '2017'), ('Duolingo', '2017'),
             ('Uber', '2017'), ('Evernote', '2017'), ('eBay', '2017')]


with open('/Users/rwg642/PycharmProjects/LexGLUE/dataloaders/unfair_toc/unfair_toc.jsonl', 'w') as out_file:
    for company, year in companies[:30]:
        for record in data[f'{company}']:
            record['data_type'] = 'train'
            record['release_year'] = year
            out_file.write(json.dumps(record) + '\n')
    for company, year in companies[30:40]:
        for record in data[f'{company}']:
            record['data_type'] = 'val'
            record['release_year'] = year
            out_file.write(json.dumps(record) + '\n')
    for company, year in companies[40:]:
        for record in data[f'{company}']:
            record['data_type'] = 'test'
            record['release_year'] = year
            out_file.write(json.dumps(record) + '\n')

print()
# import numpy as np
# import matplotlib.pyplot as plt

# ecthr = [71.5, 17.0, 15.5, 18.5, 14.7]
# ledgar = [0.1, 81.1, 0.1, 81.1]
# ecthr_mean = np.mean(ecthr)
# ledgar_mean = np.mean(ledgar)
#
# ecthr_max = np.max(ecthr)
# ledgar_max= np.max(ledgar)
#
# ecthr_std = np.std(ecthr)
# ledgar_std = np.std(ledgar)
#
#
# fig, ax = plt.subplots()
# ax.bar(np.arange(2), [ecthr_max, ledgar_max], align='center', alpha=0.3, capsize=10)
# ax.bar(np.arange(2), [ecthr_mean, ledgar_mean], yerr=[ecthr_std, ledgar_std], align='center', alpha=0.6, ecolor='black', capsize=10)
# ax.set_ylabel('Macro-F1')
# ax.set_xticks(np.arange(2))
# ax.set_xticklabels(['ECtHR (Task A)', 'LEDGAR'])
# ax.yaxis.grid(True)
#
# # Save the figure and show
# plt.tight_layout()
# plt.show()
