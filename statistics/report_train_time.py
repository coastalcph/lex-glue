import json
import os
import numpy as np
import datetime


def main():

    for dataset in ['ecthr_a', 'ecthr_b', 'scotus', 'eurlex', 'ledgar', 'unfair_tos']:
        print(f'{dataset.upper()}')
        print('-'*100)
        BASE_DIR = f'logs/{dataset}'
        score_dicts = {}
        MODELS = ['bert-base-uncased', 'roberta-base', 'microsoft/deberta-base', 'nlpaueb/legal-bert-base-uncased',
                  'zlucia/custom-legalbert', 'allenai/longformer-base-4096', 'google/bigbird-roberta-base']
        for model in MODELS:
            score_dict = {'time': [], 'epochs': [], 'time/epoch': []}

            for seed in range(1, 6):
                seed = f'seed_{seed}'
                try:
                    with open(os.path.join(BASE_DIR, model, seed, 'trainer_state.json')) as json_file:
                        json_data = json.load(json_file)
                        score_dict['time'].append(json_data['log_history'][-1]['train_runtime'])
                        score_dict['epochs'].append(json_data['log_history'][-1]['epoch'])
                        score_dict['time/epoch'].append(json_data['log_history'][-1]['train_runtime']/json_data['log_history'][-1]['epoch'])
                except:
                    continue

            score_dicts[model] = score_dict

        for algo, stats in score_dicts.items():
            total_time = np.mean(stats["time"])
            time_epoch = np.mean(stats["time/epoch"])
            print(f'{algo:>35}: TRAIN TIME: {str(datetime.timedelta(seconds=total_time)).split(".")[0]}\t '
                  f'TIME/EPOCH: {str(datetime.timedelta(seconds=time_epoch)).split(".")[0]}\t'
                  f' EPOCHS: {np.mean(stats["epochs"]):.1f} ± {np.std(stats["epochs"]):.1f}')


if __name__ == '__main__':
    main()
