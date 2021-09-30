import json
import os
import argparse
import numpy as np


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--dataset',  default='eurlex')
    parser.add_argument('--scratch', default=False)
    config = parser.parse_args()

    BASE_DIR = f'logs/{config.dataset}'

    if os.path.exists(BASE_DIR):
        print(f'{BASE_DIR} exists!')

    score_dicts = {}
    MODELS = ['bert-base-uncased', 'roberta-base', 'microsoft/deberta-base', 'nlpaueb/legal-bert-base-uncased',
              'zlucia/custom-legalbert', 'allenai/longformer-base-4096', 'google/bigbird-roberta-base']
    for model in MODELS:
        score_dict = {'train': {'micro': [], 'macro': []},
                      'dev': {'micro': [], 'macro': []},
                      'test': {'micro': [], 'macro': []}}

        for seed in range(1, 6):
            seed = f'seed_{seed}'
            try:
                with open(os.path.join(BASE_DIR, model, seed, 'all_results.json')) as json_file:
                    json_data = json.load(json_file)
                    val = float(json_data['predict_micro-f1'])
                    score_dict['dev']['micro'].append(float(json_data['eval_micro-f1']))
                    score_dict['dev']['macro'].append(float(json_data['eval_macro-f1']))
                    score_dict['test']['micro'].append(float(json_data['predict_micro-f1']))
                    score_dict['test']['macro'].append(float(json_data['predict_macro-f1']))
            except:
                continue

        score_dicts[model] = score_dict

    print(f'{" " * 36} {"VALIDATION":<60} | {"TEST":<60}')
    print('-' * 200)
    for algo, stats in score_dicts.items():
        report_line = f'{algo:>35}: MICRO-F1: {np.mean(stats["dev"]["micro"])*100:.1f} ± {np.std(stats["dev"]["micro"])*100:.1f}\t'
        report_line += f'MACRO-F1: {np.mean(stats["dev"]["macro"])*100:.1f} ± {np.std(stats["dev"]["macro"])*100:.1f}\t'
        report_line += ' | '
        report_line += f'MICRO-F1: {np.mean(stats["test"]["micro"])*100:.1f} ± {np.std(stats["test"]["micro"])*100:.1f}\t'
        report_line += f'MACRO-F1: {np.mean(stats["test"]["macro"])*100:.1f} ± {np.std(stats["test"]["macro"])*100:.1f}\t'

        print(report_line)


if __name__ == '__main__':
    main()
