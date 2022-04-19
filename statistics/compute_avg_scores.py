import copy
import json
import os
import argparse
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--dataset',  default='scotus')
    parser.add_argument('--filter_outliers', default=True)
    parser.add_argument('--top_k', default=3)
    config = parser.parse_args()

    BASE_DIR = f'logs/{config.dataset}'

    if os.path.exists(BASE_DIR):
        print(f'{BASE_DIR} exists!')

    score_dicts = {}
    MODELS = ['bert-base-uncased', 'roberta-base', 'microsoft/deberta-base', 'allenai/longformer-base-4096',
              'google/bigbird-roberta-base', 'nlpaueb/legal-bert-base-uncased', 'zlucia/custom-legalbert', 'roberta-large']

    for model in MODELS:
        score_dict = {'dev': {'micro': [], 'macro': []},
                      'test': {'micro': [], 'macro': []}}

        for seed in range(1, 6):
            seed = f'seed_{seed}'
            try:
                with open(os.path.join(BASE_DIR, model, seed, 'all_results.json')) as json_file:
                    json_data = json.load(json_file)
                    score_dict['dev']['micro'].append(float(json_data['eval_micro-f1']))
                    score_dict['dev']['macro'].append(float(json_data['eval_macro-f1']))
                    score_dict['test']['micro'].append(float(json_data['predict_micro-f1']))
                    score_dict['test']['macro'].append(float(json_data['predict_macro-f1']))
            except:
                continue

        score_dicts[model] = score_dict

    print(f'{" " * 36} {"VALIDATION":<47} | {"TEST"}')
    print('-' * 200)
    for algo, stats in score_dicts.items():
        temp_stats = copy.deepcopy(stats)
        if config.filter_outliers:
            seed_scores = [(idx, score) for (idx, score) in enumerate(stats['dev']['macro'])]
            sorted_scores = sorted(seed_scores, key=lambda tup: tup[1], reverse=True)
            top_k_ids = [idx for idx, score in sorted_scores[:config.top_k]]
            temp_stats['dev']['micro'] = [score for idx, score in enumerate(stats['dev']['micro']) if
                                           idx in top_k_ids]
            temp_stats['dev']['macro'] = [score for idx, score in enumerate(stats['dev']['macro']) if
                                           idx in top_k_ids]
            temp_stats['test']['micro'] = [score for idx, score in enumerate(stats['test']['micro']) if
                                           idx in top_k_ids[:1]]
            temp_stats['test']['macro'] = [score for idx, score in enumerate(stats['test']['macro']) if
                                           idx in top_k_ids[:1]]

        report_line = f'{algo:>35}: MICRO-F1: {np.mean(temp_stats["dev"]["micro"])*100:.1f}\t ± {np.std(temp_stats["dev"]["micro"])*100:.1f}\t'
        report_line += f'MACRO-F1: {np.mean(temp_stats["dev"]["macro"])*100:.1f}\t ± {np.std(temp_stats["dev"]["macro"])*100:.1f}\t'
        report_line += ' | '
        report_line += f'MICRO-F1: {np.mean(temp_stats["test"]["micro"])*100:.1f}\t'
        report_line += f'MACRO-F1: {np.mean(temp_stats["test"]["macro"])*100:.1f}\t'

        print(report_line)


if __name__ == '__main__':
    main()
