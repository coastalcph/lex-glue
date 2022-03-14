import copy
import json
import os
import argparse
import numpy as np
from scipy.stats import hmean, gmean


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--filter_outliers', default=True)
    parser.add_argument('--top_k', default=1)
    config = parser.parse_args()

    MODELS = ['bert-base-uncased', 'roberta-base', 'microsoft/deberta-base', 'allenai/longformer-base-4096',
              'google/bigbird-roberta-base', 'nlpaueb/legal-bert-base-uncased', 'zlucia/custom-legalbert', 'roberta-large']
    DATASETS = ['ecthr_a', 'ecthr_b', 'eurlex', 'scotus', 'ledgar', 'unfair_tos', 'casehold']
    MODEL_NAMES = ['BERT', 'RoBERTa', 'DeBERTa', 'Longformer', 'BigBird', 'Legal-BERT', 'CaseLaw-BERT', 'RoBERTa']

    score_dicts = {model: {'dev': {'micro': [], 'macro': []}, 'test': {'micro': [], 'macro': []}}
                   for model in MODELS}

    for model in MODELS:
        for dataset in DATASETS:
            BASE_DIR = f'/Users/rwg642/Desktop/LEXGLUE/RESULTS/{dataset}'

            score_dict = {'dev': {'micro': [], 'macro': []},
                          'test': {'micro': [], 'macro': []}}

            for seed in range(1, 6):
                try:
                    seed = f'seed_{seed}'
                    with open(os.path.join(BASE_DIR, model, seed, 'all_results.json')) as json_file:
                        json_data = json.load(json_file)
                        score_dict['dev']['micro'].append(float(json_data['eval_micro-f1']))
                        score_dict['dev']['macro'].append(float(json_data['eval_macro-f1']))
                        score_dict['test']['micro'].append(float(json_data['predict_micro-f1']))
                        score_dict['test']['macro'].append(float(json_data['predict_macro-f1']))
                except:
                    continue
            temp_stats = copy.deepcopy(score_dict)
            if config.filter_outliers:
                seed_scores = [(idx, score) for (idx, score) in enumerate(score_dict['dev']['macro'])]
                sorted_scores = sorted(seed_scores, key=lambda tup: tup[1], reverse=True)
                top_k_ids = [idx for idx, score in sorted_scores[:config.top_k]]
                for subset in ['dev', 'test']:
                    temp_stats[subset]['micro'] = [score for idx, score in enumerate(score_dict[subset]['micro']) if
                                                   idx in top_k_ids]
                    temp_stats[subset]['macro'] = [score for idx, score in enumerate(score_dict[subset]['macro']) if
                                                   idx in top_k_ids]
            for subset in ['dev', 'test']:
                for avg in ['micro', 'macro']:
                    score_dicts[model][subset][avg].append(np.mean(temp_stats[subset][avg]))

    print('-' * 253)
    print(f'{"MODEL NAME":>35} | {"A-MEAN":<33} | {"H-MEAN":<33} | {"G-MEAN":<33} |')
    print('-' * 253)
    for idx, (method, stats) in enumerate(score_dicts.items()):
        algo_means = {'dev': {'micro': [0.0, 0.0, 0.0], 'macro': [0.0, 0.0, 0.0]},
                      'test': {'micro': [0.0, 0.0, 0.0], 'macro': [0.0, 0.0, 0.0]}}
        for subset in ['dev', 'test']:
            for avg in ['micro', 'macro']:
                algo_means[subset][avg][0] = np.mean(stats[subset][avg])
                algo_means[subset][avg][1] = hmean(stats[subset][avg])
                algo_means[subset][avg][2] = gmean(stats[subset][avg])
        report_line = f'<tr><td>{MODEL_NAMES[idx]}</td>'
        for task_idx in range(3):
            report_line += f'<td> {algo_means["test"]["micro"][task_idx] * 100:.1f} / '
            report_line += f' {algo_means["test"]["macro"][task_idx] * 100:.1f} </td>'
        report_line += '</tr>'

        print(report_line)


if __name__ == '__main__':
    main()
