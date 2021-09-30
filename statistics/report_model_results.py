import json
import os
import argparse


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model',  default='roberta-base')
    config = parser.parse_args()

    MODEL = config.model
    TASKS = ['ecthr_a', 'ecthr_b', 'scotus', 'eurlex', 'ledgar', 'unfair_tos', 'case_hold']
    for task in TASKS:
        print('-' * 100)
        print(task.upper())
        print('-' * 100)
        BASE_DIR = f'logs/{task}'
        print(f'{" " * 10}   | {"VALIDATION":<40} | {"TEST":<40}')
        print('-' * 100)
        for seed in range(1, 6):
            seed = f'seed_{seed}'
            try:
                with open(os.path.join(BASE_DIR, MODEL, seed, 'all_results.json')) as json_file:
                    json_data = json.load(json_file)
                    dev_micro_f1 = float(json_data['eval_micro-f1'])
                    dev_macro_f1 = float(json_data['eval_macro-f1'])
                    test_micro_f1 = float(json_data['predict_micro-f1'])
                    test_macro_f1 = float(json_data['predict_macro-f1'])
                    epoch = float(json_data['epoch'])
                report_line = f'EPOCH: {epoch: 2.1f} | '
                report_line += f'MICRO-F1: {dev_micro_f1 * 100:.1f}\t'
                report_line += f'MACRO-F1: {dev_macro_f1 * 100:.1f}\t'
                report_line += ' | '
                report_line += f'MICRO-F1: {test_micro_f1 * 100:.1f}\t'
                report_line += f'MACRO-F1: {test_macro_f1 * 100:.1f}\t'
                print(report_line)
            except:
                continue


if __name__ == '__main__':
    main()