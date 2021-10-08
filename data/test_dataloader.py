from datasets import load_dataset

dataset = load_dataset('lex_glue', 'ecthr_a', data_dir='data')
dataset = load_dataset('lex_glue', 'ecthr_b', data_dir='data')
dataset = load_dataset('lex_glue', 'scotus', data_dir='data')
dataset = load_dataset('lex_glue', 'eurlex', data_dir='data')
dataset = load_dataset('lex_glue', 'ledgar', data_dir='data')
dataset = load_dataset('lex_glue', 'case_hold', data_dir='data')
dataset = load_dataset('lex_glue', 'unfair_tos', data_dir='data')
