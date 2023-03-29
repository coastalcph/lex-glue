import pandas
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from datasets import load_dataset
import logging
import os
import argparse

dataset_n_classes = {'ecthr_a': 10, 'ecthr_b': 10, 'scotus': 14, 'eurlex': 100, 'ledgar': 100, 'unfair_tos': 8, 'case_hold': 5}


def main():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument('--dataset',  default='case_hold', type=str)
    parser.add_argument('--task_type', default='multi_class', type=str)
    parser.add_argument('--text_limit', default=-1, type=int)
    config = parser.parse_args()
    n_classes = dataset_n_classes[config.dataset]

    if not os.path.exists(f'logs/{config.dataset}'):
        if not os.path.exists(f'logs'):
            os.mkdir(f'logs')
        os.mkdir(f'logs/{config.dataset}')
    handlers = [logging.FileHandler(f'logs/{config.dataset}_svm.txt'), logging.StreamHandler()]
    logging.basicConfig(handlers=handlers, level=logging.INFO)

    def get_text(dataset):
        if 'ecthr' in config.dataset:
            texts = [' '.join(text) for text in dataset['text']]
            return [' '.join(text.split()[:config.text_limit]) for text in texts]
        elif config.dataset == 'case_hold':
            data = [[context] + endings for context, endings in zip(dataset['context'], dataset['endings'])]
            return pd.DataFrame(data=data,
                                columns=['context', 'option_1', 'option_2', 'option_3', 'options_4', 'option_5']
                                )
        else:
            return [' '.join(text.split()[:config.text_limit]) for text in dataset['text']]

    def get_labels(dataset, mlb=None):
        if config.task_type == 'multi_class':
            return dataset['label']
        else:
            return mlb.transform(dataset['labels']).tolist()

    def add_zero_class(labels):
        augmented_labels = np.zeros((len(labels), len(labels[0]) + 1), dtype=np.int32)
        augmented_labels[:, :-1] = labels
        augmented_labels[:, -1] = (np.sum(labels, axis=1) == 0).astype('int32')
        return augmented_labels

    scores = {'micro-f1': [], 'macro-f1': []}
    dataset = load_dataset('lex_glue', config.dataset)

    for seed in range(1, 6):
        if config.task_type == 'multi_label':
            classifier = OneVsRestClassifier(LinearSVC(random_state=seed, max_iter=50000))
            parameters = {
                'vect__max_features': [10000, 20000, 40000],
                'clf__estimator__C': [0.1, 1, 10],
                'clf__estimator__loss': ('hinge', 'squared_hinge')
            }
        elif config.dataset == 'case_hold':
            classifier = LinearSVC(random_state=seed, max_iter=50000)
            parameters = {
                'clf__C': [0.1, 1, 10],
                'clf__loss': ('hinge', 'squared_hinge')
            }
        else:
            classifier = LinearSVC(random_state=seed, max_iter=50000)
            parameters = {
                'vect__max_features': [10000, 20000, 40000],
                'clf__C': [0.1, 1, 10],
                'clf__loss': ('hinge', 'squared_hinge')
            }

        # Init Pipeline (TF-IDF, SVM)
        if config.dataset == 'case_hold':
            text_clf = Pipeline([
                ('union', FeatureUnion([('context_tfidf',
                                Pipeline([('extract_field', FunctionTransformer(lambda x: x['context'], validate=False)),
                                          ('vect', CountVectorizer(stop_words=stopwords.words('english'),
                                                                   ngram_range=(1, 3), min_df=5, max_features=40000)),
                                          ('tfidf', TfidfTransformer())]))] +
                             [(f'option_{idx}_tfidf',
                               Pipeline([('extract_field', FunctionTransformer(lambda x: x[f'option_{idx}'], validate=False)),
                                         ('vect', CountVectorizer(stop_words=stopwords.words('english'),
                                                                  ngram_range=(1, 3), min_df=5, max_features=40000)),
                                         ('tfidf', TfidfTransformer())]))
                              for idx in range(1, 6)]
                             )),
                ('clf', classifier)
            ])
        else:
            text_clf = Pipeline([('vect', CountVectorizer(stop_words=stopwords.words('english'),
                                                          ngram_range=(1, 3), min_df=5)),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', classifier),
                                 ])

        # Fixate Validation Split
        split_index = [-1] * len(dataset['train']) + [0] * len(dataset['validation'])
        val_split = PredefinedSplit(test_fold=split_index)
        gs_clf = GridSearchCV(text_clf, parameters, cv=val_split, n_jobs=32, verbose=4, refit = False)

        # Pre-process inputs, outputs
        x_train = get_text(dataset['train'])
        x_val = get_text(dataset['validation'])
        x_train_val = pd.concat([x_train, x_val])
        
        if config.task_type == 'multi_label':
            mlb = MultiLabelBinarizer(classes=range(n_classes))
            mlb.fit(dataset['train']['labels'])
        else:
            mlb = None
        y_train = get_labels(dataset['train'], mlb)
        y_val = get_labels(dataset['validation'], mlb)
        y_train_val = y_train + y_val

        # Train classifier
        gs_clf = gs_clf.fit(x_train_val, y_train_val)

        # Print best hyper-parameters
        logging.info('Best Parameters:')
        for param_name in sorted(parameters.keys()):
            logging.info("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
        
        # Retrain model with best CV parameters only with train data
        text_clf.set_params(**gs_clf.best_params_)
        gs_clf = text_clf.fit(x_train, y_train)
        
        # Report results
        logging.info('VALIDATION RESULTS:')
        y_pred = gs_clf.predict(get_text(dataset['validation']))
        y_true = get_labels(dataset["validation"], mlb)
        if config.task_type == 'multi_label' and config.dataset != 'eurlex':
            y_true = add_zero_class(y_true)
            y_pred = add_zero_class(y_pred)

        logging.info(f'Micro-F1: {metrics.f1_score(y_true, y_pred, average="micro")*100:.1f}')
        logging.info(f'Macro-F1: {metrics.f1_score(y_true, y_pred, average="macro")*100:.1f}')

        logging.info('TEST RESULTS:')
        y_pred = gs_clf.predict(get_text(dataset['test']))
        y_true = get_labels(dataset["test"], mlb)
        if config.task_type == 'multi_label' and config.dataset != 'eurlex':
            y_true = add_zero_class(y_true)
            y_pred = add_zero_class(y_pred)
        logging.info(f'Micro-F1: {metrics.f1_score(y_true, y_pred, average="micro")*100:.1f}')
        logging.info(f'Macro-F1: {metrics.f1_score(y_true, y_pred, average="macro")*100:.1f}')

        scores['micro-f1'].append(metrics.f1_score(y_true, y_pred, average="micro"))
        scores['macro-f1'].append(metrics.f1_score(y_true, y_pred, average="macro"))

    # Report averaged results across runs
    logging.info('-' * 100)
    logging.info(f'Micro-F1: {np.mean(scores["micro-f1"])*100:.1f} +/- {np.std(scores["micro-f1"])*100:.1f}\t'
                 f'Macro-F1: {np.mean(scores["macro-f1"])*100:.1f} +/- {np.std(scores["macro-f1"])*100:.1f}')


if __name__ == '__main__':
    main()
