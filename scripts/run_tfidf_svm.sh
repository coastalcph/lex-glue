DATASET='eurlex'
TASK_TYPE='multi_label'
N_CLASSES=100

python models/tfidf_svm.py --dataset ${DATASET} --task_type ${TASK_TYPE} --n_classes ${N_CLASSES}