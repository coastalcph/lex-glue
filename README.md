# LexGLUE: A Benchmark Dataset for Legal Language Understanding in English

## Dataset Summary

Inspired by the recent widespread use of the GLUE multi-task benchmark NLP dataset (Wang et al., 2018), the subsequent more difficult SuperGLUE (Wang et al., 2019), other previous multi-task NLP benchmarks (Conneau and Kiela,2018; McCann et al., 2018), and similar initiatives in other domains (Peng et al.,  2019), we introduce LexGLUE, a benchmark dataset to evaluate the performance of NLP methods in legal tasks. LexGLUE is based on seven existing legal NLP datasets.



*Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion Androutsopoulos, Daniel Martin Katz, and Nikolaos Aletras.*
*LexGLUE: A Benchmark Dataset for Legal Language Understanding in English*
*Arxiv Preprint. 2021*
```
@article{chalkidis-etal-2021-lexglue,
        title={LexGLUE: A Benchmark Dataset for Legal Language Understanding in English}, 
        author={Chalkidis, Ilias and Jana, Abhik and Hartung, Dirk and
        Bommarito, Michael and Androutsopoulos, Ion and Katz, Daniel Martin and
        Aletras, Nikolaos},
        year={2021},
        eprint={xxxx.xxxxx},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
}
```


## Supported Tasks

| Dataset | Source | Sub-domain | Task Type | Training/Dev/Test Instances | Classes |
| --- | --- | --- | --- | --- | --- |
| ECtHR (Task A) | [Chalkidis et al. (2019)](https://aclanthology.org/P19-1424/) | ECHR | Multi-label classification | 9,000/1,000/1,000 | 10+1 |
| ECtHR (Task B) | [Chalkidis et al. (2021a)](https://aclanthology.org/2021.naacl-main.22/)  | ECHR | Multi-label classification | 9,000/1,000/1,000 | 10 | 
| SCOTUS | [Spaeth et al. (2020)](http://scdb.wustl.edu) | US Law | Multi-class classification | 5,000/1,400/1,400 | 14 | 
| EUR-LEX | [Chalkidis et al. (2021b)](https://arxiv.org/abs/2109.00904) | EU Law | Multi-label classification | 55,000/5,000/5,000 | 100 |
| LEDGAR | [Tuggener et al. (2020)](https://aclanthology.org/2020.lrec-1.155/) | Contracts | Multi-class classification | 60,000/10,000/10,000 | 100 |
| UNFAIR-ToS | [Lippi et al. (2019)](https://arxiv.org/abs/1805.01217) | Contracts | Multi-label classification | 5,532/2,275/1,607 | 8 |
| CaseHOLD | [Zheng et al. (2021)](https://arxiv.org/abs/2104.08671) | US Law | Multiple choice QA | 45,000/3,900/3,900 | n/a |


## Leaderboard

| Dataset | ECtHR Task A  | ECtHR Task B  | SCOTUS  | EUR-LEX | LEDGAR  | UNFAIR-ToS  | CaseHOLD |
| --- | ---- | --- | --- | --- | --- | --- | --- |
| Model | μ-F1  / m-F1  | μ-F1  / m-F1  | μ-F1  / m-F1  | μ-F1  / m-F1  | μ-F1  / m-F1  | μ-F1  / m-F1 | μ-F1 / m-F1   | 
|  BERT  | **71.4**  / 64.0   | 87.6  / **77.8**  | 70.5   / 60.9  | 71.6  / 55.6  | 87.7   / 82.2  | 87.5  / 81.0 | 70.7    | 
|  RoBERTa  | 69.5  / 60.7  | 87.2  / 77.3  | 70.8   / 61.2  | 71.8  / **57.5**  | 87.9  /  82.1  | 87.7 / 81.5 | 71.7  | 
|  DeBERTa  | 69.1   / 61.2  | 87.4   / 77.3  | 70.0  / 60.0  | **72.3**  / 57.2  | 87.9   / 82.0  | 87.2 / 78.8 | 72.1   | 
|  Longformer  | 69.6  / 62.4  | 88.0  / **77.8**  | 72.2  / 62.5  | 71.9  / 56.7  | 87.7  / 82.3  | 87.7 / 80.1 | 72.0   | 
|  BigBird  | 70.5  / 63.8  | **88.1**  / 76.6  | 71.7  / 61.4  | 71.8  / 56.6  | 87.7 / 82.1  | 87.7 / 80.2 | 70.4   | 
|  Legal-BERT  | 71.2  / **64.6**  | 88.0  / 77.2  | 76.2  / 65.8  | 72.2  / 56.2  | **88.1**  / **82.7** | **88.6**  / **82.3** | 75.1 | 
|  CaseLaw-BERT  | 71.2   / 64.2  | 88.0   / 77.5  | **76.4**  / **66.2**  | 71.0  / 55.9  | 88.0  / 82.3 | 88.3  / 81.0 | **75.6**   | 

## Frequently Asked Questions

### Where are the datasets?

We provide access to LexGLUE on [Hugging Face Datasets](https://huggingface.co/datasets) (Lhoest et al., 2021) at https://huggingface.co/datasets/lex_glue.  

For example to load the SCOTUS dataset, you first simply install the datasets python library and then make the following call:

```python

from datasets import load_dataset 
dataset = load_dataset("lex_glue", task='scotus')

```

### How to run experiments?

Furthermore, to make reproducing the results for the already examined models or future models even easier, we release our code in this repository. In folder `/experiments`, there are Python scripts, relying on the [Hugging Face Transformers](https://huggingface.co/transformers/) library, to run and evaluate any Transformer-based model (e.g., BERT, RoBERTa, LegalBERT, and their hierarchical variants, as well as, Longforrmer, and BigBird). We also provide bash scripts to replicate the experiments for each dataset with 5 randoms seeds, as we did for the reported results for the original leaderboard.

For example to replicate the result of RoBERTa on UNFAIR-ToS, you have to configure the relevant bash script (`run_unfair_tos.sh`):

```
> nano run_unfair_tos.sh
GPU_NUMBER=1
MODEL_NAME='roberta-base'
LOWER_CASE='False'
BATCH_SIZE=8
ACCUMULATION_STEPS=1
TASK='unfair_tos'
```

and then run it:

```
> sh run_unfair_tos.sh
```

### How to participate?
To submit new results, we ask participants to reuse and expand our code, if possible, and raise a new issue in our repository (https://github.com/coastalcph/lex-glue/issues/new) presenting their results, providing the auto-generated result logs and the relevant publication (or pre-print), if available, accompanied with a pull request including the code amendments that are needed to reproduce their experiments. Upon reviewing your results, we'll update the public leaderboard accordingly.
