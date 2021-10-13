# LexGLUE: A Benchmark Dataset for Legal Language Understanding in English :balance_scale: :trophy: :student: :woman_judge:

![LexGLUE Graphic](https://repository-images.githubusercontent.com/411072132/5c49b313-ab36-4391-b785-40d9478d0f73)

## Dataset Summary

Inspired by the recent widespread use of the GLUE multi-task benchmark NLP dataset ([Wang et al., 2018](https://aclanthology.org/W18-5446/)), the subsequent more difficult SuperGLUE ([Wang et al., 2109](https://openreview.net/forum?id=rJ4km2R5t7)), other previous multi-task NLP benchmarks ([Conneau and Kiela,2018](https://aclanthology.org/L18-1269/); [McCann et al., 2018](https://arxiv.org/abs/1806.08730)), and similar initiatives in other domains ([Peng et al., 2019](https://arxiv.org/abs/1906.05474)), we introduce LexGLUE, a benchmark dataset to evaluate the performance of NLP methods in legal tasks. LexGLUE is based on seven existing legal NLP datasets, selected using criteria largely from SuperGLUE.

We anticipate that more datasets, tasks, and languages will be added in later versions of LexGLUE. As more legal NLP datasets become available, we also plan to favor datasets checked thoroughly for validity (scores reflecting real-life performance), annotation quality, statistical power,and social bias ([Bowman and Dahl, 2021](https://aclanthology.org/2021.naacl-main.385/)).

As in GLUE and SuperGLUE ([Wang et al., 2109](https://openreview.net/forum?id=rJ4km2R5t7)) one of our goals is to push towards generic (or *foundation*) models that can cope with multiple NLP tasks, in our case legal NLP tasks,possibly with limited task-specific fine-tuning. An-other goal is to provide a convenient and informative entry point for NLP researchers and practitioners wishing to explore or develop methods for legalNLP. Having these goals in mind, the datasets we include in LexGLUE and the tasks they address have been simplified in several ways, discussed below, to make it easier for newcomers and generic models to address all tasks. We provide PythonAPIs integrated with Hugging Face (Wolf et al.,2020; Lhoest et al., 2021) to easily import all the datasets, experiment with and evaluate their performance.

By unifying and facilitating the access to a set of law-related datasets and tasks, we hope to attract not only more NLP experts, but also more interdisciplinary researchers (e.g., law doctoral students willing to take NLP courses). More broadly, we hope LexGLUE will speed up the adoption and transparent evaluation of new legal NLP methods and approaches in the commercial sector too. Indeed, there have been many commercial press releases in legal-tech industry, but almost no independent evaluation of the veracity of the performance of various machine learning and NLP-based offerings. A standard publicly available benchmark would also allay concerns of undue influence in predictive models, including the use of metadata which the relevant law expressly disregards.

If you participate, use the LexGLUE benchmark, or our experimentation library, please cite:

[*Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion Androutsopoulos, Daniel Martin Katz, and Nikolaos Aletras.*
*LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.*
*2021. arXiv: 2110.00976.*](https://arxiv.org/abs/2110.00976)
```
@article{chalkidis-etal-2021-lexglue,
        title={LexGLUE: A Benchmark Dataset for Legal Language Understanding in English}, 
        author={Chalkidis, Ilias and Jana, Abhik and Hartung, Dirk and
        Bommarito, Michael and Androutsopoulos, Ion and Katz, Daniel Martin and
        Aletras, Nikolaos},
        year={2021},
        eprint={2110.00976},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        note = {arXiv: 2110.00976},
}
```


## Supported Tasks

| Dataset | Source | Sub-domain | Task Type | Training/Dev/Test Instances | Classes |
| --- | --- | --- | --- | --- | --- |
| ECtHR (Task A) | [Chalkidis et al. (2019)](https://aclanthology.org/P19-1424/) | ECHR | Multi-label classification | 9,000/1,000/1,000 | 10+1 |
| ECtHR (Task B) | [Chalkidis et al. (2021a)](https://aclanthology.org/2021.naacl-main.22/)  | ECHR | Multi-label classification | 9,000/1,000/1,000 | 10+1 | 
| SCOTUS | [Spaeth et al. (2020)](http://scdb.wustl.edu) | US Law | Multi-class classification | 5,000/1,400/1,400 | 14 | 
| EUR-LEX | [Chalkidis et al. (2021b)](https://arxiv.org/abs/2109.00904) | EU Law | Multi-label classification | 55,000/5,000/5,000 | 100 |
| LEDGAR | [Tuggener et al. (2020)](https://aclanthology.org/2020.lrec-1.155/) | Contracts | Multi-class classification | 60,000/10,000/10,000 | 100 |
| UNFAIR-ToS | [Lippi et al. (2019)](https://arxiv.org/abs/1805.01217) | Contracts | Multi-label classification | 5,532/2,275/1,607 | 8+1 |
| CaseHOLD | [Zheng et al. (2021)](https://arxiv.org/abs/2104.08671) | US Law | Multiple choice QA | 45,000/3,900/3,900 | n/a |


### ECtHR (Task A)

The European Court of Human Rights (ECtHR) hears allegations that a state has breached human rights provisions of the European Convention of Human Rights (ECHR). For each case, the dataset provides a list of factual paragraphs (facts) from the case description. Each case is mapped to articles of the ECHR that were violated (if any).

### ECtHR (Task B)

The European Court of Human Rights (ECtHR) hears allegations that a state has breached human rights provisions of the European Convention of Human Rights (ECHR). For each case, the dataset provides a list of factual paragraphs (facts) from the case description. Each case is mapped to articles of ECHR that were allegedly violated (considered by the court).

### SCOTUS

The US Supreme Court (SCOTUS) is the highest federal court in the United States of America and generally hears only the most controversial or otherwise complex cases which have not been sufficiently well solved by lower courts. This is a single-label multi-class classification task, where given a document (court opinion), the task is to predict the relevant issue areas. The 14 issue areas cluster 278 issues whose focus is on the subject matter of the controversy (dispute).

### EUR-LEX

European Union (EU) legislation is published in EUR-Lex portal. All EU laws are annotated by EU's Publications Office with multiple concepts from the EuroVoc thesaurus, a multilingual thesaurus maintained by the Publications Office. The current version of EuroVoc contains more than 7k concepts referring to various activities of the EU and its Member States (e.g., economics, health-care, trade). Given a document, the task is to predict its EuroVoc labels (concepts).

### LEDGAR

LEDGAR dataset aims contract provision (paragraph) classification. The contract provisions come from contracts obtained from the US Securities and Exchange Commission (SEC) filings, which are publicly available from EDGAR. Each label represents the single main topic (theme) of the corresponding contract provision.

### UNFAIR-ToS

The UNFAIR-ToS dataset contains 50 Terms of Service (ToS) from on-line platforms (e.g., YouTube, Ebay, Facebook, etc.). The dataset has been annotated on the sentence-level with 8 types of unfair contractual terms (sentences), meaning terms that potentially violate user rights according to the European consumer law.

### CaseHOLD

The CaseHOLD (Case Holdings on Legal Decisions) dataset includes multiple choice questions about holdings of US court cases from the Harvard Law Library case law corpus. Holdings are short summaries of legal rulings accompany referenced decisions relevant for the present case. The input consists of an excerpt (or prompt) from a court decision, containing a reference to a particular case, while the holding statement is masked out. The model must identify the correct (masked) holding statement from a selection of five choices.

## Leaderboard

| Dataset | ECtHR Task A  | ECtHR Task B  | SCOTUS  | EUR-LEX | LEDGAR  | UNFAIR-ToS  | CaseHOLD |
| --- | ---- | --- | --- | --- | --- | --- | --- |
| Model | μ-F1  / m-F1  | μ-F1  / m-F1  | μ-F1  / m-F1  | μ-F1  / m-F1  | μ-F1  / m-F1  | μ-F1  / m-F1 | μ-F1 / m-F1   | 
|  BERT ([Devlin et al., 2018](https://arxiv.org/abs/1810.04805))  | **71.4**  / 64.0   | 79.6  / **78.3**  | 70.5   / 60.9  | 71.6  / 55.6  | 87.7   / 82.2  | 97.3  / 80.4 | 70.7    | 
|  RoBERTa ([Liu et al., 2019](https://arxiv.org/abs/1907.11692)) | 69.5  / 60.7  | 78.6  / 77.0  | 70.8   / 61.2  | 71.8  / **57.5**  | 87.9  /  82.1  | 97.2 / 79.6 | 71.7  | 
|  DeBERTa ([He et al., 2021](https://arxiv.org/abs/2006.03654)) | 69.1   / 61.2  | 79.9   / **78.3**  | 70.0  / 60.0  | **72.3**  / 57.2  | 87.9   / 82.0  | 97.2 / 80.2 | 72.1   | 
|  Longformer ([Beltagy et al., 2020](https://arxiv.org/abs/2004.05150)) | 69.6  / 62.4  | 78.8  / 75.8  | 72.2  / 62.5  | 71.9  / 56.7  | 87.7  / 82.3  | **97.5** / 81.0 | 72.0   | 
|  BigBird ([Zaheer et al., 2021](https://arxiv.org/abs/2007.14062)) | 70.5  / 63.8  | 79.9  / 76.9  | 71.7  / 61.4  | 71.8  / 56.6  | 87.7 / 82.1  | 97.4 / 81.1 | 70.4   | 
|  Legal-BERT ([Chalkidis et al., 2020](https://aclanthology.org/2020.findings-emnlp.261/)) | 71.2  / **64.6**  | **80.6**  / 77.2  | 76.2  / 65.8  | 72.2  / 56.2  | **88.1**  / **82.7** | 97.4  / **83.4** | 75.1 | 
|  CaseLaw-BERT ([Zheng et al., 2021](https://arxiv.org/abs/2104.08671)) | 71.2   / 64.2  | 88.0   / 77.5  | 79.7 / 76.8  | 71.0  / 55.9  | 88.0  / 82.3 | 97.4  / 82.4 | **75.6**   | 

## Frequently Asked Questions (FAQ)

### Where are the datasets?

We provide access to LexGLUE on [Hugging Face Datasets](https://huggingface.co/datasets) (Lhoest et al., 2021) at https://huggingface.co/datasets/lex_glue.  

For example to load the SCOTUS [Spaeth et al. (2020)](http://scdb.wustl.edu) dataset, you first simply install the datasets python library and then make the following call:

```python

from datasets import load_dataset 
dataset = load_dataset("lex_glue", "scotus")

```

### How to run experiments?

Furthermore, to make reproducing the results for the already examined models or future models even easier, we release our code in this repository. In folder `/experiments`, there are Python scripts, relying on the [Hugging Face Transformers](https://huggingface.co/transformers/) library, to run and evaluate any Transformer-based model (e.g., BERT, RoBERTa, LegalBERT, and their hierarchical variants, as well as, Longforrmer, and BigBird). We also provide bash scripts in folder `/scripts` to replicate the experiments for each dataset with 5 randoms seeds, as we did for the reported results for the original leaderboard.

Make sure that all required packages are installed:

```
torch>=1.9.0
transformers>=4.9.0
scikit-learn>=0.24.1
tqdm>=4.61.1
numpy>=1.20.1
datasets>=1.12.1
nltk>=3.5
scipy>=1.6.3
```

For example to replicate the results for RoBERTa ([Liu et al., 2019](https://arxiv.org/abs/1907.11692)) on UNFAIR-ToS [Lippi et al. (2019)](https://arxiv.org/abs/1805.01217), you have to configure the relevant bash script (`run_unfair_tos.sh`):

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

### I don't have the resources to run all these Muppets. What can I do?

You can use Google Colab with GPU acceleration for free online (https://colab.research.google.com). 
- Set Up a new notebook (https://colab.research.google.com) and git clone the project.
- Navigate to Edit → Notebook Settings and select GPU from the Hardware Accelerator drop-down. You will probably get assigned with an NVIDIA Tesla K80 12GB.
- You will also have to decrease the batch size and increase the accumulation steps for hierarchical models.

But, this is an interesting open problem (Efficient NLP), please consider using lighter pre-trained (smaller/faster) models, like: 
- The smaller [Legal-BERT](https://huggingface.co/nlpaueb/legal-bert-small-uncased) of [Chalkidis et al. (2020)](https://arxiv.org/abs/2010.02559),
- Smaller [BERT](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) models of [Turc et al. (2020)](https://arxiv.org/abs/1908.08962),
- [Mini-LM](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased) of [Wang et al. (2020)](https://arxiv.org/abs/2002.10957),

and report back the results. We are curious!

### How to participate?

We are currently still lacking some technical infrastructure, e.g., an integrated submission environment comprised of an automated evaluation and an automatically updated leaderboard. We plan to develop the necessary publicly available web infrastructure extend the public infrastructure of LexGLUE in the near future. 

In the mean-time, we ask participants to re-use and expand our code to submit new results, if possible, and raise a new issue in our repository (https://github.com/coastalcph/lex-glue/issues/new) presenting their results, providing the auto-generated result logs and the relevant publication (or pre-print), if available, accompanied with a pull request including the code amendments that are needed to reproduce their experiments. Upon reviewing your results, we'll update the public leaderboard accordingly.

### I still have open questions...

Please post your question on [Discussions](https://github.com/coastalcph/lex-glue/discussions) section or communicate with the corresponding author via e-mail.
