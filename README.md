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
*2022. In the Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics. Dublin, Ireland.*](https://arxiv.org/abs/2110.00976)
```
@inproceedings{chalkidis-etal-2021-lexglue,
        title={LexGLUE: A Benchmark Dataset for Legal Language Understanding in English}, 
        author={Chalkidis, Ilias and Jana, Abhik and Hartung, Dirk and
        Bommarito, Michael and Androutsopoulos, Ion and Katz, Daniel Martin and
        Aletras, Nikolaos},
        year={2022},
        booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
        address={Dubln, Ireland},
}
```


## Supported Tasks


<table>
        <tr><td><b>Dataset</b></td><td><b>Source</b></td><td><b>Sub-domain</b></td><td><b>Task Type</b></td><td><b>Train/Dev/Test Instances</b></td><td><b>Classes</b></td><tr>
<tr><td>ECtHR (Task A)</td><td> <a href="https://aclanthology.org/P19-1424/">Chalkidis et al. (2019)</a> </td><td>ECHR</td><td>Multi-label classification</td><td>9,000/1,000/1,000</td><td>10+1</td></tr>
<tr><td>ECtHR (Task B)</td><td> <a href="https://aclanthology.org/2021.naacl-main.22/">Chalkidis et al. (2021a)</a> </td><td>ECHR</td><td>Multi-label classification </td><td>9,000/1,000/1,000</td><td>10+1</td></tr>
<tr><td>SCOTUS</td><td> <a href="http://scdb.wustl.edu">Spaeth et al. (2020)</a></td><td>US Law</td><td>Multi-class classification</td><td>5,000/1,400/1,400</td><td>14</td></tr>
<tr><td>EUR-LEX</td><td> <a href="https://arxiv.org/abs/2109.00904">Chalkidis et al. (2021b)</a></td><td>EU Law</td><td>Multi-label classification</td><td>55,000/5,000/5,000</td><td>100</td></tr>
<tr><td>LEDGAR</td><td> <a href="https://aclanthology.org/2020.lrec-1.155/">Tuggener et al. (2020)</a></td><td>Contracts</td><td>Multi-class classification</td><td>60,000/10,000/10,000</td><td>100</td></tr>
<tr><td>UNFAIR-ToS</td><td><a href="https://arxiv.org/abs/1805.01217"> Lippi et al. (2019)</a></td><td>Contracts</td><td>Multi-label classification</td><td>5,532/2,275/1,607</td><td>8+1</td></tr>
<tr><td>CaseHOLD</td><td><a href="https://arxiv.org/abs/2104.08671">Zheng et al. (2021)</a></td><td>US Law</td><td>Multiple choice QA</td><td>45,000/3,900/3,900</td><td>n/a</td></tr>
</table>

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

### Averaged LexGLUE Scores

We report the arithmetic, harmonic, and geometric mean across tasks following [Shavrina and Malykh (2021)](https://openreview.net/pdf?id=PPGfoNJnLKd). We acknowledge that the use of scores aggregated over tasks has been criticized in general NLU benchmarks (e.g., GLUE), as models are trained with different numbers of samples, task complexity, and evaluation metrics per task. We believe that the use of a standard common metric (F1) across tasks and averaging with harmonic mean alleviate this issue.

<table>
<tr><td><b>Averaging</b></td><td><b>Arithmetic</b></td><td><b>Harmonic</b></td><td><b>Geometric</b></td></tr>
<tr><td><b>Model</b></td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1 </td></tr>
<tr><td>BERT</td><td> 77.8 /  69.5 </td><td> 76.7 /  68.2 </td><td> 77.2 /  68.8 </td></tr>
<tr><td>RoBERTa</td><td> 77.8 /  68.7 </td><td> 76.8 /  67.5 </td><td> 77.3 /  68.1 </td></tr>
<tr><td>RoBERTa (Large)</td><td> 79.4 /  70.8 </td><td> 78.4 /  69.1 </td><td> 78.9 /  70.0 </td></tr>
<tr><td>DeBERTa</td><td> 78.3 /  69.7 </td><td> 77.4 /  68.5 </td><td> 77.8 /  69.1 </td></tr>
<tr><td>Longformer</td><td> 78.5 /  70.5 </td><td> 77.5 /  69.5 </td><td> 78.0 /  70.0 </td></tr>
<tr><td>BigBird</td><td> 78.2 /  69.6 </td><td> 77.2 /  68.5 </td><td> 77.7 /  69.0 </td></tr>
<tr><td>Legal-BERT</td><td> 79.8 /  72.0 </td><td> 78.9 /  70.8 </td><td> 79.3 /  71.4 </td></tr>
<tr><td>CaseLaw-BERT</td><td> 79.4 /  70.9 </td><td> 78.5 /  69.7 </td><td> 78.9 /  70.3 </td></tr>
</table>

### Task-wise LexGLUE scores

#### Large-sized (:older_man:) Models [1]

<table>
        <tr><td><b>Dataset</b></td><td><b>ECtHR A</b></td><td><b>ECtHR B</b></td><td><b>SCOTUS</b></td><td><b>EUR-LEX</b></td><td><b>LEDGAR</b></td><td><b>UNFAIR-ToS</b></td><td><b>CaseHOLD</b></td></tr>
<tr><td><b>Model</b></td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1</td><td>μ-F1 / m-F1  </td></tr>
<tr><td>RoBERTa</td> <td> 73.8 /  67.6 </td> <td> 79.8 /  71.6 </td> <td> 67.9 /  50.3 </td> <td> 75.5 /  66.3 </td> <td> 88.6 /  83.6 </td> <td> 95.8 /  81.6 </td> <td> 74.4 </td> </tr>
</table>

[1] Results reported by [Chalkidis et al. (2021)](https://arxiv.org/abs/2110.00976). All large-sized transformer-based models follow the same specifications (L=24, H=1024, A=18).

#### Medium-sized (:man:) Models [2]

<table>
        <tr><td><b>Dataset</b></td><td><b>ECtHR A</b></td><td><b>ECtHR B</b></td><td><b>SCOTUS</b></td><td><b>EUR-LEX</b></td><td><b>LEDGAR</b></td><td><b>UNFAIR-ToS</b></td><td><b>CaseHOLD</b></td></tr>
<tr><td><b>Model</b></td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1</td><td>μ-F1 / m-F1  </td></tr>
<tr><td>TFIDF+SVM</td><td> 64.7 / 51.7  </td><td>74.6 / 65.1 </td><td> <b>78.2</b> / <b>69.5</b> </td><td>71.3  / 51.4 </td><td>87.2   / 82.4 </td><td>95.4  / 78.8</td><td>n/a   </td></tr>
        <tr><td>BERT</td> <td> <b>71.2</b> /  63.6 </td> <td> 79.7 /  73.4 </td> <td> 71.4 /  57.2 </td> <td> 68.3 /  58.3 </td> <td> 87.6 /  81.8 </td> <td> 95.6 /  81.3 </td> <td> 70.8 </td> </tr>
<tr><td>RoBERTa</td> <td> 69.2 /  59.0 </td> <td> 77.3 /  68.9 </td> <td> 71.9 /  <b>57.9</b> </td> <td> 71.6 /  62.0 </td> <td> 87.9 /  82.3 </td> <td> 95.2 /  79.2 </td> <td> 71.4 </td> </tr>
<tr><td>DeBERTa</td> <td> 70.0 /  60.8 </td> <td> 78.8 /  71.0 </td> <td> <b>72.1</b> /  57.4 </td> <td> 71.1 /  62.7 </td> <td> 88.2 /  <b>83.1</b> </td> <td> 95.5 /  80.3 </td> <td> 72.6 </td> </tr>
<tr><td>Longformer</td> <td> 69.9 /  <b>64.7</b> </td> <td> 79.4 /  71.7 </td> <td> 71.6 /  57.7 </td> <td> 72.9 /  64.0 </td> <td> 88.2 /  83.0 </td> <td> 95.5 /  80.9 </td> <td> 71.9 </td> </tr>
<tr><td>BigBird</td> <td> 70.0 /  62.9 </td> <td> 78.8 /  70.9 </td> <td> 71.5 /  56.8 </td> <td> 72.8 /  62.0 </td> <td> 87.8 /  82.6 </td> <td> 95.7 /  81.3 </td> <td> 70.8 </td> </tr>
<tr><td>Legal-BERT</td> <td> 70.0 /  64.0 </td> <td> <b>80.4</b> /  <b>74.7</b> </td> <td> <b>72.1</b> /  57.4 </td> <td> 76.4 /  66.5 </td> <td> 88.2 /  83.0 </td> <td> <b>96.0</b> /  <b>83.0</b> </td> <td> 75.3 </td> </tr>
<tr><td>CaseLaw-BERT</td> <td> 69.8 /  62.9 </td> <td> 78.8 /  70.3 </td> <td> 70.7 /  56.6 </td> <td> 76.6 /  65.9 </td> <td> <b>88.3</b> /  83.0 </td> <td> <b>96.0</b> /  82.3 </td> <td> <b>75.4</b> </td> </tr>
</table>

[2] Results reported by [Chalkidis et al. (2021)](https://arxiv.org/abs/2110.00976). All medium-sized transformer-based models follow the same specifications (L=12, H=768, A=12).

#### Small-sized (:baby:) Models [3]

<table>
<tr><td><b>Dataset</b></td><td><b>ECtHR A</b></td><td><b>ECtHR B</b></td><td><b>SCOTUS</b></td><td><b>EUR-LEX</b></td><td><b>LEDGAR</b></td><td><b>UNFAIR-ToS</b></td><td><b>CaseHOLD</b></td></tr>
        <tr><td><b>Model</b></td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1 </td><td>μ-F1  / m-F1</td><td>μ-F1 / m-F1  </td></tr>
<tr><td>BERT-Tiny</td><td>n/a</td><td>n/a</td><td>	62.8 / 40.9</td><td>	65.5 / 27.5</td><td>	83.9 / 74.7</td><td>	94.3 / 11.1</td><td>	68.3</td></tr>
<tr><td>Mini-LM (v2)</td><td>n/a</td><td>n/a</td><td>	60.8 / 45.5</td><td>	62.2 / 35.6</td><td>	86.7 / 79.6</td><td>	93.9 / 13.2</td><td>	71.3</td></tr>
<tr><td>Distil-BERT</td><td>n/a</td><td>n/a</td><td>	67.0 / 55.9</td><td>	66.0 / 51.5</td><td>	87.5 / <b>81.5</b></td><td>	<b>97.1</b> / <b>79.4</b></td><td>	68.6</td>
<tr><td>Legal-BERT </td><td>n/a</b></td><td>n/a</td><td><b>75.6</b> / <b>68.5</b></td><td> <b>73.4</b> / <b>54.4</b><td><b>87.8</b> /81.4</td><td><b>97.1</b> / 76.3</td><td><b>74.7</b></td></tr>

</table>

[3] Results reported by Atreya Shankar ([@atreyasha](https://github.com/atreyasha)) :hugs: :partying_face:. More details (e.g., validation scores, log files) are provided [here](https://github.com/coastalcph/lex-glue/discussions/categories/new-results). The small-sized models' specifications are:

* BERT-Tiny (L=2, H=128, A=2) by [Turc et al. (2020)](https://arxiv.org/abs/1908.08962)
* Mini-LM (v2) (L=12, H=386, A=12) by [Wang et al. (2020)](https://arxiv.org/abs/2002.10957)
* Distil-BERT (L=6, H=768, A=12) by [Sanh et al. (2019)](https://arxiv.org/abs/1910.01108)
* Legal-BERT (L=6, H=512, A=8) by [Chalkidis et al. (2020)](https://arxiv.org/abs/2010.02559)


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

, or non transformer-based neural models, like:

- LSTM-based [(Hochreiter and Schmidhuber, 1997)](https://dl.acm.org/doi/10.1162/neco.1997.9.8.1735) models, like the Hierarchical Attention Network (HAN) of [Yang et al. (2016)](https://aclanthology.org/N16-1174/),
- Graph-based models, like the Graph Attention Network (GAT) of [Veličković et al. (2017)](https://arxiv.org/abs/1710.10903)

, or even non neural models, like:

- Bag of Word (BoW) models using TF-IDF representations (e.g., SVM, Random Forest),
- The eXtreme Gradient Boosting (XGBoost) of [Chen and Guestrin (2016)](http://arxiv.org/abs/1603.02754),

and report back the results. We are curious!

### How to participate?

We are currently still lacking some technical infrastructure, e.g., an integrated submission environment comprised of an automated evaluation and an automatically updated leaderboard. We plan to develop the necessary publicly available web infrastructure extend the public infrastructure of LexGLUE in the near future. 

In the mean-time, we ask participants to re-use and expand our code to submit new results, if possible, and open a new discussion (submission) in our repository (https://github.com/coastalcph/lex-glue/discussions/new?category=new-results) presenting their results, providing the auto-generated result logs and the relevant publication (or pre-print), if available, accompanied with a pull request including the code amendments that are needed to reproduce their experiments. Upon reviewing your results, we'll update the public leaderboard accordingly.

### I want to re-load fine-tuned HierBERT models. How can I do this?

You can re-load fine-tuned HierBERT models following our example python script ["Re-load HierBERT models"](https://github.com/coastalcph/lex-glue/blob/main/utils/load_hierbert.py).

### I still have open questions...

Please post your question on [Discussions](https://github.com/coastalcph/lex-glue/discussions) section or communicate with the corresponding author via e-mail.
