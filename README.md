<!---

    Copyright (c) 2019 Robert Bosch GmbH and its subsidiaries.

-->

# Joint-Anonymization-NER

This is the companion code for the experiments reported in the paper

> "Closing the Gap: Joint De-Identification and Concept Extraction in the Clinical Domain"  by Lukas Lange, Heike Adel and Jannik Strötgen published at ACL 2020.

The paper can be found [here](https://www.aclweb.org/anthology/2020.acl-main.621.pdf). The code allows the users to reproduce the results reported in the paper and extend the model to new datasets and embedding configurations. 
Please cite the above paper when reporting, reproducing or extending the results as:

## Citation

```
@inproceedings{lange-etal-2020-closing,
    title = "Closing the Gap: Joint De-Identification and Concept Extraction in the Clinical Domain",
    author = {Lange, Lukas  and
      Adel, Heike  and
      Str{\"o}tgen, Jannik},
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.621",
    pages = "6945--6952",
    abstract = "Exploiting natural language processing in the clinical domain requires de-identification, i.e., anonymization of personal information in texts. However, current research considers de-identification and downstream tasks, such as concept extraction, only in isolation and does not study the effects of de-identification on other tasks. In this paper, we close this gap by reporting concept extraction performance on automatically anonymized data and investigating joint models for de-identification and concept extraction. In particular, we propose a stacked model with restricted access to privacy sensitive information and a multitask model. We set the new state of the art on benchmark datasets in English (96.1{\%} F1 for de-identification and 88.9{\%} F1 for concept extraction) and Spanish (91.4{\%} F1 for concept extraction).",
}
```

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication "Closing the Gap: Joint De-Identification and Concept Extraction in the Clinical Domain". 
It will neither be maintained nor monitored in any way.

## Setup

* Install flair (Tested with flairNLP 0.4.5, PyTorch 1.3.1 and Python 3.6.8)
* Download pre-trained word embeddings (using flair or your own).
* Prepare corpus in BIO format.
* Train a [stacked](Spanish_Stacked_Model.ipynb) or multitask model as described in the example notebook

## Data

We do not ship the corpora used in the experiments from the paper. 
The sample files provided in the [data](data) directory are given to illustrate the used data format (BIO).
More information can be found in the [data/README.md](data/README.md). 

## License

Joint-Anonymization-NER is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in Joint-Anonymization-NER, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

The software including its dependencies may be covered by third party rights, including patents. You should not execute this code unless you have obtained the appropriate rights, which the authors are not purporting to give.
