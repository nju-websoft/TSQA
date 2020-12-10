# TTGen-K-BERT

Sorce code for "TSQA: Tabular Scenario Based Question Answering", implement is based on [K-BERT](https://github.com/autoliuweijie/K-BERT). We thank the authors of K-BERT for proposing such a novel knowledge embedding model.


## Prepare

Download the ``BERT-wwm-ext`` from [here](https://github.com/ymcui/Chinese-BERT-wwm), and convert it to uer framework format by [uer](https://github.com/dbiir/UER-py), finally put the model under the directory ``./models``.


Split the GeoTSQA to 5-fold:
```shell script
python preprocess_data.py --all_data_path datasets/all.txt
```


## Sentence Ranking
Run template-level ranking of TTGen:
```shell script
python template_ranking.py --all_data_path datasets/all_template_ranking.txt --gpu 0
```

Run ranking of TTGen:
```shell script
python cross_val.py --nn TTGen --gpu 0,1
```

## QA
For multi-choice QA, we organize data in this format:
```text
[
  [
    [
      scenario text,
      table describing text,
    ],
    [
      {
        "question": question text,
        "choice": [
          option A,
          option B,
          option C,
          option D,
        ],
        "answer": one of choice
      }
    ],
    id
  ],
  ...
]
```
Download ``C3`` from [here](https://github.com/nlpdata/c3) and coarse-tune ``BERT-wwm-ext``:
```shell script
CUDA_VISIBLE_DEVICES=0 bash train_c3.sh
```

Run question answer on GeoTSQA, for example, when we select the top-1 from the ranked sentences set generated by TTGen:
```shell script
python cross_val_multi_choice.py --nn sentences_1 --data_path datasets/qa/sentences_1 --gpu 0,1
```