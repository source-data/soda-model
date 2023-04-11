<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/soda-model.svg?branch=main)](https://cirrus-ci.com/github/<USER>/soda-model)
[![ReadTheDocs](https://readthedocs.org/projects/soda-model/badge/?version=latest)](https://soda-model.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/soda-model/main.svg)](https://coveralls.io/r/<USER>/soda-model)
[![PyPI-Server](https://img.shields.io/pypi/v/soda-model.svg)](https://pypi.org/project/soda-model/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/soda-model.svg)](https://anaconda.org/conda-forge/soda-model)
[![Monthly Downloads](https://pepy.tech/badge/soda-model/month)](https://pepy.tech/project/soda-model)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/soda-model)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# SODA-MODEL: NLP models obtained with the SourceData data set

This repository has the AI models built around the [SODA-data](https://github.com/source-data/soda-data) datasets.

For transparency we offer bash scripts that will automatically run the experiments shown in the
papers related to this and the aford mentioned repository.
This scripts are located in the `scripts` folder of the repository.

Everything is ready to run on a linux machine with GPU availability using the provided
docker container.

Currently, we show the models for:

* Named-entity recognition
* Semantic identification of empirical roles of entities
* Panelization of figure captions

All these models can be found in the `token_classification` folder.

## Running the scripts

1. Clone the repository to your local computer

2. Generate a virtual encironment and activate it:

```bash
    python -m venv /path/to/new/virtual/environment

    source activate .venv/bin/activate
```

3. Make sure that you have a docker-compose version that supports versions 2.3 or higher:

```bash
    docker-compose --version

    >> docker-compose version 1.29.2, build unknown
```

4. Build the docker container and open it

```bash
    docker-compose build --force-rm --no-cache
    docker-compose up -d
    docker-compose exec nlp bash
```

5. From inside the container you can run any command.

```bash
    # Run all the available experiments
    sh scripts/run_all.sh

    # Run ner
    sh scripts/ner.sh

    # Run the panelization task
    sh scripts/panelization.sh

    # Run the semantic roles:
    sh scripts/geneprod.sh
    sh scripts/smallmol.sh
```

You can also run your own version of the models. Any model can be trained via bash
passing the [`TrainingArguments` as in HuggingFace](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments).
This means that the models can even be automatically uploaded to your own HuggingFace
account given a valid token is provided.

Example of how to train a NER model using this repository form inside the
```bash
    python -m soda_model.token_classification.trainer \
        --dataset_id "EMBO/SourceData" \
        --task NER \
        --version 1.0.0 \
        --from_pretrained microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
        --ner_labels all \
        --filter_empty \
        --max_length 512 \
        --max_steps 50 \
        --masking_probability 0.0 \
        --replacement_probability 0.0 \
        --classifier_dropout 0.2 \
        --do_train \
        --do_predict \
        --report_to none \
        --truncation \
        --padding "longest" \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 4 \
        --results_file "testing_metrics_for_gen_vs_memo_"
```

## Note

This project has been set up using PyScaffold 4.4. For details and usage
information on PyScaffold see https://pyscaffold.org/.
