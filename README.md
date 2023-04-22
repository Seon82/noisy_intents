# noisy_intents
[![Linter Actions Status](https://github.com/Seon82/noisy_intents/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/Seon82/noisy_intents/actions)

The `noisy_intents` package offers tools to reproduce allows to reproduce the work presented in *Investigating the impact of ASR transcription errors on intent classification performance*, and contains the data for the Noisy Daily Dialog Acts dataset.

**Abstract:**
>*Intent classification is a key component in most Spoken Language Understanding (SLU) pipelines. These  pipelines typically rely on two decoupled models, with an Automatic Speech Recognition (ASR) system providing textual hypotheses for the user’s voice signal, and a Natural Language Understanding (NLU) system ingesting them. 
We show that errors in these ASR hypotheses can substantially degrade performance, even for powerful pre-trained intent classification models.
This work also introduces a new evaluation dataset, Noisy Daily Dialog Acts (NoDA). This dataset, based on the Daily Dialog Act Corpus, allows to evaluate an NLU model's robustness to upstream ASR errors.*

## Installation
This package requires `python>=3.10`.

* Clone the repository.
* Run `pip install .` from the root of the project to install the package and its dependencies.

## Quickstart
Measure a model's performance on the NoDA dataset:
```python
from transformers import BertTokenizer
from torchmetrics import Accuracy
from torch.utils.data import DataLoader

from noisy_intents.data import NoDA
from noisy_intents.eval import compute_metrics, autodetect_device
from noisy_intents.models import BERT


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# The model below would need to have its classification head
# trained to do any real inference
model = BERT(num_classes=4, dropout=0.1, load_pretrained=True)

device = autodetect_device() # Automatically detect CUDA and Metal
model.to(device)

# Load the NoDA dataset
noda = NoDA(tokenizer, max_len=256)
noda_loader = DataLoader(noda, batch_size=64)

# Measure the micro-accuracy
compute_metrics(
    model, 
    noda_loader, 
    device, 
    metrics=[Accuracy("multiclass", num_classes=4, average="micro")],
    show_progress=True,
    )
```

Detailed example notebooks on how to train a model on DyDA and evaluate it on NoDA can be found in [examples](./examples/). 
## Contributing
To contribute code to the repository:

* Install [poetry](https://python-poetry.org/docs/#installation), our dependency management tool.
* Clone the repository.
* Install the project and its dependencies: `poetry install`.
  
To add dependencies to the project, use `poetry add` (for example `poetry add numpy`). Your code should be formatted using `black` and `isort`, and linted with `ruff` before every commit.


