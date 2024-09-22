# A Hands on Introduction to LLMs using distilBert

## Setup a Python Virtual Environment (Linux and Mac)

Open a terminal window and navigate to the home directory

```console
cd ~/
```

Create a python directory

```console
mkdir Python
```

Create virtual environment

```console
cd ~/Python
```

```console
python3 -m venv llm_env
```

Source and activate virtual environment

```console
source ~/Python/llm_env/bin/activate
```

Pip Install dependencies

```console
pip3 install transformers datasets torch
```

To train distilBERT<sup>[[1](#distilBERT)]</sup> on IMDB Hugging Face data, run [distilbert_imdb_train.py](./distilbert_imdb_train.py) as follow


```console
python3 distilbert_imdb_train.py
```

Run inference by editing the text line
```python
text = "I absolutely love this movie!"
```

in the file run [distilbert_imdb_infer_sentiment.py](./distilbert_imdb_infer_sentiment.py) and run as follow

```console
python3 distilbert_imdb_infer_sentiment.py
```



To deactivate virtual environment

```console
Deactivate
```



## References

[<a name="distilBERT">1</a>] Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2020). ``DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter''. https://arxiv.org/abs/1910.01108

