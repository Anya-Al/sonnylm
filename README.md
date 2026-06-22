# SonnyLM 🐈‍⬛

A tiny (~8.7M parameter) character language model that talks like **Sonny** — my fat,
long-haired orange cat: lazy, extremely vocal, hates being picked up, has furry "grinch
paws," sleeps only on dirty laundry, and loves exactly two things: tuna and mom
(mom because she brings the tuna).

<img width="500" height="450" alt="IMG_6128" src="https://github.com/user-attachments/assets/2c228bfb-369e-4de4-987a-ca55768c9b58" />


It's a 6-layer vanilla transformer trained from scratch on the
[**AnyaAl/sonnylm_generic**](https://huggingface.co/datasets/AnyaAl/sonnylm_generic)
dataset (60K single-turn cat conversations across 60 topics). Re-skinned from
[GuppyLM](https://huggingface.co/datasets/arman-bd/guppylm-60k-generic).

```
You>   can i pick you up
Sonny> do not pick me up. i will go completely limp and scream. you have been warned.

You>   do you love mom
Sonny> mom is the only good human. she brings the tuna. that is love.
```

## Quickstart (Colab, ~5 min on a free T4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sonnylm/blob/main/SonnyLM_Train.ipynb)

1. Click the badge above (update `YOUR_USERNAME` after you push this repo).
2. Runtime → Change runtime type → **T4 GPU**.
3. **Runtime → Run all.** It downloads the dataset, trains a BPE tokenizer, trains the
   model, tests it, and (optionally) uploads the weights to Hugging Face.

## Train locally

Needs a GPU for reasonable speed, but runs on CPU too (just slowly).

```bash
pip install -r requirements.txt
python prepare_data.py      # downloads the dataset, builds data/ + tokenizer
python train.py             # trains -> checkpoints/best_model.pt
python inference.py         # chat with Sonny in your terminal
```

## Architecture

| | |
| --- | --- |
| Layers | 6 |
| Model dim | 384 |
| Heads | 6 |
| FFN hidden | 768 |
| Vocab | 4096 (byte-level BPE) |
| Context | 128 tokens |
| Params | ~8.7M |

A plain transformer — multi-head attention, ReLU FFN, LayerNorm, learned positional
embeddings, tied input/output embeddings. No RoPE, GQA, or SwiGLU. Deliberately simple.

## Files

| File | What it is |
| --- | --- |
| `config.py` | model + training hyperparameters |
| `model.py` | the transformer (`SonnyLM`) |
| `dataset.py` | ChatML loading + batching |
| `prepare_data.py` | download dataset, build tokenizer + JSONL |
| `train.py` | training loop (cosine LR, AMP on GPU) |
| `inference.py` | chat interface |
| `SonnyLM_Train.ipynb` | one-click Colab notebook (self-contained) |

## Where things live

- **Code** → this GitHub repo.
- **Dataset** → [AnyaAl/sonnylm_generic](https://huggingface.co/datasets/AnyaAl/sonnylm_generic) on Hugging Face.
- **Trained weights** → pushed to a Hugging Face *model* repo by section 7 of the notebook
  (PyTorch `.bin` + quantized ONNX, ~9 MB). Weights are kept out of git on purpose.

## Credits

Architecture and training recipe adapted from **GuppyLM** by
[arman-bd](https://huggingface.co/arman-bd). Built for fun, and for Sonny.

## License

MIT — see [LICENSE](LICENSE).
