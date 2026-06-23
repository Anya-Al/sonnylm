# SonnyLM 🐈‍⬛

A tiny (~8.7M parameter) character language model that talks like **Sonny** — my fat,
long-haired orange cat: lazy, extremely vocal, hates being picked up, has furry "grinch
paws," sleeps only on dirty laundry, and loves exactly two things: tuna and mom
(mom because she brings the tuna).

<img width="500" height="450" alt="IMG_6128" src="https://github.com/user-attachments/assets/3b9a6c76-f20c-4588-85e9-00f57f0969dd" />


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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anya-Al/sonnylm/blob/main/SonnyLM_Train.ipynb)

1. Runtime → Change runtime type → **T4 GPU**.
2. **Runtime → Run all.** It downloads the dataset, trains a BPE tokenizer, trains the
   model, tests it, and (optionally) uploads the weights to Hugging Face.

## Train locally

Needs a GPU for reasonable speed, but runs on CPU too (just slowly).

```bash
pip install -r requirements.txt
python prepare_data.py      # downloads the dataset, builds data/ + tokenizer
python train.py             # trains -> checkpoints/best_model.pt
python inference.py         # chat with Sonny in your terminal
```

## Run it in your browser (no install)

[![Open In Browser](https://colab.research.google.com/assets/colab-badge.svg)]
(https://anya-al.github.io/sonnylm/)

There's a fully client-side chat demo in [`docs/`](docs/) — it runs the quantized ONNX
model and a byte-level BPE tokenizer entirely in your browser via WebAssembly. Once
deployed it lives at `https://anya-al.github.io/sonnylm/`.

To turn it on:

1. Train the model (Colab), then grab `model.onnx` and `tokenizer.json` from the
   `sonnylm.tar.gz` the notebook produces (or from your Hugging Face model repo).
2. Drop both files into the `docs/` folder, next to `index.html`.
3. On GitHub: **Settings → Pages → Source: Deploy from a branch → `main` / `/docs` → Save.**
4. Wait ~1 minute, then open `[(https://anya-al.github.io/sonnylm/)]`.

Prefer not to commit the 9 MB model to git? Leave `docs/` model-free and instead edit the
two URL lines at the top of `docs/index.html` to point at your Hugging Face repo, e.g.
`https://huggingface.co/AnyaAl/sonnylm-9M/resolve/main/model.onnx`.

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
| `docs/index.html` | in-browser chat demo (ONNX + tokenizer, runs on GitHub Pages) |

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
