## Automatic Speech Recognition (ASR) with PyTorch

An educational project for building, training and evaluating end-to-end Automatic Speech Recognition (ASR) systems with PyTorch.

This repository was created as part of an ASR task and contains a compact pipeline for:

- dataset loading (streaming support for large datasets),
- preprocessing and audio transforms,
- a small convolution + transformer encoder and an autoregressive transformer decode (the whole architecture is inspired by LAS, but with major changes),
- training and inference scripts with configurable options (Hydra-style configs under `src/configs`),
- utilities for computing WER/CER and running beam-search or greedy decoding.

If you're using this repo for experiments, treat it as a starting point — many components are intentionally simple and intended to be extended.

## Quick links

- Code: `train.py`, `inference.py`, `src/models/models.py`
- Configs: `src/configs/` (examples: `baseline.yaml`, `inference.yaml`, `metrics_eval.yaml`)
- Dataset helpers: `src/datasets/` (streaming & custom-dir loader)
- Demo notebook: `DEMO.ipynb`
- Weights links:
- encoder: https://drive.google.com/uc?export=download&id=1UpX3_UgrbRTWYunAMHPsR09a1_zHzj7E
- decoder: https://drive.google.com/uc?export=download&id=1A1Cb1TCn5LWYuIADkOsfvzlBBBi2L2bi


## Requirements

Install required Python packages listed in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

Notes:
- The code has been developed and tested with recent PyTorch and Hugging Face `datasets` versions. If you encounter import or API errors, try upgrading those packages.
- Some demo/streaming features may require authentication for Hugging Face (see `DEMO.ipynb` which shows `notebook_login`).

## Running training

Basic training example (uses the Hydra-style configs under `src/configs`):

```powershell
python train.py -cn=baseline trainer.num_epochs=55 trainer.device=cuda dataloader.batch_size_train=64
```

Key points:
- `-cn` picks a named config from `src/configs` (for example `baseline`).
- Additional Hydra keys can be passed on the command line to override config values (device, batch size, learning rate, etc.).

See `src/configs/baseline.yaml` for a working example.

## Running inference

Examples (see `DEMO.ipynb` for notebook-ready commands):

- Greedy decoding (argmax):

```powershell
python inference.py output_dir="inference_predictions" batch_size=8 datasets.hf.enabled=true strategy='greedy'
```

- Beam-search decoding:

```powershell
python inference.py output_dir="inference_predictions" batch_size=8 datasets.hf.enabled=true strategy='beam' 
```

- Inference on a local custom directory (structure: <DatasetRoot>/audio and <DatasetRoot>/transcriptions):

```powershell
python inference.py batch_size=8 datasets.custom_dir.enabled=true datasets.custom_dir.path="C:/path/to/your_ds" output_dir="inference_custom" strategy='beam'
```

The inference script will write predicted transcripts into the configured `output_dir` grouped by dataset split.

## Evaluation (WER / CER)

There is a small evaluation helper `calc_metrics.py` that compares ground-truth text files with predictions. Files should be plain text with one utterance per file. Example (from demo):

```powershell
python calc_metrics.py paths.gt_dir=./ground_truth/test_clean paths.pred_dir=./inference_predictions/test_clean
```

The script prints WER and CER and can be pointed to any folders containing `.txt` transcripts.

## Notebooks and demos

Open `DEMO.ipynb` for runnable examples showing:

- how to download ground-truth transcripts from LibriSpeech streaming,
- how to create a small local dataset from streaming datasets,
- example training and inference commands.

## Project layout

Important files and directories:

- `train.py` — training CLI entrypoint
- `inference.py` — inference CLI entrypoint
- `calc_metrics.py` — compute WER/CER between two folders of transcripts
- `src/models/models.py` — model definition (encoder + decoder with decoding strategies)
- `src/configs/` — YAML configs used by the CLI
- `src/datasets/` — dataset loaders and collate functions
- `DEMO.ipynb` — interactive demo notebook with detailed instructions

## Development notes & suggestions

- To add a new model, modify or add a file under `src/models/` and update configs to point at it.
- The repo uses positional sinusoidal encodings and simple Conv blocks for downsampling in the encoder alongside with almost vanilla transformer — you might replace these with spectrogram frontends, pretrained feature extractors and other advanced approaches for better performance.
- Train more on bigger amount of data and gpu hours
## License

This project is released under the MIT License — see the `LICENSE` file.

## Credits

This repository is derived from an educational PyTorch template made by the Deep Learning in Audio course of the HSE University, Moscow. Russia. Here is the link for the original template: https://github.com/Blinorot/pytorch_project_template/tree/example/asr

