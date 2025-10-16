import warnings
import io
from pathlib import Path
from src.datasets.collate import collate_fn
import hydra
import torch
from hydra.utils import instantiate, get_original_cwd
from src.datasets.librispeech_streaming import LibriSpeechTorchDataset, CombinedLibriSpeechDataset

from datasets import load_dataset, Audio, Features, Value
from transformers import WhisperTokenizer

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    tokenizer_name = config.tokenizer.get("name", "openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained(tokenizer_name)

    features = Features(
        {
            "file": Value("string"),
            "audio": Audio(decode=False),
            "text": Value("string"),
            "speaker_id": Value("int64"),
            "chapter_id": Value("int64"),
            "id": Value("string"),
        }
    )

  
    train_clean = load_dataset(
        "librispeech_asr", "clean", split=config.datasets.hf.train_clean_split, features=features, streaming=True
    )
    train_other = load_dataset(
        "librispeech_asr", "other", split=config.datasets.hf.train_other_split, features=features, streaming=True
    )
    val_clean = load_dataset(
        "librispeech_asr", "clean", split=config.datasets.hf.val_clean_split, features=features, streaming=True
    )
    val_other = load_dataset(
        "librispeech_asr", "other", split=config.datasets.hf.val_other_split, features=features, streaming=True
    )

    


    dataset_clean_train = LibriSpeechTorchDataset(train_clean, tokenizer, length=104014, augment=True)
    dataset_other_train = LibriSpeechTorchDataset(train_other, tokenizer, length=148688, augment=True)
    dataset_clean_val = LibriSpeechTorchDataset(val_clean, tokenizer, length=2703, augment=False)
    dataset_other_val = LibriSpeechTorchDataset(val_other, tokenizer, length=2864, augment=False)

    lengths = config.datasets.hf.get("lengths", None)

    dataset_train = CombinedLibriSpeechDataset(
        [dataset_clean_train, dataset_other_train],
        lengths=lengths,
        tokenizer=tokenizer,
        shuffle=True,
    )



    from torch.utils.data import DataLoader

    train_bs = int(config.dataloader.batch_size_train)
    val_bs = int(config.dataloader.batch_size_val)
    num_workers = int(config.dataloader.num_workers)

    train_loader = DataLoader(
        dataset_train,
        batch_size=train_bs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    loader_val_clean = DataLoader(
        dataset_clean_val,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    loader_val_other = DataLoader(
        dataset_other_val,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    encoder = instantiate(config.model.encoder)
    decoder = instantiate(config.model.decoder)

    from src.trainer.trainer import Trainer

    trainer_kwargs = dict(
        encoder=encoder,
        decoder=decoder,
        tokenizer=tokenizer,
        train_loader=train_loader,
        loader_val_clean=loader_val_clean,
        loader_val_other=loader_val_other,
        device=device,
        pad_id=int(config.trainer.pad_id),
        eos_id=int(config.trainer.eos_id),
        bos_id=int(config.trainer.bos_id),
        notimestamps_id=int(config.trainer.notimestamps_id),
        vocab_size=int(config.trainer.vocab_size),
        start_lr=float(config.trainer.start_lr),
        num_epochs=int(config.trainer.num_epochs),
        log_interval=int(config.trainer.log_interval),
        examples_per_log=int(config.trainer.examples_per_log),
        eval_batches=int(config.trainer.eval_batches),
        run_name=config.get("run_name", "default_run"),
    )
    

    trainer = Trainer(**trainer_kwargs)

    if config.get("pretrained", None):
        pretrained_cfg = config.pretrained
        if pretrained_cfg.get("enabled", False):
            enc_link = pretrained_cfg.get("encoder_link", None)
            dec_link = pretrained_cfg.get("decoder_link", None)

            if enc_link or dec_link:
                import gdown
                from pathlib import Path
                from hydra.utils import get_original_cwd

                tmp_dir = Path(get_original_cwd()) / ".pretrained"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                enc_path = dec_path = None

                if enc_link:
                    enc_path = str(tmp_dir / "encoder.pth")
                    gdown.download(enc_link, enc_path, quiet=False)
                if dec_link:
                    dec_path = str(tmp_dir / "decoder.pth")
                    gdown.download(dec_link, dec_path, quiet=False)

                trainer.load_models(enc_path, dec_path)

    trainer.train(start_epoch=config.trainer.get("start_epoch", 0))


if __name__ == "__main__":
    main()
