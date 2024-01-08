#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
import tempfile

from tempfile import NamedTemporaryFile
# from fairseq.data.audio.audio_utils import get_waveform
# from fairseq.data.audio.waveform_transforms import CompositeAudioWaveformTransform
import sys
sys.path.append(r"E:\START_UP\Process_Data\speech_to_text")
import pandas as pd
from data_utils import (
    create_zip,
    extract_fbank_features,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm
import os
from typing import BinaryIO, List, Optional, Tuple, Union
import glob
import torch
log = logging.getLogger(__name__)
import numpy as np

# SPLITS = [
#     "train-clean-100",
#     "train-clean-360",
#     "train-other-500",
#     "dev-clean",
#     "dev-other",
#     "test-clean",
#     "test-other",
# ]
SF_AUDIO_FILE_EXTENSIONS = {".wav", ".flac", ".ogg"}
FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS = {".npy", ".wav", ".flac", ".ogg"}
def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization

    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
    try:
        import torchaudio.sox_effects as ta_sox
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])
    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(
            _waveform, sample_rate, effects
        )
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate

def get_waveform(
    path_or_fp: Union[str, BinaryIO],
    normalization: bool = True,
    mono: bool = True,
    frames: int = -1,
    start: int = 0,
    always_2d: bool = True,
    output_sample_rate: Optional[int] = None,
    normalize_volume: bool = False,
    # waveform_transforms: Optional[CompositeAudioWaveformTransform] = None,
) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit WAV/FLAC/OGG Vorbis audio.

    Args:
        path_or_fp (str or BinaryIO): the path or file-like object
        normalization (bool): normalize values to [-1, 1] (Default: True)
        mono (bool): convert multi-channel audio to mono-channel one
        frames (int): the number of frames to read. (-1 for reading all)
        start (int): Where to start reading. A negative value counts from the end.
        always_2d (bool): always return 2D array even for mono-channel audios
        output_sample_rate (Optional[int]): output sample rate
        normalize_volume (bool): normalize volume
    Returns:
        waveform (numpy.ndarray): 1D or 2D waveform (channels x length)
        sample_rate (float): sample rate
    """
    if isinstance(path_or_fp, str):
        ext = Path(path_or_fp).suffix
        if ext not in SF_AUDIO_FILE_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile: pip install soundfile")

    waveform, sample_rate = sf.read(
        path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start
    )
    waveform = waveform.T  # T x C -> C x T
    waveform, sample_rate = convert_waveform(
        waveform,
        sample_rate,
        normalize_volume=normalize_volume,
        to_mono=mono,
        to_sample_rate=output_sample_rate,
    )

    if not normalization:
        waveform *= 2**15  # denormalized to 16-bit signed integers

    # if waveform_transforms is not None:
    #     waveform, sample_rate = waveform_transforms(waveform, sample_rate)

    if not always_2d:
        waveform = waveform.squeeze(axis=0)

    return waveform, sample_rate

SPLITS = [
    'train'
]
MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


def process(args):
    out_root = Path(args.output_root)
    out_root.mkdir(exist_ok=True)
    wav_root = Path(args.wav_root)
    # Extract features
    feature_root = out_root / "fbank80"
    feature_root.mkdir(exist_ok=True)
    for split in SPLITS:
        
        
        print(f"Fetching split {split}...")
        
        paths = glob.glob(f"{wav_root}/{split}/*.wav")
        
        for path in paths:
            
            wav, sample_rate = get_waveform(
                path_or_fp = path, 
                output_sample_rate = 22050
            )
            sample_id = os.path.basename(path).replace(".wav", '')
            
            extract_fbank_features(
                wav, sample_rate, feature_root / f"{sample_id}.npy"
            )
        
       
    # Pack features into ZIP
    zip_path = out_root / "fbank80.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    for split in SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        paths = glob.glob(f"{wav_root}/{split}/*.lab")
        for path in paths:
            spk_id = None
            sample_id = os.path.basename(path).replace(".lab", '')
            with open(path, 'r') as file:
                utt = file.read().strip()
            manifest["id"].append(sample_id)
            manifest["audio"].append(audio_paths[sample_id])
            manifest["n_frames"].append(audio_lengths[sample_id])
            manifest["tgt_text"].append(utt.lower())
            manifest["speaker"].append(spk_id)
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), out_root / f"{split}.tsv"
        )
        if split.startswith("train"):
            train_text.extend(manifest["tgt_text"])
    # Generate vocab
    vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}"
    with tempfile.NamedTemporaryFile(mode="w", dir=out_root, delete=False, suffix=".txt") as f:

        for t in train_text:
            f.write(t + "\n")
        print('-------------Path(f.name) --------------------')
        print(Path(f.name))
        gen_vocab(
            Path(f.name),
            out_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
        )
    # Generate config YAML
    gen_config_yaml(
        out_root,
        spm_filename=spm_filename_prefix + ".model",
        specaugment_policy="ld"
    )
    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument("--wav-root",  required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=10000, type=int)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
