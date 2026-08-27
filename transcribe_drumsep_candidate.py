"""D55 候選推論入口：先用固定 DrumSep 分離，再交給既有轉譜器。"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from transcribe import transcribe


ROOT = Path(__file__).resolve().parent
STEMS = ('kick', 'snare', 'toms', 'hh', 'ride', 'crash')


def make_drumsep_mix(audio_path, workspace):
    """以 D47 固定配方分離輸入並將六 stem 相加為單一 drum-only WAV。"""
    source = Path(audio_path).resolve()
    input_dir, output_dir = workspace / 'input', workspace / 'output'
    input_dir.mkdir()
    linked = input_dir / f'input{source.suffix.lower()}'
    try:
        os.link(source, linked)
    except OSError:
        shutil.copy2(source, linked)
    command = [
        sys.executable, str(ROOT / 'third_party' / 'Music-Source-Separation-Training' / 'inference.py'),
        '--model_type', 'mdx23c', '--config_path', str(ROOT / 'Drumsep' / 'config_drumsep_mdx23c.yaml'),
        '--start_check_point', str(ROOT / 'Drumsep' / 'MDX23C-DrumSep-aufr33-jarredou.ckpt'),
        '--input_folder', str(input_dir), '--store_dir', str(output_dir), '--disable_detailed_pbar', '--bigshifts', '1',
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    stem_dir = output_dir / 'input'
    waveforms = []
    for stem in STEMS:
        waveform, sample_rate = sf.read(stem_dir / f'{stem}.wav', dtype='float32', always_2d=True)
        if sample_rate != 44100 or waveform.size == 0:
            raise ValueError(f'Invalid DrumSep stem: {stem}')
        waveforms.append(waveform)
    length = min(len(waveform) for waveform in waveforms)
    # ponytail: 直接相加與 D55 training adapter 相同，不引入第二個融合模型。
    mixture = np.sum([waveform[:length] for waveform in waveforms], axis=0, dtype=np.float32)
    mix_path = workspace / 'drumsep_mix.wav'
    sf.write(mix_path, mixture, 44100, subtype='FLOAT')
    return mix_path


def main():
    """建立暫存 drum-only mix 並呼叫既有 transcribe 函式。"""
    parser = argparse.ArgumentParser(description='Transcribe one file through DrumSep then the existing model.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--architecture', default='dcnn-tcn-conformer')
    parser.add_argument('--threshold', type=float, default=None)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='d55_drumsep_') as temporary:
        mix_path = make_drumsep_mix(args.input, Path(temporary))
        transcribe(
            audio_path=str(mix_path), model_path=args.model, output_midi_path=args.output,
            threshold=args.threshold, architecture=args.architecture,
        )


if __name__ == '__main__':
    main()
