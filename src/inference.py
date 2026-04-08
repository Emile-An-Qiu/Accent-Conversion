import argparse
import os
import sys
import yaml
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from munch import Munch

try:
    from parallel_wavegan.utils import load_model as load_vocoder_model
except ImportError:
    load_vocoder_model = None


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_stargan_model(config, device):
    sys.path.insert(0, os.path.abspath('.'))
    from src.models import Generator, MappingNetwork, StyleEncoder

    args = Munch(config['model_params'])
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)

    nets = Munch(generator=generator, mapping_network=mapping_network, style_encoder=style_encoder)
    return nets


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_ema' in checkpoint:
        state_dict = checkpoint['model_ema']
    else:
        state_dict = checkpoint

    for key, module in model.items():
        module.load_state_dict(state_dict[key])
        module.to(device).eval()
    return model


def load_f0_model(f0_path, device):
    sys.path.insert(0, os.path.abspath('.'))
    from Utils.JDC.model import JDCNet

    f0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(f0_path, map_location='cpu')['net']
    f0_model.load_state_dict(params)
    f0_model = f0_model.to(device).eval()
    return f0_model


def load_vocoder(vocoder_path, device):
    if load_vocoder_model is None:
        raise RuntimeError('parallel_wavegan is required to load the vocoder.')
    vocoder = load_vocoder_model(vocoder_path).to(device).eval()
    if hasattr(vocoder, 'remove_weight_norm'):
        vocoder.remove_weight_norm()
    return vocoder


MEL_PARAMS = {
    'n_mels': 80,
    'n_fft': 2048,
    'win_length': 1200,
    'hop_length': 300,
}
MEAN = -4
STD = 4


def preprocess_wave(wave):
    if wave.ndim > 1:
        wave = np.mean(wave, axis=1)
    wave = wave.astype(np.float32)
    to_mel = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - MEAN) / STD
    return mel_tensor


def build_speaker_map(train_list_path):
    speaker_map = {}
    if not os.path.exists(train_list_path):
        return speaker_map

    with open(train_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '|' not in line:
                continue
            path, label = line.split('|')
            speaker = Path(path).parent.name
            speaker_map[speaker] = int(label)
    return speaker_map


def compute_style_embedding(ref_path, speaker_label, stargan, device):
    if not ref_path:
        label = torch.LongTensor([speaker_label]).to(device)
        latent_dim = stargan.mapping_network.shared[0].in_features
        ref = stargan.mapping_network(torch.randn(1, latent_dim).to(device), label)
        return ref

    wave, sr = librosa.load(ref_path, sr=None)
    if sr != 24000:
        wave = librosa.resample(wave, sr, 24000)
    wave, _ = librosa.effects.trim(wave, top_db=30)
    mel_tensor = preprocess_wave(wave).to(device)
    label = torch.LongTensor([speaker_label]).to(device)
    with torch.no_grad():
        ref = stargan.style_encoder(mel_tensor.unsqueeze(1), label)
    return ref


def convert_audio(source_path, ref_embedding, stargan, f0_model, vocoder, device):
    wave, sr = librosa.load(source_path, sr=None)
    if sr != 24000:
        wave = librosa.resample(wave, sr, 24000)
    source = preprocess_wave(wave).to(device)

    with torch.no_grad():
        f0_feat = f0_model.get_feature_GAN(source.unsqueeze(1))
        out = stargan.generator(source.unsqueeze(1), ref_embedding, F0=f0_feat)

        c = out.transpose(-1, -2).squeeze().to(device)
        wav_out = vocoder.inference(c)
        wav_out = wav_out.view(-1).cpu().numpy()

    wav_out = np.nan_to_num(wav_out, nan=0.0, posinf=0.0, neginf=0.0)
    peak = np.max(np.abs(wav_out))
    if peak > 0:
        wav_out = wav_out / peak * 0.95
    return wav_out


def run_inference_batch(
    source_path,
    reference_paths=None,
    speaker_ids=None,
    output_dir='Demo/converted',
    config_path='Models/VCTK2019/config.yml',
    checkpoint_path='Models/VCTK2019/epoch_00150.pth',
    vocoder_path='Vocoder/checkpoint-400000steps.pkl',
    f0_path='Utils/JDC/bst.t7',
    train_list_path=None,
    device=None,
):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    os.makedirs(output_dir, exist_ok=True)

    config = load_yaml(config_path)
    if train_list_path is None:
        train_list_path = config.get('train_data', 'Data/VCTK2019/train_list.txt')

    speaker_map = build_speaker_map(train_list_path)
    stargan = build_stargan_model(config, device)
    stargan = load_checkpoint(stargan, checkpoint_path, device)
    f0_model = load_f0_model(f0_path, device)
    vocoder = load_vocoder(vocoder_path, device)

    targets = []
    if reference_paths:
        for ref_path in reference_paths:
            speaker = Path(ref_path).parent.name
            speaker_label = speaker_map.get(speaker)
            if speaker_label is None:
                raise ValueError(f'Cannot determine speaker label for reference path: {ref_path}')
            targets.append((ref_path, speaker_label))

    if speaker_ids:
        for speaker_label in speaker_ids:
            targets.append((None, speaker_label))

    if not targets:
        raise ValueError('At least one reference path or speaker id must be provided.')

    results = []
    for ref_path, speaker_label in targets:
        ref_embedding = compute_style_embedding(ref_path, speaker_label, stargan, device)
        output_wav = convert_audio(source_path, ref_embedding, stargan, f0_model, vocoder, device)

        if ref_path:
            target_name = Path(ref_path).parent.name
        else:
            target_name = f'domain_{speaker_label}'

        target_name = target_name.replace('/', '_')
        out_path = Path(output_dir) / f'{Path(source_path).stem}_to_{target_name}.wav'
        sf.write(str(out_path), output_wav, 24000)
        results.append(str(out_path))

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Accent conversion inference script')
    parser.add_argument('--source', required=True, help='Input source WAV file')
    parser.add_argument('--reference_paths', nargs='*', default=[], help='Reference WAV paths for target speaker style')
    parser.add_argument('--speaker_ids', nargs='*', type=int, default=[], help='Target speaker IDs for mapping network inference')
    parser.add_argument('--output_dir', default='Demo/converted', help='Directory to save converted audio')
    parser.add_argument('--config_path', default='Models/VCTK2019/config.yml', help='StarGAN config path')
    parser.add_argument('--checkpoint_path', default='Models/VCTK2019/epoch_00150.pth', help='StarGAN checkpoint path')
    parser.add_argument('--vocoder_path', default='Vocoder/checkpoint-400000steps.pkl', help='Vocoder checkpoint path')
    parser.add_argument('--f0_path', default='Utils/JDC/bst.t7', help='F0 model path')
    parser.add_argument('--train_list_path', default=None, help='Train list path for speaker label mapping')
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_inference_batch(
        source_path=args.source,
        reference_paths=args.reference_paths,
        speaker_ids=args.speaker_ids,
        output_dir=args.output_dir,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        vocoder_path=args.vocoder_path,
        f0_path=args.f0_path,
        train_list_path=args.train_list_path,
    )
    print('Converted files:')
    for path in results:
        print(path)


if __name__ == '__main__':
    main()
