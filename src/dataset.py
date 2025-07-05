import os
import torch
import torchaudio
from torch.utils.data import Dataset
from config import config

EXT = 'wav'
SAMPLE_RATE = config.data['sample_rate']
TARGET_LENGTH = SAMPLE_RATE * 10  # 22050 Hz * 10초 = 220500 프레임


class HSTNDataSet(Dataset):
    """ Dataset class for the MUSDB18 dataset.

    Args:
        root (str): The root directory of the dataset.
    """

    def __init__(self,
                 root: str,
                 subset: str, 
                 sources: list = config.data['sources'],
                 ):
        
        self.root = root
        self.subset = subset
        self.sources = sources
        self.path = os.path.join(root, subset)
        self.songs = self._collect_songs()

    def _get_source(self, name, source):
        return os.path.join(self.path, name, f'{source}.{EXT}')

    def _normalize_length(self, wav, target_length=TARGET_LENGTH):
        """오디오를 지정된 길이로 맞춥니다.
        
        Args:
            wav (torch.Tensor): 오디오 텐서, shape [C, L]
            target_length (int): 목표 길이
            
        Returns:
            torch.Tensor: 정규화된 오디오 텐서, shape [C, target_length]
        """
        current_length = wav.shape[-1]
        
        if current_length > target_length:
            # 길이가 더 길면 앞에서부터 자르기
            return wav[..., :target_length]
        elif current_length < target_length:
            # 길이가 더 짧으면 0으로 패딩
            padding = target_length - current_length
            return torch.nn.functional.pad(wav, (0, padding), mode='constant', value=0)
        else:
            # 길이가 같으면 그대로 반환
            return wav

    def _load_sample(self, n: int):
        song = self.songs[n]
        wav_sources = []
        
        for source in self.sources + ['mix']:
            track = self._get_source(song, source)
            wav, sr = self.load_audio(track)

            if sr != SAMPLE_RATE:
                raise ValueError(f"Expected sample rate {SAMPLE_RATE}, but got {sr}.")
            
            # 모든 오디오를 TARGET_LENGTH로 정규화
            wav = self._normalize_length(wav)
            wav_sources.append(wav)
        
        wav_mixture = wav_sources.pop()
        wav_sources = torch.stack(wav_sources)
         
        return wav_mixture, wav_sources

    def _collect_songs(self):
        song_names = []
        for song_name in os.listdir(self.path):
            if not song_name.startswith('.'):
                song_names.append(song_name)

        return sorted(song_names)
    
    def __getitem__(self, n: int):
        """ Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded.

        Returns:
            wav_mixture (torch.Tensor): Tensor of waveforms, size [C, L].
            wav_sources (torch.Tensor): Tensor of waveforms, size [S, C, L].
        """
        return self._load_sample(n)    
    
    def __len__(self):
        return len(self.songs)  
  
    def load_audio(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate


if __name__ == '__main__':
    root = config.data['root']
    subset = 'valid'
    sources = config.data['sources']
    dataset = HSTNDataSet(root, subset, sources)
    wav_sources, wav_mixture = dataset[0]
    print(f'{wav_sources.size() = }')
    print(f'{wav_mixture.size() = }')
    