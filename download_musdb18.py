import os
import torchaudio.datasets as datasets


if __name__ == '__main__':
    os.makedirs('/home/saens/data/MUSDB18', exist_ok=True)
    dataset = datasets.MUSDB_HQ(root='/home/saens/data/MUSDB18', subset='train', download=True)
    