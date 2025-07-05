import os
import torchaudio
from src.config import config

def check_all_songs():
    """모든 곡을 체크해서 문제가 있는 곡을 찾습니다."""
    
    root = config.data['root']
    subset = 'train'
    sources = config.data['sources']  # ['piano']
    
    path = os.path.join(root, subset)
    songs = sorted([song for song in os.listdir(path) if not song.startswith('.')])
    
    print(f"전체 {len(songs)}개 곡을 체크합니다...")
    print(f"Sources: {sources + ['mix']}")
    print("-" * 50)
    
    problematic_songs = []
    
    for i, song in enumerate(songs):
        if i % 50 == 0:  # 50개마다 진행상황 출력
            print(f"진행상황: {i+1}/{len(songs)}")
        
        song_path = os.path.join(path, song)
        lengths = {}
        
        try:
            for source in sources + ['mix']:
                file_path = os.path.join(song_path, f'{source}.wav')
                
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Missing {source}.wav")
                
                waveform, sample_rate = torchaudio.load(file_path)
                lengths[source] = waveform.shape[-1]
            
            # 길이 비교
            unique_lengths = set(lengths.values())
            if len(unique_lengths) > 1:
                print(f"❌ 곡 {song}: 길이 불일치 {lengths}")
                problematic_songs.append((song, lengths))
                
        except Exception as e:
            print(f"❌ 곡 {song}: 에러 - {e}")
            problematic_songs.append((song, f"Error: {e}"))
    
    print("\n" + "=" * 50)
    if problematic_songs:
        print(f"문제가 있는 곡들 ({len(problematic_songs)}개):")
        for song, issue in problematic_songs:
            print(f"  {song}: {issue}")
    else:
        print("모든 곡이 정상입니다!")
    
    return problematic_songs

if __name__ == "__main__":
    check_all_songs()