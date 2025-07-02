import os
import torch
import torchaudio
from datetime import datetime
import argparse

from hstasnet import HSTasNet


def load_model(model_path, device='cpu'):
    """훈련된 모델을 로드합니다.
    
    Args:
        model_path (str): 모델 파일 경로
        device (str): 사용할 디바이스
        
    Returns:
        model: 로드된 모델
    """
    # 모델 파라미터 (train.py에서 가져옴)
    model_args = {
        'num_sources': 2,
        'num_channels': 2,
        'time_win_size': 1024,
        'time_hop_size': 512,
        'time_ftr_size': 512,
        'spec_win_size': 1024,
        'spec_hop_size': 512,
        'spec_fft_size': 1024,
        'rnn_hidden_size': 512,
        'device': torch.device(device)
    }
    
    # 모델 인스턴스 생성
    model = HSTasNet(**model_args)
    
    # 저장된 파일이 dict라면 state_dict만 추출
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def load_audio(audio_path, sample_rate=44100):
    """오디오 파일을 로드합니다.
    
    Args:
        audio_path (str): 오디오 파일 경로
        sample_rate (int): 목표 샘플레이트
        
    Returns:
        waveform (torch.Tensor): [C, L] 형태의 오디오 텐서
    """
    waveform, sr = torchaudio.load(audio_path)
    
    # 샘플레이트 확인 및 리샘플링
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # 모노를 스테레오로 변환
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    # 스테레오 이상을 스테레오로 변환
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]
    
    return waveform


def save_separated_sources(separated_sources, output_dir, source_names, timestamp):
    """분리된 소스들을 개별 파일로 저장합니다.
    
    Args:
        separated_sources (torch.Tensor): [S, C, L] 형태의 분리된 소스들
        output_dir (str): 출력 디렉토리
        source_names (list): 소스 이름들
        timestamp (str): 타임스탬프 문자열
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, source_name in enumerate(source_names):
        output_path = os.path.join(output_dir, f"{timestamp}_{source_name}.wav")
        torchaudio.save(output_path, separated_sources[i], 44100)
        print(f"저장됨: {output_path}")


def infer_single_audio(model, audio_path, output_dir, device='cpu'):
    """단일 오디오 파일에 대해 소스 분리를 수행합니다.
    
    Args:
        model: 훈련된 HSTasNet 모델
        audio_path (str): 입력 오디오 파일 경로
        output_dir (str): 출력 디렉토리
        device (str): 사용할 디바이스
    """
    # 소스 이름들 (train.py에서 가져옴)
    source_names = ['other', 'vocals']
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    
    print(f"오디오 로드 중: {audio_path}")
    waveform = load_audio(audio_path)
    
    # 배치 차원 추가: [C, L] -> [1, C, L]
    waveform = waveform.unsqueeze(0)
    waveform = waveform.to(device)
    
    print("소스 분리 수행 중...")
    with torch.no_grad():
        # 모델 추론: [1, C, L] -> [1, S, C, L]
        separated = model(waveform)
        
        # 배치 차원 제거: [1, S, C, L] -> [S, C, L]
        separated = separated.squeeze(0)
        
        # CPU로 이동
        separated = separated.cpu()
    
    print("분리된 소스 저장 중...")
    save_separated_sources(separated, output_dir, source_names, timestamp)
    
    print(f"소스 분리 완료! 결과는 {output_dir}에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description='HSTasNet 모델을 이용한 소스 분리')
    parser.add_argument('--model_path', type=str, 
                       default='out/models/hstasnet_20250701.pt',
                       help='훈련된 모델 파일 경로')
    parser.add_argument('--input_audio', type=str, 
                       default='./test.wav',
                       help='입력 오디오 파일 경로')
    parser.add_argument('--output_dir', type=str, 
                       default='out/separated',
                       help='출력 디렉토리')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='사용할 디바이스 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 모델 파일 존재 확인
    if not os.path.exists(args.model_path):
        print(f"에러: 모델 파일을 찾을 수 없습니다: {args.model_path}")
        return
    
    # 입력 오디오 파일 존재 확인
    if not os.path.exists(args.input_audio):
        print(f"에러: 입력 오디오 파일을 찾을 수 없습니다: {args.input_audio}")
        return
    
    print(f"모델 로드 중: {args.model_path}")
    model = load_model(args.model_path, args.device)
    
    print(f"디바이스: {args.device}")
    model = model.to(args.device)
    
    # 추론 실행
    infer_single_audio(model, args.input_audio, args.output_dir, args.device)


if __name__ == '__main__':
    main()
