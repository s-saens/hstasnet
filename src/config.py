import os
import yaml
from typing import Dict, Any


class Config:
    """Configuration manager for HSTasNet project.
    
    This class loads and manages all configuration parameters from train.yml
    and provides easy access to different sections of the configuration.
    """
    
    def __init__(self, config_path: str = "src/train.yml"):
        """Initialize configuration.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get data-related configuration."""
        return self._config['data']
    
    @property
    def model(self) -> Dict[str, Any]:
        """Get model-related configuration."""
        return self._config['model']
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training-related configuration."""
        return self._config['training']
    
    @property
    def paths(self) -> Dict[str, Any]:
        """Get paths-related configuration."""
        return self._config['paths']
    
    @property
    def other(self) -> Dict[str, Any]:
        """Get other configuration."""
        return self._config['other']
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config
    
    def get(self, key: str, default=None):
        """Get configuration value by key.
        
        Args:
            key (str): Configuration key (supports dot notation like 'data.sample_rate').
            default: Default value if key not found.
            
        Returns:
            Configuration value or default.
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default


# Global configuration instance
config = Config()


def get_model_args(device=None):
    """Get model arguments dictionary ready for HSTasNet initialization.
    
    Args:
        device: PyTorch device. If None, uses config device.
        
    Returns:
        Dict: Model arguments dictionary.
    """
    import torch
    
    model_config = config.model
    
    return {
        'num_sources': model_config['num_sources'],
        'num_channels': model_config['num_channels'],
        'time_win_size': model_config['time_win_size'],
        'time_hop_size': model_config['time_hop_size'],
        'time_ftr_size': model_config['time_ftr_size'],
        'spec_win_size': model_config['spec_win_size'],
        'spec_hop_size': model_config['spec_hop_size'],
        'spec_fft_size': model_config['spec_fft_size'],
        'rnn_hidden_size': model_config['rnn_hidden_size'],
        'rnn_num_layers': model_config['rnn_num_layers'],
        'device': device or torch.device(config.training['device'])
    }


def get_file_paths(date_suffix=None):
    """Get commonly used file paths.
    
    Args:
        date_suffix (str): Date suffix for files. If None, uses today's date.
        
    Returns:
        Dict: Dictionary with common file paths.
    """
    from datetime import datetime
    
    if date_suffix is None:
        date_suffix = datetime.today().strftime('%Y%m%d')
    
    model_name = config.model['name']
    paths_config = config.paths
    
    return {
        'model_path': os.path.join(paths_config['models_dir'], f"{model_name}_{date_suffix}.pt"),
        'solver_path': os.path.join(paths_config['solvers_dir'], f"{model_name}_{date_suffix}.pkl"),
        'log_path': os.path.join(paths_config['logs_dir'], f"{model_name}_{date_suffix}.log"),
        'output_dir': paths_config['separated_dir']
    }