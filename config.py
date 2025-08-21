import json
from datetime import datetime
import logging
from pathlib import Path

# 中文註釋：全局變數，用於確保配置僅載入一次
_config_cache = None

def load_config():
    """中文註釋：載入所有設定檔案並生成 requirements.txt。"""
    global _config_cache
    if _config_cache is not None:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從快取載入配置")
        logging.info("Loaded config from cache")
        return _config_cache

    config = {}
    root_dir = "C:\\Trading"
    config_dir = Path(root_dir) / "config"
    config_files = {
        'api_keys': config_dir / 'api_keys.json',
        'trading_params': config_dir / 'trading_params.json',
        'system_config': config_dir / 'system_config.json'
    }
    
    try:
        for key, file_path in config_files.items():
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    config[key] = json.load(f)
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 成功載入設定檔：{file_path}")
                logging.info(f"Successfully loaded config file: {file_path}")
            else:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 設定檔不存在：{file_path}")
                logging.error(f"Config file not found: {file_path}")
                config[key] = {}
        
        system_config = config.get('system_config', {})
        dependencies = system_config.get('dependencies', [])
        requirements_path = Path(root_dir) / 'requirements.txt'
        with open(requirements_path, 'w', encoding='utf-8') as f:
            for dep in dependencies:
                f.write(f"{dep}\n")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已生成 requirements.txt：{requirements_path}")
        logging.info(f"Generated requirements.txt: {requirements_path}")
        
        _config_cache = config
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 設定檔載入成功！")
        logging.info("Config files loaded successfully")
        return config
    
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 設定檔載入失敗：{str(e)}")
        logging.error(f"Failed to load config files: {str(e)}")
        return {}

if __name__ == "__main__":
    logging.basicConfig(
        filename=Path('C:\\Trading') / 'logs' / 'config.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    load_config()