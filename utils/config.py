import yaml

def load_config(path):
    """加载 yaml 配置文件"""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg
