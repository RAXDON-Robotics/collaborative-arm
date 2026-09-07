"""
record data.

example:
  source Y1/devel/setup.bash
  python -m scripts.record_data config_file=cfg/one_master_slave.yaml

"""

from omegaconf import OmegaConf
from utils.data_collection_cfg import DataCollectionCfg, build_config
from utils.data_collection import DataCollection

def main():
  # build data collection config
  cfg: DataCollectionCfg = build_config()
  print("cfg:\n",OmegaConf.to_yaml(cfg))
  
  # build DataCollection object
  data_collection = DataCollection(cfg)
  
  # start collection
  data_collection.run()

if __name__ == '__main__':
  main()