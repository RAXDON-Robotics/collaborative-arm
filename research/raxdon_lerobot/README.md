### Introduction 
lerobot : The code is based on the [Lerobot](https://github.com/lerobot/lerobot) repository. its corresponding commit version is 0217e1e3ad2ea74d07f997921e3f2bf86872c7d9

eval_robot: real robotic arm model inference

### Environment Setup
本仓库已并入 collaborative-arm monorepo（research/raxdon_lerobot）。LeRobot 为第三方框架依赖（官方 https://github.com/lerobot/lerobot，固定 commit 0217e1e3ad2ea74d07f997921e3f2bf86872c7d9），不再以 submodule 附带，请在本目录下自行拉取：

# Get the official lerobot source (pinned commit) into ./lerobot
git clone https://github.com/lerobot/lerobot.git lerobot
cd lerobot && git checkout 0217e1e3ad2ea74d07f997921e3f2bf86872c7d9 && cd ..

# Create a conda environment
conda create -y -n raxdon_lerobot python=3.10
conda activate raxdon_lerobot

# Install LeRobot
cd lerobot && pip install -e .

# 
pip install h5py tyro matplotlib

# Install raxdon_lerobot
cd ../ && pip install -e .

pip install "numpy<2.0"
pip install rospkg

### Data Collection and Conversion 
Refer to another repository: [data_collection]

# Visualize datasets

```bash
lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```

# Act
## Train Act Policy
python lerobot/src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=v30/pick_two_water_bottle_20251215 \
    --policy.type=act \
    --policy.push_to_hub=False

## eval act with dataset
### with temporal_ensemble (建议)
python eval_robot/raxdon_y1/eval_y1_dataset.py --policy.path=outputs/train/2025-12-09/13-37-48_act/checkpoints/last/pretrained_model/ --repo_id=v30/pick_two_water_bottle_20251215 --episode_index=2 --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1

### without temporal_ensemble
python eval_robot/raxdon_y1/eval_y1_dataset.py --policy.path=outputs/train/2025-12-09/13-37-48_act/checkpoints/last/pretrained_model/ --repo_id=v30/pick_two_water_bottle_20251215 --episode_index=2

## eval act with raxdon y1 robot
python eval_robot/raxdon_y1/eval_y1.py --policy.path=outputs/train/2025-12-09/13-37-48_act/checkpoints/last/pretrained_model/ --repo_id=v30/pick_two_water_bottle_20251215 --episode_index=2 --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1

# Diffusion Policy
## Train Diffusion Policy
python lerobot/src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=v30/pick_two_water_bottle_20251215 \
    --policy.type=diffusion \
    --policy.push_to_hub=False

## eval diffusion policy with dataset



## eval diffusion policy with raxdon y1 robot


## Train smolvla (fine tune)
python src/lerobot/scripts/train.py   --policy.path=lerobot/smolvla_base   --dataset.repo_id=piper/piper_place_and_place_0724  --batch_size=32  --policy.push_to_hub=false --wandb.enable=true

## Train Pi0 VLA Policy

## Train Pi05 VLA Policy

## Train Gr00t VLA Policy


# resume training
python src/lerobot/scripts/train.py   --policy.path=lerobot/smolvla_base   --dataset.repo_id=piper/piper_place_and_place_0724  --batch_size=32  --policy.push_to_hub=false --wandb.enable=true --resume=true --output_dir=output/...

### eval
# eval dataset
python eval_robot/raxdon_y1/eval_y1_dataset.py --policy.path=outputs/train/2025-07-23/16-34-41_act/checkpoints/last/pretrained_model/ --repo_id=piper/piper_place_and_place_0722

# eval real robot
python eval_robot/eval_piper/eval_piper.py --policy.path=outputs/train/2025-07-23/16-34-41_act/checkpoints/last/pretrained_model/ --repo_id=piper/piper_place_and_place_0722


# eval act dataset
## with temporal_ensemble (建议)
python eval_robot/raxdon_y1/eval_y1_dataset.py --policy.path=outputs/train/2025-12-09/13-37-48_act/checkpoints/last/pretrained_model/ --repo_id=v30/folded_orange_towel_1122 --episode_index=2 --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1

## without temporal_ensemble
python eval_robot/raxdon_y1/eval_y1_dataset.py --policy.path=outputs/train/2025-12-09/13-37-48_act/checkpoints/last/pretrained_model/ --repo_id=v30/folded_orange_towel_1122 --episode_index=2

# eval act with raxdon y1 robot
python eval_robot/raxdon_y1/eval_y1.py --policy.path=outputs/train/2025-12-09/13-37-48_act/checkpoints/last/pretrained_model/ --repo_id=v30/folded_orange_towel_1122 --episode_index=2 --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1
