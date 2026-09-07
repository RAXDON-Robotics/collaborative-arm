### Task parameters
DATA_DIR = "/home/raxdon/RAXDON_LAB/data_collection/data"
TASK_CONFIGS = {
        # single arm
        'pick_and_place':{
        'dataset_dir': DATA_DIR + '/pick_and_place',
        'num_episodes': 50,
        'camera_names': ["cam_right_wrist", "cam_front"],
        "state_dim": 7,
        },

        # dual arm
        'folded_orange_towel':{
        'dataset_dir': DATA_DIR + '/folded_orange_towel',
        'num_episodes': 50,
        'camera_names': ["cam_right_wrist", "cam_front", "cam_left_wrist"],
        "state_dim": 14,
        },
}