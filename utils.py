import os
import json


def create_dir(base_dir, exp_name):
    if not os.path.exists(base_dir):
        print(f"{base_dir} directory created.")
        os.mkdir(base_dir)

    exp_path = os.path.join(base_dir, exp_name)
    if not os.path.exists(exp_path):
        print(f"{exp_name} experiment directory created.")
        os.mkdir(exp_path)

    return exp_path


def save_args(args, exp_path):
    with open(os.path.join(exp_path, "args.json"), "w") as f:
        json.dump(vars(args), f)
