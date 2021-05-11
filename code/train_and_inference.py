import subprocess
import json
import argparse


"""
configs/default.json에서 train.py이나 inference.py에 argument로 넘기고 싶은 argument 추가 가능
train.py에 넘기고 싶은 argument는 train_args 하위에, inference.py는 inference_args 하위에 추가

--config로 원하는 configuration을 설정할 수 있는 방식.
python subprocess_test.py --config ./configs/default.json을 실행 시 train.py와 inference.py가 순차적으로 실행
"""

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/default.json')

args = parser.parse_args()

with open(args.config) as f:
    cfg = json.load(f)

train_args = []
inference_args = []

# Add train_args
for key in cfg["train_args"].keys():
    train_args.extend(["--{}".format(key), str(cfg["train_args"][key])])

# Add inference args
inference_args.extend(['--model_name_or_path', cfg['train_args']['output_dir']])
for key in cfg["inference_args"].keys():
    inference_args.extend(["--{}".format(key), str(cfg["inference_args"][key])])
# Train Reader Model
subprocess.run(["python",
                "train.py",
                "--do_train"] + train_args)
# Evaluate Model
subprocess.run(["python",
                "train.py",
                "--do_eval"] + train_args)
# Inference Model
subprocess.run(["python",
                "inference.py",
                "--dataset_name", "/opt/ml/input/data/data/test_dataset/",
                "--do_predict"] + inference_args)

