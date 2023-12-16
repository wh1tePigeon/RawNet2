# RawNet2 implementation

## Installation guide

1) clone repository
```shell
git clone https://github.com/wh1tePigeon/RawNet2
```
2) install requirements
```shell
pip install -r ./requirements.txt
```

## Train 
Download [dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) and specify path to train, dev and eval files and protocols in config. Then run 
```shell
python3 train.py -c source/configs/config_kaggle.json
```
Main model was trained with `source/configs/config_kaggle.json`.

## Download checkpoints
Checkpoints and training logs can be found [here](https://drive.google.com/drive/folders/1oMc90dS7YGLoC5Emh5pl-hp2wZKzUdft?usp=sharing). To download them use download_model.py script. If you want to test models other than main, make sure you edited code according to model.


~~## Test~~
~~To test solution run~~
```shell
python3 test.py
```
## Wandb
[Wandb report](https://wandb.ai/belki/rawnet_project/reports/RawNet2--Vmlldzo2Mjc5MjQ3?accessToken=1eohvrtqb6toopxyeh64kavy168naeftwnkkp42vaiv2dzeh7bj2wyb9j49pmwye)

## Credits

This repository is based on a heavily modified fork
of [hw template](https://github.com/WrathOfGrapes/asr_project_template) repository.


