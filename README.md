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


## Test
To test solution run
```shell
python3 test.py
```

## Credits

This repository is based on a heavily modified fork
of [hw template](https://github.com/WrathOfGrapes/asr_project_template) repository.


