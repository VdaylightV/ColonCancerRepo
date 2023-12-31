# ColonCancerRepo
## Introduction:
This is the repo for classify colon cancer image into T1 & T2 or T3 & T4 stages based on MR images.

This project employ the SVM algoritm based on AutoEncoder generated features. 

## Usage
**Install**
```
pip install -r ./requirements
```
**Train**
```
python train.py train_config.yaml
```
**Test**
```
python test.py test_config.yaml
```
**Test on Single Image**
```
python test_api.py test_api_config.yaml
```

Please refer the source code and config file for more details.


## Reference
Wang, Yizhang, et al. "A feature extraction based support vector machine model for rectal cancer T-stage prediction using MRI images." Multimedia Tools and Applications 80.20 (2021): 30907-30917.
