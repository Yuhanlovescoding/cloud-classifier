# Purpose
This is for training a cloud classifier in torch. The classifier is MobileNet from torchvision, quantized for deployment.

# Environment
The main dependencies are
```
python==3.8.10
torch==2.0.1
torchvision==0.15.2
numpy==1.24.4
```
The model is trained on a NVIDIA RTX A4000 for 60 epochs

# Dataset
[Cirrus Cumulus Stratus Nimbus(CCSN) Database](https://github.com/upuil/CCSN-Database)
Download the .zip file and unzip it into a folder. for example:
```
data/
|   data/Ac/Ac-N001.jpg
|   data/Ac/Ac-N002.jpg
...
|   data/St/St-N001.jpg
...
```
The dataset contains 2,543 images. 85% are used to form a training set, 15% are held out for validation.
The final accuracy test is done on the entire dataset.

# Train & Test 
If everything is set up properly, just run `python train.py` and wait for the results.