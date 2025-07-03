# Vietnam_Signlanguage_FE
Vietnamese sign language recognition using 3DCNN <br>
## Requirements
> Python >= 3.9<br> 
Torch >=1.13.1+cu116<br>Torchvision 1.14.1
<br>

Install require library on cli:
## Training
### Prepare
1. Install library Clone this repo
```sh
git clone https://github.com/pml0607/Vietnam_Signlanguage_FE.git
cd Vietnam_Signlanguage_FE
```
Before install, assert python is avaiable in your environment
```sh
pip install -r requirements.txt
```
2. Data preparation
- Make sure your raw data is stored as follows:
```explorer
dataset
|__A1P1
|  |__rgb
|     |__file1.avi
|     |__file2.avi
|        ...
|__A1P2
|  |__rgb
|     |__file1.avi
|     |__file2.avi
|        ...
|  ...

```
Change paths of `data root(video_root)`, `segmented root(output_root),` ... in Configurate/segmentation.<br><br> 
In this step, we will segment and randomly add backgrounds to the data. If the background diversity in your dataset is already high, you can skip this step.
```sh
python Ultralytics/segmentation.py
python Ultralytics/add_bg.py
```
After the step, you have 2 directories like those
```
segmented
|__bitwised
|  |__A1P1
|  |  |__rbg
|  |    |__file1.avi
|  |    |__file2.avi
|  |       ...
|  |__A1P2
|  |  |__rgb
|  |     |__file1.avi
|  |     |__file2.avi
|  |        ''' 
|     ...
|__mask
|  |__A1P1
|  |  |__rgb
|  |     |__file1.avi
|  |     |__file2.avi
|  |        ...
|  |__A1P2
|  |  |__rgb
|  |     |__file1.avi
|  |        ...
|     ...
```
```
video_with_random_background
|__A1P1
|  |__rgb
|     |__file1.avi
|        ...
|  ...

#similar to raw data
```
Finally, you chose a dataset (raw data or data with random background) and change the path of input in `Configurate/data_preprocess.yaml` (splits). Remmember that you must have csv files which restore video path and its label. After all of above, run this script:
```sh
python Pytorch/preprocess_sliding.py     # for 3 channel dataset
#or
python Pytorch/preprocess_sliding_6ch.py # for 6 channel dataset 
```
3. Training 
- Change data root path, number of classes, augument method on `Configurate/train.yaml'
- Run this script
```sh
python Pytorch/train.py
```