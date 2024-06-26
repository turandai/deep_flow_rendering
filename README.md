# Deep Flow Rendering

This is the original implementation for the Computer Graphics Forum (2022) paper:
</br> "[**Deep Flow Rendering: View Synthesis via Layer-aware Reflection Flow**](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14593)",
</br> by [Pinxuan Dai](https://turandai.github.io/) & [Ning Xie](http://www.xielab.cn/publications.html), [UESTC](https://en.uestc.edu.cn/).</br>
<!-- ![alt text](https://github.com/turandai/dfr/blob/main/teaser.jpg) -->
* Open-access paper is avaliable at [Eurographics Digital Library](https://diglib.eg.org/bitstream/handle/10.1111/cgf14593/v41i4pp139-148.pdf).
* Oral recording at [EGSR 2022](https://egsr.eu/2022/) and supplementary video are [here](https://www.bilibili.com/video/BV14Z4y1i7id/).

## Reqiurments
TensorFlow 1.15.0, [NVdiffrast](https://github.com/NVlabs/nvdiffrast) 0.3.0, and install other packages via:
``` 
conda env create -f requirements.yml
conda activate dfr
```

## Usage
* Clone this repository and prepare test data as [below](https://github.com/turandai/dfr#data).
* Specify data path, model name, and training configurations directly in _**code/main.py**_.
* Run:
```
cd dfr/code
python main.py
```

## Data
### Example data: 
* Download example data used in the paper from [here](https://drive.google.com/file/d/1BJkghOcSqPv10ZhDOH2sWg1KyIM8Ygg8/view?usp=share_link).
* Unzip it in the _**dfr**_ base dir:
``` 
mv path_to_download/dfr_data.zip ./
unzip dfr_data.zip 
```
### Custome data: 
* Use [COLMAP](https://github.com/colmap/colmap)'s:
  * Sparse reconstruction for camera poses (use pinhole model and txt output) to get _**cameras.txt**_ and _**images.txt**_,
  * Dense reconstruction for mesh (might need manual configuration for a fine mesh) and convert it into .obj format.
* Use [Blender](https://www.blender.org/) (or any other equivalent like [xatlas](https://github.com/jpcy/xatlas)) to generate texture atlas for the reconstructed _**mesh.obj**_.
* Arrange your custome data dir _**custome_scene**_ in the same way as the example data:
``` 
dfr/
|—— code/...
|—— result/...
|—— data/
|   |—— custome_scene/
|   |   |—— images/
|   |   |   |—— img_0.jpg
|   |   |   |—— ...
|   |   |   |—— img_n.jpg
|   |   |—— cameras.txt
|   |   |—— images.txt
|   |   |—— mesh.obj
|   |—— ...
```

## Citation
```
@article{DaiDFR_CGF2022,
    author = {Dai, Pinxuan and Xie, Ning},
    title = {Deep Flow Rendering: View Synthesis via Layer-aware Reflection Flow},
    journal = {Computer Graphics Forum},
    volume = {41},
    number = {4},
    pages = {139-148},
    doi = {https://doi.org/10.1111/cgf.14593},
    year = {2022}
}
```
