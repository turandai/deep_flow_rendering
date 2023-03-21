# Deep Flow Rendering

Implementation for EGSR &amp; CGF 2022 paper "Deep Flow Rendering: View Synthesis via Layer-aware Reflection Flow".

## Reqiurments
TensorFlow 1.15.0, [NVdiffrast](https://github.com/NVlabs/nvdiffrast) 0.3.0 (others are same as nvdiffrast), or you can install the original conda env via:
``` 
conda env create -f requirements.yml
conda activate dfr
```


## Usage
Edit data path, model name, and training configurations directly in main.py.
```
python main.py
```

## Data
### Example data: 
Download from [here](https://github.com/turandai/dfr).
Unzip it in the base dir:
``` 
mv path_to_download/dfr.zip ./
unzip dfr_data.zip 
```
### Custome data: 
* Use [COLMAP](https://github.com/colmap/colmap)'s:
  * Sparse reconstruction for camera poses (use pinhole model and txt output) to get _**cameras.txt**_ and _**images.txt**_,
  * Dense reconstruction for mesh (might need manual configuration for a fine mesh) to get _**mesh.obj**_.
  * Use [Blender](https://www.blender.org/) (or any other equivalent like [xatlas](https://github.com/jpcy/xatlas)) to generate texture atlas for _**mesh.obj**_.
* Arrange your custome data dir custome_scene as the way same as the example data:
``` 
dfr/
|—— code/
|—— result/
|—— data/
|   |—— custome_scene/
|   |   |—— images/
|   |   |   |—— img_0.jpg
|   |   |   |—— ...
|   |   |   |—— img_n.jpg
|   |   |—— cameras.txt
|   |   |—— images.txt
|   |   |—— mesh.obj
```
