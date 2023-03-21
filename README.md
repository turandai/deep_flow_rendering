# Deep Flow Rendering

Implementation for EGSR &amp; CGF 2022 paper "Deep Flow Rendering: View Synthesis via Layer-aware Reflection Flow".

### Reqiurments
tensorflow 1.15.0, nvdiffrast 0.3.0 (others are same as nvdiffrast), or you can install the original conda env via:
<pre><code> conda env create -f requirements.yml </pre></code>


### Usage
Edit model name and training configurations directly in main.py.

### Data
1. Example data: 
&emsp; - download from [here](https://github.com/turandai/dfr).
3. Custome data: 
   &emsp;&emsp; use [COLMAP](https://github.com/colmap/colmap) for pose estimation and mesh reconstruction.
