## Data preparation

### 1. nuScenes data preprocess

Download the [nuScenes](https://www.nuscenes.org/) dataset.

use script data_prepare/nuscenes_seq_process.py (same as [motionnet](https://github.com/pxiangwu/MotionNet))

### 2. generate static mask

Note: to use the default Raft model provided by torchvision, should use torchvision>=0.13.

use script data_prepare/generate_static_mask.py

### 3. generate rigid piece

use script data_prepare/generate_rigid_piece.py