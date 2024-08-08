# Fish2Mesh Transformer: 3D Human Mesh Reconstruction from Egocentric Vision

# News!
- Jun 2024: Our new work: Fish2Mesh Transformer: 3D Human Mesh Reconstruction from Egocentric Vision
- Dec 2023: [SoloPose](https://github.com/Santa-Clara-Media-Lab/SoloPose) is released!
- Nov 2023: [MoEmo](https://github.com/Santa-Clara-Media-Lab/MoEmo_Vision_Transformer) codes are released!
- Oct 2023: MoEmo was accepted by IROS 2023 (IEEE/RSJ International Conference on Intelligent Robots and Systems).

# Install
My torch version is 2.0.1. I am not sure whether I missing some package which we need to intall. If I miss something, please let me know.
```
conda create -n pose python=3.8
conda activate pose
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install opencv-python
pip install matplotlib
```
Download SMPL model, which is essential for regression a Human Mesh to 3D joints.


https://drive.google.com/file/d/1cYFB2-5fuEmbECJIAb-hDB3O0lw4rk70/view?usp=drive_link





