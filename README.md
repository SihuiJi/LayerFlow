<p align="center">

<!-- ### [<a href="" target="_blank">arXiv</a>] [<a href="" target="_blank">Project Page</a>] [<a href="" target="_blank">Model Weights</a>]
_**[Sihui Ji<sup>1,2*</sup>](https://sihuiji.github.io/), [Hao Luo<sup>2,3</sup>](https://menghanxia.github.io/), [Xi Chen<sup>1</sup>](https://xavierchen34.github.io/), [Yuanpeng Tu<sup>1</sup>](https://yuanpengtu.github.io/), [Yiyang Wang<sup>1</sup>](https://scholar.google.com/citations?user=nKr8TJwAAAAJ&hl=en), <br>[Hengshuang Zhao<sup>1†</sup>](https://hszhao.github.io/)**_
<br>
(*Work done during an internship at DAMO Academy, Alibaba Group †corresponding author)

<sup>1</sup>The University of Hong Kong, <sup>2</sup>DAMO Academy, Alibaba Group, <sup>3</sup>Hupan Lab.

</div> -->

  <h2 align="center">LayerFlow: A Unified Model for Layer-aware Video Generation</h2>
  <p align="center">
    <a href="https://sihuiji.github.io/"><strong>Sihui Ji</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=7QvWnzMAAAAJ&hl=zh-CN"><strong>Hao Luo</strong></a>
    ·
    <a href="https://xavierchen34.github.io/"><strong>Xi Chen</strong></a>
    ·
    <a href="https://yuanpengtu.github.io/"><strong>Yuanpeng Tu</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=nKr8TJwAAAAJ&hl=en"><strong>Yiyang Wang</strong></a>
    ·
    <a href="https://hszhao.github.io/"><strong>Hengshuang Zhao</strong></a>
    <br>
    <br>
        <a href="https://arxiv.org/abs/2506.04228"><img src='https://img.shields.io/badge/arXiv-LayerFlow-red' alt='Paper PDF'></a>
        <a href='https://sihuiji.github.io/LayerFlow-Page/'><img src='https://img.shields.io/badge/Project_Page-LayerFlow-green' alt='Project Page'></a>
        <a href=''><img src='https://img.shields.io/badge/ModelScope-Weights-yellow'></a>
        <a href='https://huggingface.co/zjuJish/LayerFlow'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-blue'></a>
        <!-- <a href='https://replicate.com/lucataco/anydoor'><img src='https://replicate.com/lucataco/anydoor/badge'></a> -->
    <br>
    <b>The University of Hong Kong &nbsp; | &nbsp;  DAMO Academy, Alibaba Group  | &nbsp;  Hupan Lab </b>
  </p>
  
  <table align="center">
    <tr>
    <td>
      <img src="./teaser.png">
    </td>
    </tr>
  </table>


<!-- <div align="center">
<div align="center" style="margin-top: 0px; margin-bottom: 0px;">
<img src=logo.png width="30%"/>
</div> -->

<!-- **Important Note:** This open-source repository is intended to provide a reference implementation. Due to the difference in the underlying T2V model's performance, the open-source version may not achieve the same performance as the model in our paper. If you'd like to use the best version of ReCamMaster, please upload your video to [this link](https://docs.google.com/forms/d/e/1FAIpQLSezOzGPbm8JMXQDq6EINiDf6iXn7rV4ozj6KcbQCSAzE8Vsnw/viewform?usp=dialog). Additionally, we are working on developing an online trial website. Please stay tuned to updates on the [Kling website](https://app.klingai.com/global/). -->

## 🔥 News
<!-- - __[2025.04.15]__: Please feel free to explore our related work, [SynCamMaster](https://github.com/KwaiVGI/SynCamMaster). -->
<!-- and  [model checkpoint](https://huggingface.co/zjuJish/LayerFlow). -->
<!-- - __[2025.03.31]__: Release the [MultiCamVideo Dataset](https://huggingface.co/datasets/KwaiVGI/MultiCamVideo-Dataset).
- __[2025.03.31]__: We have sent the inference results to the first 1000 trial users. -->
- __[2025.06.17]__: Release the [inference code](https://github.com/SihuiJi/LayerFlow?tab=readme-ov-file#inference).
- __[2025.06.04]__: Release the [project page](https://sihuiji.github.io/LayerFlow-Page/) and the [arxiv paper](https://arxiv.org/abs/2506.04228).
- __[2025.03.29]__: LayerFLow is accepted by Siggragh 2025 🎉🎉🎉.

  
## 📖 Introduction

**TL;DR:** We present LayerFlow, a unified solution for layer-aware video generation. Given per-layer prompts, LayerFlow generates videos for the transparent foreground, clean background, and blended scene. It also supports versatile variants like decomposing a blended video or generating the background for the given foreground and vice versa.  <br>

<!-- https://github.com/user-attachments/assets/52455e86-1adb-458d-bc37-4540a65a60d4 -->

<!-- ## 🚀 Trail: Try ReCamMaster with Your Own Videos

**Update:** We are actively processing the videos uploaded by users. So far, we have sent the inference results to the email addresses of the first **1256** testers. You should receive an email titled "Inference Results of ReCamMaster" from either jianhongbai@zju.edu.cn or cpurgicn@gmail.com. Please also check your spam folder, and let us know if you haven't received the email after a long time. If you enjoyed the videos we created, please consider giving us a star 🌟.

**You can try out our ReCamMaster by uploading your own video to [this link](https://docs.google.com/forms/d/e/1FAIpQLSezOzGPbm8JMXQDq6EINiDf6iXn7rV4ozj6KcbQCSAzE8Vsnw/viewform?usp=dialog), which will generate a video with camera movements along a new trajectory.** We will send the mp4 file generated by ReCamMaster to your inbox as soon as possible. For camera movement trajectories, we offer 10 basic camera trajectories as follows:

| Index       | Basic Trajectory                  |
|-------------------|-----------------------------|
| 1    | Pan Right                   |
| 2 | Pan Left                    |
| 3 | Tilt Up                     |
| 4 | Tilt Down                   |
| 5 | Zoom In                     |
| 6 | Zoom Out                    |
| 7 | Translate Up (with rotation)   |
| 8 | Translate Down (with rotation) |
| 9 | Arc Left (with rotation)    |
| 10 | Arc Right (with rotation)   |

If you would like to use ReCamMaster as a baseline and need qualitative or quantitative comparisons, please feel free to drop an email to [jianhongbai@zju.edu.cn](mailto:jianhongbai@zju.edu.cn). We can assist you with batch inference of our model. -->

## 📑 Open-source Plan

- [x] Inference code
- [x] Model checkpoints
- [ ] Training code


## 🛠️ Installation
Begin by cloning the repository:
```sh
git clone https://github.com/SihuiJi/LayerFlow.git
cd LayerFlow
```

Our project is developed based on the SAT version code of [CogVideoX](https://github.com/THUDM/CogVideo?tab=readme-ov-file#sat). You can follow the [instructions](https://github.com/THUDM/CogVideo/blob/main/sat/README.md) of CogVideoX to install dependencies or: <br> 

```bash
conda create -n layer python==3.10
conda activate layer
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install -r requirements.txt
```

## 🧱 Download Pretrained Models


| Models       | Download Link (RGB version)                                                                                                                                           |    Download Link (RGBA version)                      |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| Multi-layer generation      | 🤗 [Huggingface](https://huggingface.co/zjuJish/LayerFlow/tree/main/multi-layer-generation-rgb)      🤖 [ModelScope]()             | 🤗 [Huggingface](https://huggingface.co/zjuJish/LayerFlow/tree/main/multi-layer-generation-rgba)      🤖 [ModelScope]()
| Multi-layer decomposition | 🤗 [Huggingface](https://huggingface.co/zjuJish/LayerFlow/tree/main/multi-layer-decomposition-rgb)    🤖 [ModelScope]()     | 🤗 [Huggingface](https://huggingface.co/zjuJish/LayerFlow/tree/main/multi-layer-decomposition-rgba)      🤖 [ModelScope]()
| Foreground-conditioned generation | 🤗 [Huggingface](https://huggingface.co/zjuJish/LayerFlow/tree/main/foreground-conditioned-generation-rgb)    🤖 [ModelScope]()     | 🤗 [Huggingface](https://huggingface.co/zjuJish/LayerFlow/tree/main/foreground-conditioned-generation-rgba)      🤖 [ModelScope]()
| Background-conditioned generation     | 🤗 [Huggingface](https://huggingface.co/zjuJish/LayerFlow/tree/main/background-conditioned-generation-rgb)     🤖 [ModelScope]()            | 🤗 [Huggingface](https://huggingface.co/zjuJish/LayerFlow/tree/main/background-conditioned-generation-rgba)      🤖 [ModelScope]()        


> 💡Note: 
> * All models are finetuned from CogVideoX-2B.
> * RGB version represents the models generating foreground layer without alpha-matte, while the model of RGBA version simultaneously generate foreground videos and its alpha-matte which can be combined into RGBA videos.
However, due to difficulties in cross-domain generation and channel alignment, the results are generally less stable compared to RGB version.

Download models using huggingface-cli:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download zjuJish/LayerFlow --local-dir ./sat/ckpts_2b_lora
```
or using git:
``` sh
git lfs install
git clone https://huggingface.co/zjuJish/LayerFlow
```

<!-- Download models using modelscope-cli:
``` sh
pip install modelscope
modelscope download zjuhuijun/CogVideo --local_dir ./sat/ckpts_2b_lora
``` -->
For the pretrained VAE from CogVideoX-2B model, download as follows:

```
mkdir CogVideoX-2b-sat
cd CogVideoX-2b-sat
wget https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1
mv 'index.html?dl=1' vae.zip
unzip vae.zip
```

<!-- Arrange the model files in the following structure:

```
.
├── transformer
│   ├── 1000 (or 1)
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── vae
    └── 3d-vae.pt
``` -->

Since model weight files are large, it’s recommended to use `git lfs`.
See [here](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#installing) for `git lfs` installation.

```
git lfs install
```

Next, clone the T5 model, which is used as an encoder and doesn’t require training or fine-tuning.

```
git clone https://huggingface.co/THUDM/CogVideoX-2b.git # Download model from Huggingface
# git clone https://www.modelscope.cn/ZhipuAI/CogVideoX-2b.git # Download from Modelscope
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/*  CogVideoX-2b-sat/t5-v1_1-xxl
```

> You may also use the model file location on [Modelscope](https://modelscope.cn/models/ZhipuAI/CogVideoX-2b).

Arrange the above model files in the following structure:

```
CogVideoX-2b-sat
│
├── t5-v1_1-xxl
│   ├── added_tokens.json
│   ├── config.json
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   ├── model.safetensors.index.json
│   ├── special_tokens_map.json
│   ├── spiece.model
│   └── tokenizer_config.json
└── vae
    └── 3d-vae.pt

sat
│
├── ckpts_2b_lora
│   ├── multi-layer-generation
│   ├── 1000
│   │   └── mp_rank_00_model_states.pt
│   └── latest
│   ├── multi-layer-decomposition
│   ├── 1000
│   │   └── mp_rank_00_model_states.pt
│   └── latest
│   ├── foreground-conditioned-generation
│   ├── 1000
│   │   └── mp_rank_00_model_states.pt
│   └── latest
│   ├── background-conditioned-generation
│   ├── 1000
│   │   └── mp_rank_00_model_states.pt
│   └── latest

```

<!-- ```
│   ├── 1000 (or 1)
│   │   └── mp_rank_00_model_states.pt
│   └── latest
``` -->


## 🔑 Inference

```
cd sat
```

Run multi-layer generation (RGB version)

```
bash 'inference_stage2_gen_rgb.sh'
```

Run multi-layer generation (RGBA version)

```
bash 'inference_stage2_gen_rgba.sh'
```

Run multi-layer decomposition (RGB version)

```
bash 'inference_stage2_seg_rgb.sh'
```

Run multi-layer decomposition (RGBA version)

```
bash 'inference_stage2_seg_rgba.sh'
```

Run foreground-conditioned generation (RGB version)

```
bash 'inference_stage2_fg2bg_rgb.sh'
```

Run foreground-conditioned generation (RGBA version)

```
bash 'inference_stage2_fg2bg_rgba.sh'
```

Run background-conditioned generation (RGB version)

```
bash 'inference_stage2_bg2fg_rgb.sh'
```

Run background-conditioned generation (RGBA version)

```
bash 'inference_stage2_bg2fg_rgba.sh'
```


<!-- ## ⚙️ Code: ReCamMaster + Wan2.1 (Inference & Training)
The model utilized in our paper is an internally developed T2V model, not [Wan2.1](https://github.com/Wan-Video/Wan2.1). Due to company policy restrictions, we are unable to open-source the model used in the paper. Consequently, we migrated ReCamMaster to Wan2.1 to validate the effectiveness of our method. Due to differences in the underlying T2V model, you may not achieve the same results as demonstrated in the demo.
### Inference
Step 1: Set up the environment

[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) requires Rust and Cargo to compile extensions. You can install them using the following command:
```shell
curl --proto '=https' --tlsv1.2 -sSf [https://sh.rustup.rs](https://sh.rustup.rs/) | sh
. "$HOME/.cargo/env"
```

Install [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio):
```shell
git clone https://github.com/KwaiVGI/ReCamMaster.git
cd ReCamMaster
pip install -e .
```

Step 2: Download the pretrained checkpoints
1. Download the pre-trained Wan2.1 models

```shell
cd ReCamMaster
python download_wan2.1.py
```
2. Download the pre-trained ReCamMaster checkpoint

Please download from [huggingface](https://huggingface.co/KwaiVGI/ReCamMaster-Wan2.1/blob/main/step20000.ckpt) and place it in ```models/ReCamMaster/checkpoints```.

Step 3: Test the example videos
```shell
python inference_recammaster.py --cam_type 1
```

Step 4: Test your own videos

If you want to test your own videos, you need to prepare your test data following the structure of the ```example_test_data``` folder. This includes N mp4 videos, each with at least 81 frames, and a ```metadata.csv``` file that stores their paths and corresponding captions. You can refer to the [Prompt Extension section](https://github.com/Wan-Video/Wan2.1?tab=readme-ov-file#2-using-prompt-extension) in Wan2.1 for guidance on preparing video captions. 

```shell
python inference_recammaster.py --cam_type 1 --dataset_path path/to/your/data
```

We provide several preset camera types, as shown in the table below. Additionally, you can generate new trajectories for testing.

| cam_type       | Trajectory                  |
|-------------------|-----------------------------|
| 1    | Pan Right                   |
| 2 | Pan Left                    |
| 3 | Tilt Up                     |
| 4 | Tilt Down                   |
| 5 | Zoom In                     |
| 6 | Zoom Out                    |
| 7 | Translate Up (with rotation)   |
| 8 | Translate Down (with rotation) |
| 9 | Arc Left (with rotation)    |
| 10 | Arc Right (with rotation)   |

### Training

Step 1: Set up the environment

```shell
pip install lightning pandas websockets
```

Step 2: Prepare the training dataset

1. Download the [MultiCamVideo dataset](https://huggingface.co/datasets/KwaiVGI/MultiCamVideo-Dataset).

2. Extract VAE features

```shell
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_recammaster.py   --task data_process   --dataset_path path/to/the/MultiCamVideo/Dataset   --output_path ./models   --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"   --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"   --tiled   --num_frames 81   --height 480   --width 832 --dataloader_num_workers 2
```

3. Generate Captions for Each Video

You can use video caption tools like [LLaVA](https://github.com/haotian-liu/LLaVA) to generate captions for each video and store them in the ```metadata.csv``` file.

Step 3: Training
```shell
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_recammaster.py   --task train  --dataset_path recam_train_data   --output_path ./models/train   --dit_path "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"   --steps_per_epoch 8000   --max_epochs 100   --learning_rate 1e-4   --accumulate_grad_batches 1   --use_gradient_checkpointing  --dataloader_num_workers 4
```
We do not explore the optimal set of hyper-parameters and train with a batch size of 1 on each GPU. You may achieve better model performance by adjusting hyper-parameters such as the learning rate and increasing the batch size.

Step 4: Test the model

```shell
python inference_recammaster.py --cam_type 1 --ckpt_path path/to/the/checkpoint
``` -->

<!-- ## 📷 Dataset: MultiCamVideo Dataset
### 1. Dataset Introduction

**TL;DR:** The MultiCamVideo Dataset is a multi-camera synchronized video dataset rendered using Unreal Engine 5. It includes synchronized multi-camera videos and their corresponding camera trajectories. The MultiCamVideo Dataset can be valuable in fields such as camera-controlled video generation, synchronized video production, and 3D/4D reconstruction. If you are looking for synchronized videos captured with stationary cameras, please explore our [SynCamVideo Dataset](https://github.com/KwaiVGI/SynCamMaster).

https://github.com/user-attachments/assets/6fa25bcf-1136-43be-8110-b527638874d4

The MultiCamVideo Dataset is a multi-camera synchronized video dataset rendered using Unreal Engine 5. It includes synchronized multi-camera videos and their corresponding camera trajectories.
It consists of 13.6K different dynamic scenes, each captured by 10 cameras, resulting in a total of 136K videos. Each dynamic scene is composed of four elements: {3D environment, character, animation, camera}. Specifically, we use animation to drive the character, 
and position the animated character within the 3D environment. Then, Time-synchronized cameras are set up to move along predefined trajectories to render the multi-camera video data.
<p align="center">
  <img src="https://github.com/user-attachments/assets/107c9607-e99b-4493-b715-3e194fcb3933" alt="Example Image" width="70%">
</p>

**3D Environment:** We collect 37 high-quality 3D environments assets from [Fab](https://www.fab.com). To minimize the domain gap between rendered data and real-world videos, we primarily select visually realistic 3D scenes, while choosing a few stylized or surreal 3D scenes as a supplement. To ensure data diversity, the selected scenes cover a variety of indoor and outdoor settings, such as city streets, shopping malls, cafes, office rooms, and the countryside.

**Character:** We collect 66 different human 3D models as characters from [Fab](https://www.fab.com) and [Mixamo](https://www.mixamo.com).

**Animation:** We collect 93 different animations from [Fab](https://www.fab.com) and [Mixamo](https://www.mixamo.com), including common actions such as waving, dancing, and cheering. We use these animations to drive the collected characters and create diverse datasets through various combinations.

**Camera:** To ensure camera movements are diverse and closely resemble real-world distributions, we create a wide range of camera trajectories and parameters to cover various situations. To achieve this by designing rules to batch-generate random camera starting positions and movement trajectories:

1. Camera Starting Position.

We take the character's position as the center of a hemisphere with a radius of {3m, 5m, 7m, 10m} based on the size of the 3D scene and randomly sample within this range as the camera's starting point, ensuring the closest distance to the character is greater than 0.5m and the pitch angle is within 45 degrees.

2. Camera Trajectories.

- **Pan & Tilt**:  
  The camera rotation angles are randomly selected within the range, with pan angles ranging from 5 to 45 degrees and tilt angles ranging from 5 to 30 degrees, with directions randomly chosen left/right or up/down.

- **Basic Translation**:  
  The camera translates along the positive and negative directions of the xyz axes, with movement distances randomly selected within the range of $[\frac{1}{4}, 1] \times \text{distance2character}$.

- **Basic Arc Trajectory**:  
  The camera moves along an arc, with rotation angles randomly selected within the range of 15 to 75 degrees.

- **Random Trajectories**:  
  1-3 points are sampled in space, and the camera moves from the initial position through these points as the movement trajectory, with the total movement distance randomly selected within the range of $[\frac{1}{4}, 1] \times \text{distance2character}$. The polyline is smoothed to make the movement more natural.

- **Static Camera**:  
  The camera does not translate or rotate during shooting, maintaining a fixed position.

3. Camera Movement Speed.

To further enhance the diversity of trajectories, 50% of the training data uses constant-speed camera trajectories, while the other 50% uses variable-speed trajectories generated by nonlinear functions. Consider a camera trajectory with a total of $f$ frames, starting at location $L_{start}$ and ending at position $L_{end}$. The location at the $i$-th frame is given by:
```math
L_i = L_{start} + (L_{end} - L_{start}) \cdot \left( \frac{1 - \exp(-a \cdot i/f)}{1 - \exp(-a)} \right),
```
where $a$ is an adjustable parameter to control the trajectory speed. When $a > 0$, the trajectory starts fast and then slows down; when $a < 0$, the trajectory starts slow and then speeds up. The larger the absolute value of $a$, the more drastic the change.

4. Camera Parameters.

We chose four set of camera parameters: {focal=18mm, aperture=10}, {focal=24mm, aperture=5}, {focal=35mm, aperture=2.4} and {focal=50mm, aperture=2.4}.

### 2. Statistics and Configurations

Dataset Statistics:

| Number of Dynamic Scenes | Camera per Scene | Total Videos |
|:------------------------:|:----------------:|:------------:|
| 13,600                   | 10               | 136,000      |

Video Configurations:

| Resolution  | Frame Number | FPS                      |
|:-----------:|:------------:|:------------------------:|
| 1280x1280   | 81           | 15                       |

Note: You can use 'center crop' to adjust the video's aspect ratio to fit your video generation model, such as 16:9, 9:16, 4:3, or 3:4.

Camera Configurations:

| Focal Length            | Aperture           | Sensor Height | Sensor Width |
|:-----------------------:|:------------------:|:-------------:|:------------:|
| 18mm, 24mm, 35mm, 50mm  | 10.0, 5.0, 2.4     | 23.76mm       | 23.76mm      |



### 3. File Structure
```
MultiCamVideo-Dataset
├── train
│   ├── f18_aperture10
│   │   ├── scene1    # one dynamic scene
│   │   │   ├── videos
│   │   │   │   ├── cam01.mp4    # synchronized 81-frame videos at 1280x1280 resolution
│   │   │   │   ├── cam02.mp4
│   │   │   │   ├── ...
│   │   │   │   └── cam10.mp4
│   │   │   └── cameras
│   │   │       └── camera_extrinsics.json    # 81-frame camera extrinsics of the 10 cameras 
│   │   ├── ...
│   │   └── scene3400
│   ├── f24_aperture5
│   │   ├── scene1
│   │   ├── ...
│   │   └── scene3400
│   ├── f35_aperture2.4
│   │   ├── scene1
│   │   ├── ...
│   │   └── scene3400
│   └── f50_aperture2.4
│       ├── scene1
│       ├── ...
│       └── scene3400
└── val
    └── 10basic_trajectories
        ├── videos
        │   ├── cam01.mp4    # example videos corresponding to the validation cameras
        │   ├── cam02.mp4
        │   ├── ...
        │   └── cam10.mp4
        └── cameras
            └── camera_extrinsics.json    # 10 different trajectories for validation
```

### 3. Useful scripts
- Data Extraction
```bash
cat MultiCamVideo-Dataset.part* > MultiCamVideo-Dataset.tar.gz
tar -xzvf MultiCamVideo-Dataset.tar.gz
```
- Camera Visualization
```python
python vis_cam.py
```

The visualization script is modified from [CameraCtrl](https://github.com/hehao13/CameraCtrl/blob/main/tools/visualize_trajectory.py), thanks for their inspiring work.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f9cf342d-2fb3-40ef-a7be-edafb5775004" alt="Example Image" width="40%">
</p> -->

<!-- ## 🤗 Awesome Related Works
Feel free to explore these outstanding related works, including but not limited to:

[GCD](https://gcd.cs.columbia.edu/): GCD synthesizes large-angle novel viewpoints of 4D dynamic scenes from a monocular video.

[ReCapture](https://generative-video-camera-controls.github.io/): a method for generating new videos with novel camera trajectories from a single user-provided video.

[Trajectory Attention](https://xizaoqu.github.io/trajattn/): Trajectory Attention facilitates various tasks like camera motion control on images and videos, and video editing.

[GS-DiT](https://wkbian.github.io/Projects/GS-DiT/): GS-DiT provides 4D video control for a single monocular video.

[Diffusion as Shader](https://igl-hkust.github.io/das/): a versatile video generation control model for various tasks.

[TrajectoryCrafter](https://trajectorycrafter.github.io/): TrajectoryCrafter achieves high-fidelity novel views generation from casually captured monocular video.

[GEN3C](https://research.nvidia.com/labs/toronto-ai/GEN3C/): a generative video model with precise Camera Control and temporal 3D Consistency. -->

## 🤗 Acknowledgements

This project is developed on the codebase of [CogVideoX](https://github.com/THUDM/CogVideo). We  appreciate this great work! 

## 🌟 Citation

Please leave us a star 🌟 and cite our paper if you find our work helpful.
```
@article{ji2025layerflow,
  title={LayerFlow : A Unified Model for Layer-aware Video Generation}, 
  author={Ji, Sihui and Luo, Hao and Chen, Xi and Tu, Yuanpeng and Wang, Yiyang and Zhao, Hengshuang},
  year={2025},
  journal={arXiv preprint arXiv:2506.04228}, 
}
```