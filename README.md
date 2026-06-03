# [CVPR 2026] Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation
Official Pytorch Implementation of Avatar Forcing; Motion Latent Diffusion Forcing for Interactive Head Avatar Generation

![preview](./assets/overview.png)

### [CVPR 2026] Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation

[Taekyung Ki<sup>1</sup>*](https://taekyungki.github.io), &nbsp; [Sangwon Jang<sup>1</sup>*](https://agwmon.github.io/), &nbsp; [Jaehyeong Jo<sup>1</sup>](https://harryjo97.github.io/), &nbsp; [Jaehong Yoon<sup>2</sup>](https://jaehong31.github.io/), &nbsp;[Sung Ju hwang<sup>1,3</sup>](http://www.sungjuhwang.com/) <br>
<sup>1</sup>KAIST &nbsp; <sup>2</sup>NTU Singapore &nbsp; <sup>3</sup>DeepAuto.ai &nbsp; &nbsp; &nbsp; &nbsp; <sup>*</sup>Equal contribution

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://taekyungki.github.io/AvatarForcing/)
[![Pape](https://img.shields.io/badge/arXiv-2601.00664-b31b1b?logo=arxiv)](https://arxiv.org/abs/2601.00664v2)
[![YouTube](https://img.shields.io/badge/Watch-YouTube-red?logo=youtube)](https://www.youtube.com/watch?v=TARPGnJ8GW4)
[![Hugging Face](https://img.shields.io/badge/Hugging-Face-yellow?logo=huggingface)](https://huggingface.co/papers/2601.00664)


#### TL:DR: Interactive Head Avatar Generation Model via Diffusion Forcing toward Human-like Conversation

## Abstract

Talking head generation creates lifelike avatars from static portraits for virtual communication and content creation. However, current models do not yet convey the feeling of truly interactive communication, often generating one-way responses that lack emotional engagement. We identify two key challenges toward truly interactive avatars: generating motion in real-time under causal constraints and learning expressive, vibrant reactions without additional labeled data. To address these challenges, we propose Avatar Forcing, a new framework for interactive head avatar generation that models real-time user-avatar interactions through diffusion forcing. This design allows the avatar to process real-time multimodal inputs, including the user's audio and motion, with low latency for instant reactions to both verbal and non-verbal cues such as speech, nods, and laughter. Furthermore, we introduce a direct preference optimization method that leverages synthetic losing samples constructed by dropping user conditions, enabling label-free learning of expressive interaction. Experimental results demonstrate that our framework enables real-time interaction with low latency (approximately 500ms), achieving 6.8x speedup compared to the baseline, and produces reactive and expressive avatar motion, which is preferred over 80% against the baseline.


## Getting Started
### Requirements

```bash
# 1. Create Conda environment
conda create -n avatarforcing python==3.10
conda activate avatarforcing

# 2. Install torch and requirements
bash environment.sh

# or manual installation
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Model checkpoints
```bash
# 3. Download Avatar Forcing model and Wav2Vec2 from HuggingFace
bash download_weights.sh

# or manually download the weights from Google Drive
# https://drive.google.com/drive/folders/1rN52J2QXD8A-r2CZ8nDqFdwvZuPRmMsc?hl=ko

# and wav2vec2.0-960h from HuggingFace
# https://huggingface.co/facebook/wav2vec2-base-960h
```

The checkpoints (`pretrained_dir`) should be organized as follows:
```bash
./pretrained_dir
├── checkpoint_here
├── flow_transformer.pth              # main DFoT model
├── motion_autoencoder.pth            # motion AE model
└── wav2vec2-base-960h/               # pretrained wav2vec2 model
    ├── .gitattributes
    ├── config.json
    ├── feature_extractor_config.json
    ├── model.safetensors
    ├── preprocessor_config.json
    ├── pytorch_model.bin
    ├── README.md
    ├── special_tokens_map.json
    ├── tf_model.h5
    ├── tokenizer_config.json
    └── vocab.json
```


### Preprocessing
#### 1. Target User Speech Extraction from Video
Please use [IIANet](https://github.com/JusperLee/IIANet) for target speaker extraction and [ClearVoice](https://github.com/modelscope/ClearerVoice-Studio) for speaker separation. We observed that the performance of a generated interactive avatar heavily depends on the quality of its preprocessed data. For example, interaction quality and lip-sync accuracy rely on the performance of audio separation models. 

#### 2. User Video Pre-processing
```bash
python preprocess_user_video.py --user_video_path data/user.mp4 --output_path data --pad_ratio 1.0
```
The outputs will be the frames of input user video and its audio. This script converts given user video into video frames of 25fps and crop the facial region for better conditioning. Please adjust the `--pad_ratio` (default: `1.0`) if you want to scale the size of the bbox.


### Inference

```bash
CUDA_VISIBLE_DEVICES=XX python inference.py \ 
        --input_image_path data/rumi.jpg \ 
        --avatar_audio_path data/avatar.wav \ 
        --user_audio_path data/user.wav \ 
        --user_video_path data/user/ \ 
        --a_cfg_scale 2 \ 
        --u_cfg_scale 1 \ 
        --nfe 10
```

This repository only supports **minimal Pytorch inference pipeline** using a reference image, avatar audio, and a user video. Real-time conversation demos (e.g., GPT Voice API-based applications) are not included. Building a real-time conversational avatar system using this model and the acceleration techniques from our work is possible, but is not covered in this repository.

Note that you can also use Avatar Forcing for **talking-only** or **Listening-only** head avatar model.


## Citation
```bibtex
@InProceedings{Ki_2026_CVPR,
    author    = {Ki, Taekyung and Jang, Sangwon and Jo, Jaehyeong and Yoon, Jaehong and Hwang, Sung Ju},
    title     = {Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {18074-18084}
}
```

## Related Works
- FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait [ICCV 2025] 
- Self-Forcig: Bridging the Train-Test Gap in Autoregressive Video Diffusion [Neurips 2025] 
- History Guided Video Diffusion [ICML 2025] 
- Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion [Neurips 2024] 
- INFP: Audio-Driven Interactive Head Generation in Dyadic Conversations [CVPR 2025]
- IIANet: An Intra- and Inter-Modality Attention Network for Audio-Visual Speech Separation [ICML 2024]
- Clear Voice. https://github.com/modelscope/ClearerVoice-Studio


## Acknowledgements
The source image used in this codebase was generated with Gemini, and the audio sample was selected from the RealTalk dataset. We would also like to acknowledge the excellent open-source codebases and prior work that inspired and supported this project, including FLOAT, Self-Forcing, Diffusion Forcing, and LIA.

