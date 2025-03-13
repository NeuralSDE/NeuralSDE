This directory contains the implementation for the video prediction task. Our approach enables to generate high-quality predictions with few inference steps and offer "free" temporal interpolation beyond the training schedule.

This repository is mostly based on [river](https://github.com/Araachie/river)

## Setup

Create a conda environment and install dependencies:
```bash
cd video
conda env create -f environment.yml
conda activate video
```
## Training your own models

To train your own video prediction models you need to start by preparing data. 

### Datasets

The training code expects the dataset to be packed into .hdf5 files in a custom manner. 
To create such files, use the provided `dataset/convert_to_h5.py` script. 
Usage example:

```angular2html
python dataset/convert_to_h5.py --out_dir <directory_to_store_the_dataset> --data_dir <path_to_video_frames> --image_size 128 --extension png
```

The output of `python dataset/convert_to_h5.py --help` is as follows:

```angular2html
usage: convert_to_h5.py [-h] [--out_dir OUT_DIR] [--data_dir DATA_DIR] [--image_size IMAGE_SIZE] [--extension EXTENSION]

optional arguments:
  -h, --help            show this help message and exit
  --out_dir OUT_DIR     Directory to save .hdf5 files
  --data_dir DATA_DIR   Directory with videos
  --image_size IMAGE_SIZE
                        Resolution to resize the images to
  --extension EXTENSION
                        Video frames extension

```

The video frames at `--data_dir` should be organized in the following way:

```angular2html
data_dir/
|---train/
|   |---00000/
|   |   |---00000.png
|   |   |---00001.png
|   |   |---00002.png
|   |   |---...
|   |---00001/
|   |   |---00000.png
|   |   |---00001.png
|   |   |---00002.png
|   |   |---...
|   |---...
|---val/
|   |---...
|---test/
|   |---...
```

To extract individual frames from a set of video files, we recommend using the `convert_video_directory.py` script from the [official PVG repository](https://github.com/willi-menapace/PlayableVideoGeneration#custom-datasets).


**KTH:** Download the videos from the [dataset's official website](https://www.csc.kth.se/cvap/actions/).

**CLEVRER:** Download the videos from the [official dataset's website](http://clevrer.csail.mit.edu/).

### Training autoencoder

We recommend to use the official [taming transformers repository](https://github.com/CompVis/taming-transformers) for 
training VQGAN. To use the trained VQGAN at the second stage, update the `model->autoencoder` field in the config accordingly. 
To do this, set `type` to `ldm-vq`, `config` to `f8_small`, `f8` or `f16` depending on the VQGAN config that was used at training.
We recommend using low-dimensional latents, e.g. from 4 to 8, and down-sampling images at least to 16 x 16 resolution. 

Besides, we also provide our own autoencoder architecture at `model/vqgan/vqvae.py` that one may use to train simpler VQVAEs.
For instance, our pretrained model on the CLEVRER dataset uses this custom implementation.

### Training main model

To launch the training of the main model, use the `train.py` script from this repository.
Usage example:
```angular2html
python train.py --config <path_to_config> --run-name <run_name> --wandb
```
The default config file name is in format: <dataset_name>_<component_name>-<noise_level>-<loss_function>.
The default run name is in format:
<component_name>-<noise_level>-<loss_function>.

```bash
python train.py --config ./configs/clevrer_f-01-null.yaml --run-name f-01-null --wandb
python train.py --config ./configs/clevrer_d-01.yaml --run-name d-01 --wandb
python train.py --config ./configs/clevrer_g-01-null.yaml --run-name g-01-null --wandb
```
Please don't forget to set the corresponding flow and denoiser components before training diffusion.

The output of `python train.py --help` is as follows:

```angular2html
usage: train.py [-h] --run-name RUN_NAME --config CONFIG [--num-gpus NUM_GPUS] [--resume-step RESUME_STEP] [--vqvae-path VQVAE_PATH] [--random-seed RANDOM_SEED] [--wandb]

optional arguments:
  -h, --help            show this help message and exit
  --run-name RUN_NAME   Name of the current run.
  --config CONFIG       Path to the config file.
  --num-gpus NUM_GPUS   Number of gpus to use for training. By default uses all available gpus.
  --resume-step RESUME_STEP
                        Step to resume the training from.
  --random-seed RANDOM_SEED
                        Random seed.
  --wandb               If defined, use wandb for logging.
```

Use the configs provided in this repository as examples. 


## Evaluation
Once you have trained flow, denoiser and diffusion components, run the following command to generate videos. You may need to change the model path based on your config file name and run name.

```bash
python evaluation/generate_videos.py
```