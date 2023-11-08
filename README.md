# Video Instance Matting

[Jiachen Li](https://chrisjuniorli.github.io/), Roberto Henschel, [Vidit Goel](https://vidit98.github.io/), Marianna Ohanyan, Shant Navasardyan, [Humphrey Shi](https://www.humphreyshi.com/)

[[`arXiv`](https://arxiv.org/pdf/2311.04212.pdf)] [[`Code`](https://github.com/SHI-Labs/VIM)]

## Updates

11/02/2023: [Codes](https://github.com/SHI-Labs/VIM) and [arxiv](https://arxiv.org/pdf/2311.04212.pdf) are released.

## Installation

Step 1: Clone this repo
```bash
git clone https://github.com/SHI-Labs/VIM.git
```

Step 2: Create conda environment
```bash
conda create --name vim python=3.9
conda activate vim
```

Step 3: Install pytorch and torchvision

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

## Data Preparation
* [VIM50](https://drive.google.com/drive/folders/1gYtZd66qeCA4JWdbguRaWecG90aqfvs5?usp=sharing)
* [MTRCNN masks](https://drive.google.com/drive/folders/1gYtZd66qeCA4JWdbguRaWecG90aqfvs5?usp=sharing)
* [SeqFormer masks](https://drive.google.com/drive/folders/1gYtZd66qeCA4JWdbguRaWecG90aqfvs5?usp=sharing)
* [Checkpoints](https://drive.google.com/drive/folders/1gYtZd66qeCA4JWdbguRaWecG90aqfvs5?usp=sharing)

## Inference & Evaluation
Inference on the VIM50 with MTRCNN mask guidance:

```
CUDA_VISIBLE_DEVICES=0 python infer_vim_clip.py --config config/VIM.toml --checkpoint /path/to/msgvim.pth --image-dir /path/to/VIM50 --tg-mask-dir /path/to/MTRCNN/tg_masks/ --re-mask-dir /path/to/MTRCNN/re_masks/ --output outputs/MTRCNN_msgvim
```

Evaluation the results

```
CUDA_VISIBLE_DEVICES=0 python metrics_vim.py --gt-dir /path/to/VIM50 --output-dir /path/to/outputs/MTRCNN_msgvim
```

Inference on the VIM50 with MTRCNN mask guidance:

```
CUDA_VISIBLE_DEVICES=0 python infer_vim_clip.py --config config/VIM.toml --checkpoint /path/to/msgvim.pth --image-dir /path/to/VIM50 --tg-mask-dir /path/to/SeqFormer/tg_masks/ --re-mask-dir /path/to/SeqFormer/re_masks/ --output outputs/SeqFormer_msgvim
```

Evaluation the results

```
CUDA_VISIBLE_DEVICES=0 python metrics_vim.py --gt-dir /path/to/VIM50 --output-dir /path/to/outputs/SeqFormer_msgvim
```

## Citation

```
@article{li2023vim,
      title={Video Instance Matting}, 
      author={Jiachen Li and Roberto Henschel and Vidit Goel and Marianna Ohanyan and Shant Navasardyan and Humphrey Shi},
      journal={arXiv preprint},
      year={2023},
}
```

## Acknowledgement

This repo is based on [MGMatting](https://github.com/yucornetto/MGMatting). Thanks for their open-sourced works.
