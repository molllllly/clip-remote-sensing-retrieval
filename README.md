# CLIP Remote Sensing Retrieval

An experimental project using CLIP to retrieve remote sensing images from text descriptions on the RSICD dataset.

## Approach

- Use pretrained CLIP ViT-B/32 through OpenCLIP.
- Generate pseudo-labels using image–text similarity.
- Fine-tune with supervised and pseudo-label contrastive losses.
- Combine multiple remote sensing prompt templates for retrieval.
- Compare zero-shot and fine-tuned performance using Recall@1, Recall@5, and Recall@10.

## Main File

`semi+ multi tem17.ipynb` contains the training and evaluation pipeline.

## Usage

1. Install `torch`, `torchvision`, `open_clip_torch`, `Pillow`, `tqdm`, and Jupyter.
2. Download the RSICD images and annotations.
3. Set `ROOT_DIR`, `DATA_JSON`, and `RSICD_IMG_DIR` in the notebook.
4. Run the notebook to evaluate the pretrained model, generate pseudo-labels, and fine-tune.

The dataset and model weights are not included.

## Evaluation Notes

Retrieval scores are averaged across each image’s captions and prompt templates. Validation captions are also used to generate pseudo-labels, so validation results are not an independent held-out evaluation.

**Technologies:** Python, PyTorch, OpenCLIP, torchvision.
