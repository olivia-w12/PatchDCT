# PatchDCT: Patch Refinement for High Quality Instance Segmentation(ICLR 2023)
> [**PatchDCT: Patch Refinement for High Quality Instance Segmentation**]
> Qinrou Wen, Jirui Yang, Xue Yang, Kewei Liang
> 
> *arXiv preprint([arXiv:2302.02693](https://arxiv.org/abs/2302.02693))*

In this repository, we release code for PatchDCT in Detectron2. 

## Contributions
- PatchDCT is the fist compressed vector based multi-stage refinement framework.
- By using a classifier to refine foreground and background patches, and predicting an informative low-dimensional DCT vector for each mixed patch, PatchDCT generates high-resolution masks with
fine boundaries and low computational cost.


## Installation
#### Requirements
- PyTorch ≥ 1.8

This implementation is based on [detectron2](https://github.com/facebookresearch/detectron2). Please refer to [INSTALL.md](INSTALL.md). for installation and dataset preparation.

## Usage 
The codes of this project is on projects/PatchDCT/ 
### Train with multiple GPUs
    cd ./projects/PatchDCT/
    ./train.sh

### Testing
    cd ./projects/PatchDCT/
    ./test.sh
### Speed Testing
    cd ./projects/PatchDCT/
    ./test_speed.sh
### Upper Bound of Model Performance(Table 1 in the paper)
    cd ./projects/PatchDCT/
    ./test_up.sh
    
For Swin-B backbone, use train_net_swinb.py instead of train_net.py
## Model ZOO 
### Trained models on COCO
Model |  Backbone | Schedule | Multi-scale training | FPS | AP (val) | Link
--- |:---:|:---:|:---:|:---:|:---:|:---:
PatchDCT | R50 | 1x | Yes |   12.3 | 37.2  | [download](https://1drv.ms/u/s!AnjAyCPH6yfahX01VBk0dmCXk7sm)
PatchDCT | R101 | 3x | Yes |  11.8 | 40.5 | [download](https://1drv.ms/u/s!AnjAyCPH6yfahXP4vPsUhEkSyTJ4)
PatchDCT | RX101 | 3x | Yes |   11.7 | 41.8  | [download](https://1drv.ms/u/s!AnjAyCPH6yfahXUDaBrVCGFheE8G)
PatchDCT | SwinB  | 3x | Yes |   7.3 | 46.1  | [download](https://1drv.ms/u/s!AnjAyCPH6yfahXfrnZBNRNkvDgBp)



### Trained models on Cityscapes
Model |Data|  Backbone | Schedule | Multi-scale training | AP (val) | Link
--- |:---:|:---:|:---:|:---:|:---:|:---:
PatchDCT | Fine-Only | R50 | 1x | Yes | 38.2 | [download](https://1drv.ms/u/s!AnjAyCPH6yfahW7NhlkFeGI-GtqP)
PatchDCT | COCO Pretrain+Fine | R50 | 1x | Yes | 40.3  | [download](https://1drv.ms/u/s!AnjAyCPH6yfahXvFBzn4B4brE-vB)


#### Notes
- We observe about 0.2 AP noise in COCO.
- The inference time is measured on NVIDIA A100 with batchsize=1.
- [Lvis 0.5](https://) is used for evaluation.

## Contributing to the project
Any pull requests or issues are welcome. 

If there is any problem with this project, please contact [Qinrou Wen](qinrou.wen@zju.edu.cn).

## Citations
Please consider citing our papers in your publications if the project helps your research. 

## License
- MIT License.
