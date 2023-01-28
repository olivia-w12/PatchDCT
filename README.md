# PatchDCT: Patch Refinement for High Quality Instance Segmentation
> [**PatchDCT: Patch Refinement for High Quality Instance Segmentation**]
> Qinrou Wen, Jirui Yang, Xue Yang, Kewei Liang
>

In this repository, we release code for PatchDCT in Detectron2. PatchDCT is the fist compressed vector based multi-stage refinement framework.
By using a classifier to refine foreground and background patches, and predicting an informative low-dimensional DCT vector for each mixed patch, PatchDCT generates high-resolution masks with
fine boundaries and low computational cost.



## Installation
#### Requirements
- PyTorch ≥ 1.5 
- einops ≥ 0.6.0

This implementation is based on [detectron2](https://github.com/facebookresearch/detectron2). Please refer to [INSTALL.md](INSTALL.md). for installation and dataset preparation.

## Usage 
The codes of this project is on projects/PatchDCT/ 
### Train with multiple GPUs
    cd ./projects/PatchDCT/
    ./train.sh

### Testing
    cd ./projects/PatchDCT/
    ./test.sh
## Model ZOO 
### Trained models on COCO
Model |  Backbone | Schedule | Multi-scale training | FPS | AP (val) | Link
--- |:---:|:---:|:---:|:---:|:---:|:---:
PatchDCT | R50 | 1x | Yes |   12.3 | 37.2  | [download(Fetch code: xdma)]()
PatchDCT | R101 | 3x | Yes |  11.8 | 39.9  | [download(Fetch code: jkdf)]
PatchDCT | RX101 | 3x | Yes |   11.7 | 41.2  | [download(Fetch code: dfae)]
PatchDCT | SwinB  | 3x | Yes |   7.3 | 46.1  | [download(Fetch code: dafe)]

### Trained models on Cityscapes
Model |Data|  Backbone | Schedule | Multi-scale training | AP (val) | Link
--- |:---:|:---:|:---:|:---:|:---:|:---:
PatchDCT | Fine-Only | R50 | 1x | Yes | 37.0  | [download(Fetch code: faef)]
PatchDCT | CoCo-Pretrain +Fine | R50 | 1x | Yes |   |

#### Notes
- We observe about 0.2 AP noise in COCO.
- The inference time is measured on NVIDIA A100.
- [Lvis 0.5](https://) is used for evaluation.

## Contributing to the project
Any pull requests or issues are welcome. 

If there is any problem with this project, please contact [Qinrou Wen](qinrou.wen@zju.edu.cn).

## Citations
Please consider citing our papers in your publications if the project helps your research. 

## License
- MIT License.
