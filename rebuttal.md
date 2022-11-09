
# Response to Reviewer NpMa (8: accept, good paper)
**Q1: Can the paper describes more on the speed advantages compared to previous SOTA methods? What's the speed of using one-stage PatchDCT and two-stage PatchDCT respectively?**
**A1.1**

In the latest version we update the speed comparison with other SOTA methods and add supplemental description (refer to the red part in Table 4 and Section 4.3). Runtime is measured on a single A100. We present a brief version of Table 4 below to demonstrate the advantages of PatchDCT in speed.
|Method| Backbone |AP|FPS                        
|----------------|-------------------------------|-----------------------------|-----------------------------|
Mask RCNN|R101-FPN|38.8|13.8
DCT-Mask|R101-FPN|40.1|13.0
MaskTransfiner|R101-FPN|40.7|5.5
SOLQ|R101-FPN|40.9|10.7
HTC|RX101-FPN|41.2|4.7
PointRend|RX101-FPN|41.4|11.4
RefineMask|RX101-FPN|41.8|7.6
PatchDCT|R101-FPN|40.7|11.8
PatchDCT|RX101-FPN|42.2|11.7

All the models expect SOLQ are trained using '3x' schedules (~36 epochs) on COCO 2017val. SOLQ is trained using 50 epochs. The speed of PatchDCT is competitive compared to other multi-stage refinement methods.

**A1.2:**

The speeds using one-stage PatchDCT and two-stage PatchDCT with R50-FPN are shown in the Table below:
|Method|AP|AP$^*$|(G)FLOPs|FPS                        
|----------------|-------------------------------|-----------------------------|-----------------------------|-----------------------------|
|one-stage|37.2|40.8|5.1|12.3
|two-stage|37.4|41.2|9.6|8.4

We also update the speed of PachDCT for different stages in Table 10(refer to the red part in Table 10). We observe that although two-stage PatchDCT achieves a certain improvement over one-stage PatchDCT, the computational cost increases and the inference speed reduces.
 

**Q2: What are the typical failure/bad cases of the proposed methods?**

**A2:**

In the process of visualization, we observe that mixed patches with extremely unbalanced propotion of foreground and background pixels may be hard samples for the three-class classifier. For example, it's difficult for the model to distinguish an  $8\times8$ mixed patch with only 10 foreground pixels from background patches around it. The phenomenon is probably because there is only slight difference between these patches, but we artificially divide them into two categories. The misclassification of such patches has only little influence on the overall segmentation performance, but may generate unsmooth boundaries in some cases.

# Response to Reviewer feWT (8: accept, good paper)
**Q1:   There is only runtime result compared with Mask-RCNN and DCT-Mask. Please complement more experiments to compare the efficiency of PatchDCT with other refinement models.**

**A1:**

In the latest version of our paper, we update Table 4 with the speed of other SOTA methods measured on a single A100. We also add supplemental description in Section 4.3 (refer to the red part in Table 4 and Section 4.3). We present a brief version of Table 4 below  to demonstrate the efficiency of our model.
|Method| Backbone |AP|FPS                        
|----------------|-------------------------------|-----------------------------|-----------------------------|
Mask RCNN|R101-FPN|38.8|13.8
DCT-Mask|R101-FPN|40.1|13.0
MaskTransfiner|R101-FPN|40.7|5.5
SOLQ|R101-FPN|40.9|10.7
HTC|RX101-FPN|41.2|4.7
PointRend|RX101-FPN|41.4|11.4
RefineMask|RX101-FPN|41.8|7.6
PatchDCT|R101-FPN|40.7|11.8
PatchDCT|RX101-FPN|42.2|11.7

All the models expect SOLQ are trained using '3x' schedules (~36 epochs) on COCO 2017val. SOLQ is trained using 50 epochs.

**Q2: In this paper, the result in Table 1 suggests that when using 1x1 patch and 1-dim DCT vector the network has the best performance (57.6 AP). But when encoding 1x1 patch (single-pixel) using DCT, the result should be the value of the pixel itself. What is the difference between this method and directly refining the mask with 1x1 conv when the patch size is 1x1? I think this result is inconsistent with DCT-Mask, nor "binary grid refinement". According to DCT-Mask (Table 1), directly increasing the resolution decreases the mask AP, which is the main reason they use DCT encoding.**

**A2:**

In Table 1 of PatchDCT, we measure mAP on COCO val using **ground truth** information. The first row($1\times 1$ patch and $dim=1$) means that we directly **replace predicted masks by ground truth masks** in Mask-RCNN. The third row ($8\times 8$ patch and $dim=6$) means we seperate ground truth masks in $8\times 8$ patches and encode mixed patches in $6$-dimensional DCT vectors, then we obtain masks of mixed patches by decoding these DCT vectors.

The mAP in Table 1 can be viewed as **the prediction upper bound** of each setteing, namely **the model performance when predicting all ground truth values correctly**. Therefore, although with $1\times 1$ patch and $dim=1$ the prediction upper bound of the model is the highest(57.6 AP), **the upper bound is almost impossible to achieve** unless the model predict all ground truth values exactly. However, as discribed in DCT-Mask, predicting $112\times 112$ masks with the binary grid mask representation results in the performance degradation due to the high training complexity, i.e $112\times112 =12544$ values to predict. This is also proved by the poor model performance (35.7mAP) obtained by  binary grid refinement in Section 4.4.  PatchDCT effectively reduces training complexity and achieves a better performance by predicting at most 1176($14\times14\times 6$) outputs. Thus our results are consistent with DCT-Mask.

# Response to Reviewer K4c6 (5: marginally below the acceptance threshold)
**Q1: There some previous works also focus on the segmentation boundary, such as Fully Connected CRF in DeepLab[1], CRFasRNN[2]. Comparison to these methods maybe helpful.**

**A1:** 

In the latest version we refer to these related works in Section 2 and compare them with PatchDCT (refer to the red part of Section 2).

# Response to Reviewer FMns (6: marginally above the acceptance threshold)
**Q1:  Backbone is limited to CNN-based models. Vision transformer-based model which also uses a patching technique would be of interest to see whether they will make any difference to the conclusions.**

**A1**

**Q2: In table 4, with the same backbone R101-FPN, SOLQ seems to have a better performance than PatchDCT with R101-FPN. However, the authors did not give any clarification in the paper.**

**A2:**

**Q3:  For the classifier given in the paper in Figure 2, if a foreground patch is assigned to the background, will this patch make a rectangle hole in the object?**

**A3:**

 To our knowledge we find no rectangle holes in the visualization progress on COCO 2017val. Since the three-class classifier has only $14\times 14\times 3 = 588$ values to predict, the classification task is not complicated. In addition, masks decoded from 300-dimensional DCT vectors also provide auxiliary information for the refinement. Therefore, we have reason to believe that the model is able to avoid the risk of rectangle holes caused by the three-category prediction.


**Q4: If  $N$  and  $n$  are the same notation, please make them consistent.**

**A4:** 

$N$ is the dimension of the DCT vector of the entire mask and $n$ is the dimension of patch DCT vectors. We set $N=300$ and $n=6$ in the paper. Therefore, $N$ and $n$ are not the same notation.

**Q5:  About clarity: mentioned that "MaskTransifer runs at 5.5 FPS on the A100 GPU, which is almost two times slower than PatchDCT", however, the paper did not provide concrete data points.**

**A5:**
In our latest version, we update Table 4 with the speeds of other SOTA methods measured on a single A100 and give supplemental description about our advantage in speed in Section 4.3(refer to the red part in Table 4 and Section 4.3). With the same backbone, we measure the inference speed of MaskTransfiner on a single A100 GPU  and find its FPS equals to 5.5, while the FPS of PatchDCT is 11.8. We present a brief version of Table 4 to demonstrate the advantage of PatchDCT in speed.

|Method| Backbone |AP|FPS                        
|----------------|-------------------------------|-----------------------------|-----------------------------|
Mask RCNN|R101-FPN|38.8|13.8
DCT-Mask|R101-FPN|40.1|13.0
MaskTransfiner|R101-FPN|40.7|5.5
SOLQ|R101-FPN|40.9|10.7
HTC|RX101-FPN|41.2|4.7
PointRend|RX101-FPN|41.4|11.4
RefineMask|RX101-FPN|41.8|7.6
PatchDCT|R101-FPN|40.7|11.8
PatchDCT|RX101-FPN|42.2|11.7

All the models expect SOLQ are trained using '3x' schedules (~36 epochs) on COCO 2017val. SOLQ is trained using 50 epochs according to [1].

[1]Dong B, Zeng F, Wang T, et al. Solq: Segmenting objects by learning queries[J]. Advances in Neural Information Processing Systems, 2021, 34: 21898-21909.
