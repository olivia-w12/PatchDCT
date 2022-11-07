# Response to Reviewer NpMa (8: accept, good paper)
**Q1: Can the paper describes more on the speed advantages compared to previous SOTA methods? What's the speed of using one-stage PatchDCT and two-stage PatchDCT respectively?**
**A1:**
**A1.1**
**A1.2:**
The speeds using one-stage PatchDCT and two-stage PatchDCT with R50-FPN are shown in the Table below:
|Method|AP|AP$^*$|(G)FLOPs|FPS                        
|----------------|-------------------------------|-----------------------------|-----------------------------|-----------------------------|
|one-stage|37.2|40.8|5.1|11.8
|two-stage|37.4|41.2|9.6|8.4
We observe that although two-stage PatchDCT achieves a certain improvement over one-stage PatchDCT, the computational cost increases and the inference speed reduces.
 

**Q2: What are the typical failure/bad cases of the proposed methods?**
**A2:**
In the process of visualization, we observe that mixed patches with extremely unbalanced propotion of foreground and background pixels may be hard samples for the three-class classifier. For example, it's difficult for the model to distinguish an  $8\times8$ mixed patch with only 10 foreground pixels from background patches around it. The phenomenon is because there is only slight difference between these patches, but we artificially divide them into two categories. The misclassification of such patches has only little influence on the overall segmentation performance, but may generate unsmooth boundaries in some cases.

# Response to Reviewer feWT (8: accept, good paper)
**Q1:   There is only runtime result compared with Mask-RCNN and DCT-Mask. Please complement more experiments to compare the efficiency of PatchDCT with other refinement models.**
**A1:**

**Q2: In this paper, the result in Table 1 suggests that when using 1x1 patch and 1-dim DCT vector the network has the best performance (57.6 AP). But when encoding 1x1 patch (single-pixel) using DCT, the result should be the value of the pixel itself. What is the difference between this method and directly refining the mask with 1x1 conv when the patch size is 1x1? I think this result is inconsistent with DCT-Mask, nor "binary grid refinement". According to DCT-Mask (Table 1), directly increasing the resolution decreases the mask AP, which is the main reason they use DCT encoding.**

**A2:**
In Table 1 of PatchDCT, we measure mAP on COCO val using ground truth information. The first row($1\times 1$ patch and $dim=1$) means that we directly replace predicted masks by ground truth masks in Mask-RCNN. The third row ($8\times 8$ patch and $dim=6$) means we seperate ground truth masks in $8\times 8$ patches and encode mixed patches in $6$-dimensional DCT vectors, then we obtain masks of mixed patches by decoding these DCT vectors. Therefore, the mAP in Table 1 can be viewed as the prediction upper bound of each setteing, namely the model performance when predicting all GT ground truth values correctly. Therefore, although with $1\times 1$ patch and $dim=1$ the prediction upper bound of the model is the highest(57.6 AP), the upper bound is almost impossible to achieve unless the model predict all ground truth values exactly. However, as discribed in DCT-Mask, predicting $112\times 112$ masks with the binary grid mask representation results in the performance degradation due to the high training complexity, i.e $112\times112 =12544$ values to predict. This is also proved by the poor model performance (35.7mAP) obtained by  binary grid refinement in Section 4.  Thus our results are consistent with DCT-Mask. PatchDCT uses patch DCT vectors to reduce the training complexity.  For example, dividing $112\times 112$ mask into $8\times8$ patches and selecting the patch DCT dimension as $6$, the model only needs to predict at most $14\times14\times 6 = 1176$ values.

# Response to Reviewer K4c6 (5: marginally below the acceptance threshold)
**Q1: The paper introduces PatchDCT, which improves the quality of instance segmentation. The experiments show the competitive performance of the proposed method. The paper provides detailed information for reproduction. There some previous works also focus on the segmentation boundary, such as Fully Connected CRF in DeepLab[1], CRFasRNN[2]. Comparison to these methods maybe helpful.**
**A1:** 
In the latest version we refer to these related works in Section 2 and compare them with PatchDCT (refer to the red part of Section 2).

# Response to Reviewer FMns (6: marginally above the acceptance threshold)
**Q1:  Backbone is limited to CNN-based models. Vision transformer-based model which also uses a patching technique would be of interest to see whether they will make any difference to the conclusions.**
**A1**

**Q2: In table 4, with the same backbone R101-FPN, SOLQ seems to have a better performance than PatchDCT with R101-FPN. However, the authors did not give any clarification in the paper.**
**A2:**

**Q3:  For the classifier given in the paper in Figure 2, if a foreground patch is assigned to the background, will this patch make a rectangle hole in the object?**
**A3:**
 To our knowledge we find no rectangle holes in the visualization progress on COCO 2017val. We note that Mask RCNN is already able to predict approximate binary masks, and a main reason of its limited performance is due to the imprecise segmentation around instance boundaries(refer to [1]). Also shown in Table 1 in [1], correcting mis-predicted pixels  within 1px of the boundary will give a boost to model performance, i.e 36.4 AP to 45.8 AP. Mask-RCNN predict $28\times28 =784$ values, while our three-class classifier has only $14\times 14\times 3 = 588$ values to predict. The training  complexity is not high and we have masks decoded from $300$-dimensional DCT vectors to provide auxiliary information. Therefore, we have reason to believe that the model is able to avoid the risk of rectangle holes caused by the three-category prediction.
 
 [1]Tang C, Chen H, Li X, et al. Look closer to segment better: Boundary patch refinement for instance segmentation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 13926-13935.

**Q4: If  $N$  and  $n$  are the same notation, please make them consistent.**
**A4:** 
$N$ is the dimension of the DCT vector of the entire mask and $n$ is the dimension of patch DCT vectors. We set $N=300$ and $n=6$ in the paper. Therefore, $N$ and $n$ are not the same notation.
