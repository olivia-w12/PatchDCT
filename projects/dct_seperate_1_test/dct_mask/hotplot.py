from matplotlib import pyplot as plt  


def hotplot(pred_mask_logits,name="hotplot"):
    pred_mask_logits = pred_mask_logits.cpu().numpy()
    plt.imshow(pred_mask_logits)
    plt.colorbar()
    plt.savefig(name)
    plt.close()

def hotplot2(pred_mask_logits,gt_mask,name="hotplot"):
    pred_mask_logits = pred_mask_logits.real.cpu().numpy()
    gt_mask = gt_mask.real.cpu().numpy()
    plt.subplot(321)
    plt.imshow(pred_mask_logits)
    plt.colorbar()

    plt.subplot(322)
    plt.imshow(gt_mask)
    plt.colorbar()



    plt.subplot(323)
    b_pred_mask_logits = pred_mask_logits
    b_pred_mask_logits[pred_mask_logits > 0.5] = 1
    b_pred_mask_logits[pred_mask_logits <= 0.5] = 0
    plt.imshow(b_pred_mask_logits)
    plt.colorbar()

    plt.subplot(324)
    b_gt_mask=gt_mask
    b_gt_mask[gt_mask>0.5]=1
    b_gt_mask[gt_mask<=0.5]=0
    plt.imshow(b_gt_mask)
    plt.colorbar()

    plt.subplot(325)
    plt.imshow(gt_mask - pred_mask_logits)
    plt.colorbar()

    plt.subplot(326)
    plt.imshow(b_gt_mask-b_pred_mask_logits)
    plt.colorbar()

    plt.savefig(name)
    plt.close()

def hotplot_number(pred_mask_logits,number=300):
    pred_mask_logits = pred_mask_logits[:, :number]
    pred_mask_logits = pred_mask_logits.cpu().numpy()
    name = "hotplot_{}".format(number)
    plt.imshow(pred_mask_logits)
    plt.colorbar()
    plt.savefig(name)
    plt.close()

def hp(mask11,mask22,name="plot/hotplot"):
    n = mask11.shape[0]
    mask11 = mask11.real.cpu().numpy()
    mask22 = mask22.real.cpu().numpy()
    for i in range(n):
        mask1 = mask11[i,:,:]
        mask2 = mask22[i,:,:]

        plt.subplot(221)
        plt.imshow(mask1)

        plt.subplot(222)
        plt.imshow(mask2)

        plt.subplot(223)
        mask3=mask1
        mask3[mask1>0.5]=1
        mask3[mask1<=0.5]=0
        plt.imshow(mask3)

        plt.subplot(224)
        mask4 = mask2
        mask4[mask2 > 0.5] = 1
        mask4[mask2 <= 0.5] = 0
        plt.imshow(mask4)


        plt.savefig(name+str(i))
        plt.close()
