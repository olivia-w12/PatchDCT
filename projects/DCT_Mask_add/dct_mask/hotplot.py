from matplotlib import pyplot as plt  


def hotplot(pred_mask_logits,name="hotplot"):
    pred_mask_logits = pred_mask_logits.cpu().numpy()
    plt.imshow(pred_mask_logits)
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
