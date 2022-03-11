import matplotlib.pyplot as plt
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])
unloader = transforms.ToPILImage()
def hotplot(inputs,outputs,name="plot/hotplot"):
    image = inputs[0]['image'].cpu()
    image = image.squeeze(0)
    image = unloader(image)
    instance1 = outputs[0][0]['instances']
    instance2 = outputs[1][0]['instances']
    gt_instance = outputs[2][0]['instances']
    mask11 = instance1.pred_masks
    mask22 = instance2.pred_masks
    gt_mask0 = gt_instance.pred_masks
    n = mask11.shape[0]

    for i in range(n):
        mask1 = mask11[i,:,:].cpu().numpy()
        mask2 = mask22[i,:,:].cpu().numpy()
        gt_mask = gt_mask0[i,:,:].cpu().numpy()

        plt.subplot(141)
        plt.imshow(mask1)

        plt.subplot(142)
        plt.imshow(mask2)

        plt.subplot(143)
        plt.imshow(gt_mask)

        plt.subplot(144)
        plt.imshow(image)

        plt.savefig(name+str(i))
        plt.close()

def hotplot2(mask11,mask22,name="plot/hotplot"):
    n = mask11.shape[0]

    for i in range(n):
        mask1 = mask11[i,:,:].cpu().numpy()
        mask2 = mask22[i,:,:].cpu().numpy()

        plt.subplot(121)
        plt.imshow(mask1)

        plt.subplot(122)
        plt.imshow(mask2)


        plt.savefig(name+str(i))
        plt.close()