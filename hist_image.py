import torch
import matplotlib.pyplot as plt
# vector = torch.load("./8/0.pth",map_location="cuda:0")
# for i in tqdm(range(1,4997)):
#     a = torch.load("./8/{}.pth".format(i),map_location="cuda:0")
#     vector = torch.cat((vector,a),dim=0)
# torch.save(vector,"8_vector.pth")
vector = torch.load("./8_vector.pth",map_location="cuda:0")
idx = vector[:,0]
vector_mixed = vector[(idx>0)&(idx<8),:6].reshape(-1)
vector_bg = vector[(idx==0),:6].reshape(-1)
vector_fg = vector[(idx==8),:6].reshape(-1)
# vector = vector.cpu().numpy()
vector_m = vector_mixed.cpu().numpy()
vector_bg = vector_bg.cpu().numpy()
vector_fg = vector_fg.cpu().numpy()

plt.figure(dpi=100,figsize=(12,7))
plt.hist(vector_bg,8,(0,8))
#plt.tick_params(labelsize=20)
#plt.ticklabel_format(style='sci', scilimits=(0,0))
plt.yticks(size=34)
plt.xticks(size=34)
plt.ylabel('number of elements',fontsize=34)
plt.savefig("1b.png")
plt.close()

plt.figure(dpi=100,figsize=(12,7))
plt.hist(vector_fg,bins=8)
plt.yticks(size=34)
plt.xticks(size=34)
plt.ylabel('number of elements',fontsize=34)
plt.savefig("1f.png")
plt.close()


plt.figure(dpi=100,figsize=(12,7))
plt.hist(vector_m,bins=8)
plt.yticks(size=34)
plt.xticks(size=34)
plt.ylabel('number of elements',fontsize=34)
plt.savefig("1m.png")
plt.close()

