import torch
import matplotlib.pyplot as plt


vector = torch.load("8_vector.pth",map_location="cuda:1")
dcc = vector[:,0]
dcc[(dcc>0)&(dcc<8)] = 4
count = plt.hist(dcc.cpu().numpy(),bins=8)[0]
plt.close()
count = [count[0],count[4],count[7]]
name = ["bg","mixed","fg"]
plt.bar(name,count)
x_t = list(range(3))
# plt.xticks(x_t,name)
plt.tick_params(labelsize=20)
plt.ticklabel_format(axis="y",style='sci', scilimits=(0,0))
plt.savefig("count.png")
print('----------------')