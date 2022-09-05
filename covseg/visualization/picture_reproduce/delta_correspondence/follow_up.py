import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = sns.load_dataset('iris')

import random
x_l = []
y_l = []

a = [4.1509359862480555e-06, 7.382312929824498e-05, 0.0006960114672998229, 0.0035143033142668732, 0.01094316647575658, 0.020378633877657426, 0.03018358404608332, 0.03923041376224448, 0.041733384356838134, 0.04323217335061413, 0.15544718204384872, 0.18065438601035216, 0.19863262620787328, 0.24402600354998172, 0.3153497614624078, 0.36497284050073686, 0.3658828088207444, 0.4418756233899741, 0.4521614829209883, 0.4588252769570094, 0.4703508045409063, 0.48295986330548185]

a = np.array(a) * 0.275 / np.average(a) * 60

b = [3.96728310525191e-05, 9.946633392947989e-05, 0.0006679443884688594, 0.002493327290387457, 0.011654487612631104,
    0.03774811078606688, 0.04627190054048219, 0.056429175660256306, 0.09088677664578783, 0.11139854373645958, 0.13667064933257847,
    0.1423207563953303, 0.15221714884935733, 0.2486181348050525, 0.254228647205447, 0.3313235442918575, 0.3400730254664427, 0.3469593852115563,
    0.454869214818623, 0.5014832916091555, 0.6567377241280014]

b = np.array(b) * 0.170 / np.average(b) * 60

c = [19, 36, 22, 14, 16, 23, 10, 20, 16, 22, 16, 30, 21, 17, 16, 21, 11, 16, 18, 13, 16, 12, 10, 6, 11, 6, 17, 14, 9, 12, 19, 19, 19, 6, 43]
c = c + [12, 28, 31, 25, 32, 17, 12, 20, 21, 20, 9, 26, 22, 23, 25, 21, 15, 19, 12, 30, 22, 27, 18, 19, 28, 6, 3, 21, 14, 22, 20, 8, 10, 15, 14, 17]
d = [28, 30, 9, 21, 23, 17, 28, 36, 17, 44, 12, 12, 13, 13, 39, 36, 13, 15, 15, 14, 33, 17, 18, 15, 35, 13, 13, 17, 17, 12, 32, 25, 29, 37, 27, 13, 12, 31, 17, 31,
     24, 13, 36, 20, 30, 15, 8, 10, 29, 35, 29, 29, 25, 28, 33, 18, 10, 27, 24, 31, 29, 34, 29, 30, 14, 6, 48, 48, 48, 47, 47, 46, 45]

c = np.array(c) * 17.7 / np.average(c)
d = np.array(d) * 24.9 / np.average(d)

x_t = []

for y in a:
    x_l.append(2)
    y_l.append(y)
    x_t.append('LRR, delta')
for y in b:
    x_l.append(3)
    y_l.append(y)
    x_t.append('LRR, baseline')
for y in c:
    x_l.append(1)
    y_l.append(y)
    x_t.append('First negative, baseline')
for y in d:
    x_l.append(0)
    y_l.append(y)
    x_t.append('First negative, delta')


ax = sns.boxplot(x=x_t, y=y_l, order=['First negative, delta', 'First negative, baseline', 'LRR, delta', 'LRR, baseline'])
ax.set_ylim([0, 50])
ax.set_ylabel('Days needed to first return negative')
ax.set_title('Short-term Follow-up Results')
ax2 = ax.twinx()
ax2.set_ylabel('Lesion ratio remained')
plt.show()
exit()
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600
plt.savefig('/home/zhoul0a/Desktop/Figure_1.svg')
plt.close()

exit()
sns.violinplot(x=df['species'], y=df['sepal_length'])

