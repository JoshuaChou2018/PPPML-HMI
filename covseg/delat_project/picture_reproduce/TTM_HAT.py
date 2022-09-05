import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = sns.load_dataset('iris')

import random
x_l = []
y_l = []

a = [2.169476175168695, 2.906415146917313, 3.42717650689743, 3.510599582245801, 3.775085633258732, 3.867821420061076, 4.221032914202098, 4.377812217167522, 4.395106627427128, 4.574590353399358, 4.604860855447649, 4.626073122698873, 4.696084297534316, 4.765672122653314, 4.855300751800316, 4.911799657127456, 5.047977293539129, 5.09130844083251, 5.197423768142961, 5.546858532663027, 6.1278212246389065, 7.001063002756405, 7.0893315650200135, 7.10068816657847, 7.553070021079694, 8.533348804331155, 10.044807151459295, 10.697645069736613, 11.919920636317867, 17.090934939564622]

a = np.array(a) * 2.70 / np.average(a)

b = [2.4557587076447636, 2.8995931801515553, 3.129347555939603, 3.189030601114874, 3.681622786356927, 4.682705517582669, 4.873239449500578, 5.10264952746517, 5.224158295347454, 5.244838249027886, 5.292088752501154, 5.621784431538316, 5.861088433595217, 6.006639134289271, 6.32841546423089, 6.438142373255571, 6.590705428169194, 7.0306626590339505, 7.032246761408768, 7.080946136099229, 7.168768769704469, 7.4742427594947465, 7.5979374356378795, 7.776609738762602, 8.305419671074242, 8.421384501782102, 10.766589617521475, 11.364692324394625, 11.387335939194717, 13.205507602373675, 13.465671086819473, 14.339873481848514, 15.347422188349388, 16.146531150757113]

b = np.array(b) * 4.96 / np.average(b)

c = [1.2698914063398288, 1.9819007205082544, 2.6515992334354994, 2.7751413651177077, 3.150487546074137, 3.3117735278624494, 3.350670373739322, 3.417621843620471, 3.4303598262706565, 3.5940429091467494, 3.595704269187846, 3.6339950086705763, 3.769638960963322, 4.045681083263005, 4.071770103385529, 4.076766673534227, 4.154902171717323, 4.2019248547669354, 4.292095628779709, 5.02690036406273, 5.135590895420658, 5.577512581447415, 5.7716757175251825, 5.874121859306439, 6.066286795513897, 6.084271300861923, 7.042881965558815, 7.14495978016138, 7.491279917159459, 9.044238159892593, 9.218304356650378, 11.071384534781588]

d = [2.444927830816619, 2.7183929607682433, 2.943039164633232, 3.031076610930251, 3.1395302956955264, 3.148326470876783, 3.31619539317375, 3.5599352716109536, 3.8838144497002975, 3.9558757715949673, 4.062191423274239, 4.413929496423435, 4.545868769483908, 4.7554813157776135, 5.068734923140555, 5.122534337597791, 5.26028687369554, 5.38194776489912, 5.679603145355074, 5.727163630328926, 5.74931094105369, 5.7856378314991765, 5.836348235609087, 6.111157443311852, 6.28655439181793, 6.305267853875966, 6.61287462950434, 7.170448071239139, 7.795441098009771, 7.926895937630118, 8.114014405083743, 8.798370697769045, 10.828969883621381, 13.265102331795802, 14.51226384467082, 16.55102034272741]


c = np.array(c) * 7.93 / np.average(c)
d = np.array(d) * 6.48 / np.average(d)

for y in a:
    x_l.append('TTM, delta')
    y_l.append(y)
for y in b:
    x_l.append('TTM, baseline')
    y_l.append(y)
for y in c:
    x_l.append('HAT, delta')
    y_l.append(y)
for y in d:
    x_l.append('HAT, baseline')
    y_l.append(y)


ax = sns.boxplot(x=x_l, y=y_l, order=['TTM, delta', 'TTM, baseline', 'HAT, delta', 'HAT, baseline'])
ax.set_ylabel('Days')
ax.set_title('Inpatient CT Progression Pattern')
plt.show()
exit()
sns.violinplot(x=df['species'], y=df['sepal_length'])
plt.show()