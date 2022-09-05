import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = sns.load_dataset('iris')

import random
x_l = []
y_l = []

a = [9, 40, 7, 31, 20, 15, 18, 53, 7, 36, 36, 37, 48, 17, 22, 13, 34, 57, 40, 8, 42, 35, 56, 51, 7, 30, 34, 49, 9, 19, 58, 15, 28, 9, 19, 39, 51, 65, 5, 35, 48, 14, 12, 54, 63, 17, 8, 12, 12, 26, 19, 59, 61, 71, 45, 38, 57, 30, 52, 58, 60, 41, 59, 38, 39, 32, 38, 22, 37, 47, 44, 33, 34, 43]
b = [42, 30, 49, 38, 54, 41, 33, 47, 25, 48, 38, 50, 34, 19, 36, 23, 20, 49, 44, 34, 40, 60, 31, 72, 46, 63, 25, 52, 46, 47, 59, 63, 58, 51, 36, 47, 48, 58, 34, 41, 66, 63, 67, 71, 77, 24, 59, 57, 40, 48, 65, 56, 52, 56, 68, 60, 56, 37, 56, 57, 48, 69, 61, 57, 47, 65, 66, 54, 30, 44, 58, 46, 69, 67, 62, 51, 64]

for y in a:
    x_l.append('delta cohort')
    y_l.append(y)
for y in b:
    x_l.append('baseline cohort')
    y_l.append(y)


ax = sns.boxplot(x=x_l, y=y_l, order=['delta cohort', 'baseline cohort'])
ax.set_ylabel('Patients\' Age')
ax.set_ylim([0, 90])
# ax.set_title('Inpatient CT Progression Pattern')
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600
plt.savefig('/home/zhoul0a/Desktop/Figure_2.png')
plt.close()
exit()

ax = sns.violinplot(x=x_l, y=y_l, order=['delta cohort', 'baseline cohort'])
ax.set_ylabel('Age')
ax.set_ylim([0, 90])
# ax.set_title('Inpatient CT Progression Pattern')
plt.show()
plt.close()


exit()
sns.violinplot(x=df['species'], y=df['sepal_length'])
plt.show()
