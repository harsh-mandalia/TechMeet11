import pandas as pd
import matplotlib.pyplot as plt

Saved_data = pd.read_csv('task3_8best.csv')

plt.figure(1)

# plt.subplot(3,2,1)
plt.plot(Saved_data['time'], Saved_data['z1'])
plt.plot(Saved_data['time'], Saved_data['z1_dot'])
plt.plot(Saved_data['time'], Saved_data['z2'])
plt.plot(Saved_data['time'], Saved_data['z2_dot'])
plt.legend(["z1","z1 dot", "z2","z2 dot"])
plt.grid()

plt.show()