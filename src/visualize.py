import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visualize_csv(file):
    df = pd.read_csv(file)

    df['Acc_Mag'] = np.sqrt(df['X_Acc']**2 + df['Y_Acc']**2 + df['Z_Acc']**2)
    df['Gyro_Mag'] = np.sqrt(df['X_Gyro']**2 + df['Y_Gyro']**2 + df['Z_Gyro']**2)

    plt.plot(df['Acc_Mag'])
    plt.plot(df['Gyro_Mag'])
    plt.show()