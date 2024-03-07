import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('tempdata.csv')

clean_data = data.loc[data['node'] == 7]

temp = clean_data['temp']

inps = []
for inp in range(len(temp)):
    inps.append([inp])

x = np.array(inps)
y = np.array(temp)

split = int(len(x) * 0.8)

xtr, ytr = x[:split], y[:split]
xte, yte = x[split:], y[split:]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

model.compile(
    optimizer='adam',
    loss='mae'
)

model.fit(xtr, ytr, epochs=50)

preds = model.predict(xte)

print(mean_squared_error(yte, preds))

plt.plot(xtr, ytr, c='b')
plt.plot(xte, yte, c='g')
plt.plot(xte, preds, c='r')
plt.show()
