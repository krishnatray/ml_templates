###############################
# Keras classification template
# pima indian diabetes dateset
###############################

# sklearn
import pandas as pd
import numpy as np
import seaborn as sns
#import matplotlib

from sklearn.model_selection import train_test_split

#keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# load data
df = pd.read_csv("data/diabetes.csv")

# EDA
print(df.keys())
print("*" * 40)
print(df.info())
print("*" * 40)
print(df.describe().T)


# Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(df.drop("Outcome"))
X = df
y =
model.summary()

# Model
model = Sequantial()
model.add(Dense(32, input_shape=(8,), activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

learning_rate = 0.05
model.compile(Adam(lr=learning_rate), loss = 'categorical_crossentropy',
              metrics=['Accuracy'])

# data
X_train, y_train, X_test, y_test = train_test_split(X, y_cat,
                                                    random_state=123,
                                                    test_size=0.3)

model.fit(X_train, y_trains, epochs = 20, verbose=2, validation_split = 0.1 )



y_pred = model.predict(X_test)

y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy_score(y_test_class, y_pred_class)

print(classification_report(y_test_class, y_pred_class))

confusion_matrix(y_test_class, y_pred_class)
