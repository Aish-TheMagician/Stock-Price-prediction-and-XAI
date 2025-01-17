import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse

link="https://docs.google.com/spreadsheets/d/e/2PACX-1vSHmoEGdf3Oghhv-H4_rVe1b4vckt7I_ov1LvdPhtk1mR_wckbJrA8aPJJsjH4F4h2KscsJcF1k1BFX/pub?gid=0&single=true&output=csv"
df=pd.read_csv(link)

df=df[['date','close']]

t=df

def date_clean(s):
    s_dateonly=''
    for i in range(10):
        s_dateonly=s_dateonly+(s[i])
    return str(s_dateonly)

df['date']=pd.to_datetime(df['date'])

def date_only(s):
    s=s.date()
    return s

df['date']=df['date'].dt.date

df['x1']=df['close'].shift(-1)
df['x2']=df['x1'].shift(-1)
df['y']=df['x2'].shift(-1)

arr=np.array(df)

for i in range(1255):
    arr[i][0]=arr[i+3][0]
    
df=pd.DataFrame(arr)

df=df.dropna()

arr=np.array(df)

train,test=train_test_split(arr,test_size=0.3,random_state=1)
test,val=train_test_split(test,test_size=0.5,random_state=1)

train.shape

test.shape

train=pd.DataFrame(train)

test=pd.DataFrame(test)
val=pd.DataFrame(test)

train=train.set_index(train[0])

plt.scatter(train[0],train[4],s=3,color='blue')

plt.scatter(val[0],val[4],s=3,color='green')
plt.scatter(test[0],test[4],s=3,color='red')

x_train=train[[1,2,3]]
x_train=x_train.astype(np.float32)
y_train=train[4]
y_train=y_train.astype(np.float32)
x_test=test[[1,2,3]]
y_test=test[4]
x_val=val[[1,2,3]]
y_val=val[4]

x_test=np.array(x_test)
y_test=np.array(y_test)
x_train=np.array(x_train)
y_train=np.array(y_train)
x_val=np.array(x_val)
y_val=np.array(y_val)

x_test=x_test.astype(np.float32)
y_test=y_test.astype(np.float32)
x_val=x_val.astype(np.float32)
y_val=y_val.astype(np.float32)

model=Sequential(
    [
        Input(shape=(3,1)),
        LSTM(units=64),
        Dense(32,activation='relu'),
        Dense(16,activation='relu'),
        Dense(1)

    ]
)

model.compile(loss='mse',optimizer='adam',metrics=['mean_absolute_error'])

x_train=tf.convert_to_tensor(x_train)

model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=300,batch_size=50)

pred=model.predict(x_test)

loss, mae = model.evaluate(x_test, y_test)

pred=pd.DataFrame(pred)
pred['actual']=y_test
pred['absolute_error']=pred['actual']-pred[0]
pred.style
y_test.mean()

!pip install lime
import lime
from lime import lime_tabular
train['0'] = pd.to_numeric(train['0'])
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(train),
    feature_names=train.columns,
    class_names=['predicted_value'],
    discretize_continuous=True
)

# Generate explanations for a few predicted values
for i in range(5):
    explanation = explainer.explain_instance(
        data_row=x_test[i],
        predict_fn=model.predict
    )

    # Print the explanation
    print(f"Explanation for predicted value {i + 1}:")
    print(explanation.as_list())

!pip install lime
import lime
from lime import lime_tabular
# Assuming the column you want to convert is the first one (index 0)
train.iloc[:, 0] = pd.to_numeric(train.iloc[:, 0]) # Convert the first column to numeric
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(train),
    feature_names=train.columns,
    class_names=['predicted_value'],
    discretize_continuous=True
)

# Generate explanations for a few predicted values
for i in range(5):
    explanation = explainer.explain_instance(
        data_row=x_test[i],
        predict_fn=model.predict
    )

    # Print the explanation
    print(f"Explanation for predicted value {i + 1}:")
    print(explanation.as_list())

!pip install lime shap

import lime
import lime.lime_tabular
import shap

# LIME Explainability
# Convert x_train to a NumPy array and get column names if available
x_train_np = x_train.numpy()
feature_names = x_train.columns if hasattr(x_train, 'columns') else [f'feature_{i}' for i in range(x_train_np.shape[1])]

explainer = lime.lime_tabular.LimeTabularExplainer(
    x_train_np,
    feature_names=feature_names,
    class_names=['target'],
    verbose=True,
    mode='regression'
)

# Explain a single prediction
i = 0  # Index of the test sample to explain
exp = explainer.explain_instance(x_test[i], model.predict, num_features=5)
exp.show_in_notebook(show_table=True)

import shap
import numpy as np # Import numpy for array conversion

# Create a Shapley explainer
# Convert x_train to a NumPy array
explainer = shap.KernelExplainer(model.predict, np.array(x_train))

# Calculate Shapley values for the test data
shap_values = explainer.shap_values(x_test)

# Print the Shapley values for the first prediction
print("Shapley values for the first prediction:")
print(shap_values[0])

# Plot the Shapley values for all predictions
shap.summary_plot(shap_values, x_test)

