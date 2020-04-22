 
import pandas as pd
import numpy as np

import joblib


joblib_model= joblib.load('models/classifier.pkl')

pred = pd.DataFrame(np.random.randint(0,10, size=(1,9)))

pred.at[0,0]=np.random.randint(150,160)
pred.at[0,1]=np.random.randint(10,14)
pred.at[0,2]=np.random.randint(4.4,6.7)

pred.at[0,3]=np.random.randint(20,260)
pred.at[0,4]=np.random.randint(0,1)
pred.at[0,5]=np.random.randint(45,51)

pred.at[0,6]=np.random.randint(23.5,29.6)
pred.at[0,7]=np.random.randint(90.2,101.2)
pred.at[0,8]=np.random.randint(48,55)

y_pred = joblib_model.predict(pred)
print(y_pred)