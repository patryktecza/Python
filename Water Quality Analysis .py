#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

data = pd.read_csv("water_potability.csv")
data.head()


# In[19]:


data = data.dropna()
data.isnull().sum()


# In[23]:


plt.figure(figsize=(10,10))
sns.countplot(x='Potability',data=data)
plt.show()


# In[28]:


pip install plotly==5.18.0


# In[30]:


import plotly.express as px
data = data
figure = px.histogram(data, x = "ph", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: PH")
figure.show()


# In[31]:


figure = px.histogram(data, x = "Hardness", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Hardness")
figure.show()


# In[32]:


figure = px.histogram(data, x = "Solids", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Solids")
figure.show()


# In[33]:


figure = px.histogram(data, x = "Chloramines", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Chloramines")
figure.show()


# In[34]:


figure = px.histogram(data, x = "Sulfate", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Sulfate")
figure.show()


# In[35]:


figure = px.histogram(data, x = "Conductivity", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Conductivity")
figure.show()


# In[36]:


figure = px.histogram(data, x = "Organic_carbon", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Organic Carbon")
figure.show()


# In[37]:


figure = px.histogram(data, x = "Trihalomethanes", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Trihalomethanes")
figure.show()


# In[38]:


figure = px.histogram(data, x = "Turbidity", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Turbidity")
figure.show()


# In[47]:


pip install pycaret


# In[4]:


correlation = data.corr()
correlation["ph"].sort_values(ascending=False)


# In[7]:


from pycaret.classification import *
clf = setup(data, target = "Potability", remove_outliers = True, session_id = 786)
compare_models()


# In[8]:


model = create_model("rf")
predict = predict_model(model, data=data)
predict.head()


# In[ ]:




