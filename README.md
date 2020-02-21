# Weather Prediction

> Weather Forecast using RNN LSTM

#### WEATHER PREDICTION FOR FARMERS 
> • Farmers in many parts of India are largely dependent on timely rainfall for harvest and subsequent profits. Uncertainty surrounding this phenomenon has also haunted them since the beginning of civilization. 
> • Over time however, this uncertainty had reduced significantly as farmers back in the day could almost accurately plant crops based on previous experience with weather conditions. Over time however, this uncertainty had reduced significantly as farmers back in the day could almost accurately plant crops based on previous experience with weather conditions. This wisdom has been passed on from one generation of farmers to the other. 
> • Consider the huge amount of data available in this sector and the tech behind machine learning, which could assist farmers in much better way

Our Idea is to basically turn this into a simple Machine Learning Regression problem
With data given for previous years weather and crop yield dependent upon these weather and soil conditions we will make predictions on what time to grow which crop to get maximum yield and predict the weather for the coming time period so that the farmer can plan accordingly and in advance
Features that are crucial for this scenario are amount of temperature, rainfall, soil acidity, natural anomalies that arise, carbon content, lighting and humidity
Additional features that we will create will depend upon the existing dataset
If data given is poor and skewed we will use polynomial features to better expand the data
If no such anomalies are present we will use Standard Scaling to clean the data
The constraints what we can face is the Browning Motion of weather i.e. whatever prediction we make will be based on past data and as we all know weather is fickle and will not follow an algorithm

An article can sum up best what I’m trying to convey in this draft:
https://www.ibm.com/developerworks/community/blogs/jfp/entry/Hindsight?lang=en

To get random predictions or alter preds:
```
python app.py
```