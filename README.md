# ML-Anamoly-Detection

## Goal:-
To develope ML pipelines for anamolies detection of time series data and configure modular code to be adopted for CI/CD functionality.

## Business Usage Scenarios:- 
Modern digital businesses generate large amounts of data from various sources and need a holistic, real-time system to detect anomalies and potential incidents to save money, improve performance, and identify new opportunities. Traditional monitoring tools often have built-in delays and operate independently of each department, making it challenging to associate anomalies with their potential impact.
Real-time monitoring and analysis of data patterns can identify unexpected changes that need investigation. For instance, an e-commerce company may detect an increase in online gift card purchases and a decrease in expected gift card revenue, indicating a potential issue like a price glitch that requires prompt attention.
Anomaly detection systems are crucial for businesses to manage the increasing volume of metrics and identify incidents that may affect revenue. Traditional manual detection methods, such as monitoring dashboards and setting alert thresholds, are prone to human error and can miss anomalies.

#### Marketing Analytics (Application):- Pricing Analytics, analysis of web traffic, user interaction , user conversion.

## Data description:-
Data Type :- Pollution particle measurement sensor's reading collected by Cairsense Denver (Govt of USA)

### Why Machine Sensor data ? if Marketing/ Customer Analytics data can also be used ?

Ans:- The main goal of this project for me to understand of basics the  time series data modelling, building a robust anamoly detection model and use indepth mathematical concepts/statistics for attacking a problem. As sensor data does not have any pre defined weightage to it as would be in case of any customer data it becomes a solely modelling problem without any implication of domian knowledge. Also, modelling a complete system for  checking anamolies in a unknown data is really challenging and would certainly help me to improve my basics of Machine learning.

### Data Source :- https://catalog.data.gov/dataset/cairsense-denver

<image src="https://user-images.githubusercontent.com/89546195/212576489-020f037b-9faa-49f9-8fe9-6546313e9b30.png" width=70% height=70%>
<image src="https://user-images.githubusercontent.com/89546195/212576657-36ff452c-66cd-4ce5-bb02-72035841a051.png" width=50% height=50%>

## Context:-
This project is based on a public dataset provided by "CAIRSENSE-Denver". The databases contain continuous sensor information as well as time stamped equivalent reference data. The aim of this project is to analyze the data to find implication of various natural aprameter such as tempreature, humidity, wind speed etc over a period of time. find some anamolies in the data using univariate and multivariate time series analysis method. A in depth time series analysis is done as a various unsupervised ML methods such as Isolation forest, Kmeans , SVM and comparing it with baseline statsistical maethods such as IQR and eucledian distance.

## Methodology:-
In attacking the problem of the Anamoly detection  one needs to  consider what type of data input we are using. Is it a batch processing data or a real time streaming data.
In this project we will address the problem by using data input based on both type of origin, that is a custom ML pipeline for anamoly detection of batch processing data and a ML pipeline for  real time data processing data. 

## Techstack used:-
### Git-version control, Tensorflow, Scikit-learn.
<image src="https://user-images.githubusercontent.com/89546195/225781842-22fb4c61-dfe6-46d8-b5b8-6b00dedee6e3.png" width=15% height=15%><image src="https://user-images.githubusercontent.com/89546195/225764006-ce83be94-53a6-4312-83a5-ff67b98788cf.png" width=20% height=20%> <image src="https://user-images.githubusercontent.com/89546195/225764357-ae84186d-8ad0-4e50-ba9d-d7bdf8e1f59f.png" width=20% height=20%> <image src="https://user-images.githubusercontent.com/89546195/225764601-6166b326-c5a1-4da1-8048-35586b9493bd.png" width=20% height=20%>

## This Github repository is managed using version control:-
 <image src="https://user-images.githubusercontent.com/89546195/225790576-59df3439-dee3-44ff-ad7a-bbb8d159a5c9.png" width=20% height=20%> 

## Installation:-
## Create a virtual enviorment (anaconda) & install dependency
1) open conda promt & enter:-
```
   code .
```
2) Run in terminal & Clone the github repository:-
```
   git clone https://github.com/prathameshk30/Multivariate-Anamoly-Detection
```
3) open terminal in vscode & type:-
```
   conda create -p myvenv python==3.9 anaconda -y
```
4) Activate the enviornment :-
```
   conda activate myvenv/
```
5) in terminal run setup file:-
```
  pip install -r requirements.txt
```

## All of the training and Analysis shall be done in this file
The link to complete project analysis for ML Algorithms:-
  
https://github.com/prathameshk30/Multivariate-Time-Series-Anamoly-Detect/blob/main/Jupter%20Notebooks/Anomaly_Detection%20Analysis.ipynb

