---
toc: true
layout: post
description: Notes on machine learning model monitoring concepts, challenges and howto.
title:  "Machine Learning Model Monitoring"
date:   '2021-12-13'
categories: 
  - ML
  - data drift
  - model drift
  - MLOps

---

# Machine Learning Model Monitoring

![MLOps. You Desing It. Your Train It. You Run It.](./images/awesome-mlops-intro.png)

# Why monitoring matters:

Machine learning model monitoring is important as it allows to check for changes on the model performance. It is a cyclical and interactive process and need also to consider the monitoring of the infrastructure such as database and application.

 Model monitoring should account for:

- Data skews

- Model staleness

- Negative feedback loops

  

Functional and non-functional monitoring points are:

- Functional:
  - Predictive peformance
  - Changes in serving data
  - Metrics used during training
  - Characteristics of features
- Non-functional
  - System performance
  - System status
  - System reliability



# Concepts

### Data Skew:

 Data skews occurs when the model training data is not representative of the live data. There are several reasons for data skew, such as:

- Training data was designed wrong such as the distribution of the features in the training is different from the distribution of the features in real life data.
- Feature not available in production

### Model Staleness

Model staleness can occur based on:

- Shifts in the environment as historic data used during model training may change as time progress (e.g. financial models using time of recession might not be effective for predicting default when economy is healthy). 
- Consumer behaviour change such as trends in politics, fashion, etc.
- Adversarial scenarios where bad actors (e.g. criminals) seek to weaken the model. 

### Negative feedback loops

Negative feedback loop arises when you train data collected in production that can lead to bias.



### Model Decay

Production ML models often operation in dynamic environments (e.g. recommendation system of clothes need to change over time as the clothes style change over time.

If the. Model is static, it will move further away from the truth, issue known as Model drift. Model drift can be split in:

- **Data drift:** statistical properties of the input features changes. (e.g. distribution of age feature in a population over time). Real examples [here](https://www.bankofengland.co.uk/bank-overground/2021/how-has-covid-affected-the-performance-of-machine-learning-models-used-by-uk-banks) and [here](https://medium.com/eliiza-ai/why-your-models-might-not-work-after-covid-19-a00509e4920b#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6ImMxODkyZWI0OWQ3ZWY5YWRmOGIyZTE0YzA1Y2EwZDAzMjcxNGEyMzciLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2Mzk0MDEwOTIsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjExMDQ2Mzc1MjEzMjEzNDkzNDM0OCIsImVtYWlsIjoibWFyY2VsY2FzdHJvYnJAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImF6cCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsIm5hbWUiOiJNYXJjZWwgQ2FzdHJvIiwicGljdHVyZSI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdnbG94QXhza3ZfS01tWndGSGt2MzA0NDloZmtqdTBrU2hyQ3o1OT1zOTYtYyIsImdpdmVuX25hbWUiOiJNYXJjZWwiLCJmYW1pbHlfbmFtZSI6IkNhc3RybyIsImlhdCI6MTYzOTQwMTM5MiwiZXhwIjoxNjM5NDA0OTkyLCJqdGkiOiI3NzAxNjM5YjZiNTY5ZjY1ODk4MTIwOTZlNzg3ZWI3ZjI4MzVkYTA1In0.MF4MzS2sYN613RnhZ_79M1pr0LvheloeBZYjUkuOAyxGmXGubyKfEmHpz8YoLqDcDZb1y_h4i3woncCTyqjR9tIzxseAcW711QlMTn1liS_om4y7dcPhFXymho1i8Oxct1g7K1cKHZgjrdXX5b-S-0usbsb9_GtUS3kD4vKV7-lS3sz0JGXU87O6KiiPRPc1JS6FejJ7WPLCTAjNGTHEVNIolToE2ixhnZmtjuMgrjLfEkscn9YO1OpltLqXen7fQ1GKh28xhqR8cQc2td6E9NA9XRmVJiA4uXd9TJn5yM944_zs1O_IMAFABkUwtZYgPO2lhl2SkeBD1pxYwWottQ).
- **Concept drift:** occurs when the relationship between the features and labels changes. Examples are prediction drift and label drift. A real example [here](https://towardsdatascience.com/the-covid-19-concept-drift-using-sydney-ferry-activity-data-32bbff63cb9f).



# What and How to Monitor in ML models:



WHAT should we monitor in an ML model in production:

- Model input distribution
  - Errors: input values fall within an allowed set/range? 
  - Changes: does the distribution align with what was seen during training? 
- Model prediction distribution 
  - Statistical significance: e.g. if variables are normally distributes, we might expect the mean values to be within the standard euro of the mean interval.
- Model versions
- Input/prediction correlation



HOW should we monitor it:

- Tracing your ML model through logging. 
  - Observability of ML model while logging distributed tracings might be challenging. However, tools like Dapper, Zipkin and Jaeger could help to do the job.
- Detecting drift:
  - Check for statistical properties of the logged data, model predictions and possibly ground truth over time. Examples of tools that can be used are TensorFlow data validation (TFDV),  [scikit-multiflow library](https://scikit-multiflow.github.io/), or Google Vertex prediction.
  - What if Drift is detected:
    - Determine the portion of your training dataset that is still correct.
    - Keep good data and discard the bad.
    - Create an entirely new training dataset from the new data.
  - When to retrain my model:
    - On demand -> manual retrain  the model
    - On schedule -> when new labelled data is available at a daily/weekely/yearly basis
    - Availability of new training data -> new data is available on an ad-hoc basis.





# References:

[[1] Deploying Machine Learning in Production, Deeplearning](https://www.coursera.org/learn/deploying-machine-learning-models-in-production/lecture/Bew5j/why-monitoring-matters)

[[2] MLOps: What It Is, Why It Matters, and How to Implement It]()

[[3] Awesome MLOps](https://github.com/visenger/awesome-mlops)

[[4] Retraining Model During Deployment: Continuous Training and Continuous Testing](https://neptune.ai/blog/retraining-model-during-deployment-continuous-training-continuous-testing)

