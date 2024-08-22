# Databricks notebook source
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Assuming df is your DataFrame and it has features 'feature1', 'feature2', ..., 'featureN' and target variable 'label'

# Assemble features
assembler = VectorAssembler(inputCols=["feature1", "feature2", "featureN"], outputCol="features")
df_assembled = assembler.transform(df)

# Split the data into training and test sets
train_data, test_data = df_assembled.randomSplit([0.7, 0.3])

# Define Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="label")

# Fit the model
lr_model = lr.fit(train_data)

# Predict on the test data
predictions = lr_model.transform(test_data)

# Display predictions
display(predictions.select("prediction", "label", "features"))
