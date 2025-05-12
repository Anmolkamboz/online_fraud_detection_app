# Import required libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import functions as F

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Online Fraud Detection") \
    .getOrCreate()

# Load CSV into Spark DataFrame
df = spark.read.csv("hdfs://path/to/onlinefraud.csv", header=True, inferSchema=True)

# Display the first few rows of the DataFrame
df.show()

# Check for null values
df.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in df.columns]).show()

# Drop duplicates
df = df.dropDuplicates()

# Drop unnecessary columns
df = df.drop('nameOrig', 'nameDest', 'isFlaggedFraud', 'step')

# Label Encoding for 'type' column using StringIndexer
indexer = StringIndexer(inputCol="type", outputCol="type_index")
df = indexer.fit(df).transform(df)

# Convert 'isFraud' to string labels
df = df.withColumn("isFraud", F.when(df['isFraud'] == 1, 'isfraud').otherwise('no fraud'))

# Select features and target variable
df = df.select('type_index', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'isFraud')

# Split data into train and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Set up the features and labels for training
feature_cols = ['type_index', 'amount', 'oldbalanceOrg', 'newbalanceOrig']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_data = assembler.transform(train_data)
test_data = assembler.transform(test_data)

# Initialize RandomForestClassifier
rf = RandomForestClassifier(labelCol="isFraud", featuresCol="features", numTrees=10)

# Fit the model
model = rf.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="isFraud", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy}")

# Save the model
model.save("hdfs://path/to/random_forest_model")

# Show classification report (precision, recall, F1 score)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="isFraud", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"F1 Score: {f1_score}")