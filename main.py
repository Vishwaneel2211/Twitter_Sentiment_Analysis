# =========================================
# IMPROVED TWITTER SENTIMENT ANALYSIS
# (Using Logistic Regression)
# =========================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, regexp_replace, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Start Spark
spark = SparkSession.builder.appName("ImprovedSentimentAnalysis").getOrCreate()

# 2. Load Dataset (NO HEADER)
data = spark.read.csv("tweets.csv", inferSchema=True)

# 3. Rename Columns
data = data.toDF("sentiment", "id", "date", "query", "user", "tweet")

print("Dataset Loaded Successfully")
data.printSchema()

# 4. Convert sentiment to binary (0 = negative, 1 = positive)
data = data.withColumn("sentiment",
    when(data["sentiment"] == 4, 1).otherwise(0)
)

# 5. Clean Text
data = data.withColumn("clean_text", lower(data["tweet"]))
data = data.withColumn("clean_text", regexp_replace("clean_text", "[^a-zA-Z\\s]", ""))

# 6. Tokenization
tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
data = tokenizer.transform(data)

# 7. Remove Stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
data = remover.transform(data)

# 8. Convert Text → Features (IMPROVED)
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=1000)
featurized = hashingTF.transform(data)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized)
data = idf_model.transform(featurized)

# 9. Convert Labels to Numeric
indexer = StringIndexer(inputCol="sentiment", outputCol="label")
data = indexer.fit(data).transform(data)

# 10. Logistic Regression Model (IMPROVED)
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
model = lr.fit(data)

# 11. Prediction
predictions = model.transform(data)

print("\nPredictions:")
predictions.select("tweet", "sentiment", "prediction").show(100, truncate=False)

# 12. Accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("\nImproved Model Accuracy:", accuracy)

# 13. Stop Spark
spark.stop()