from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("IMDBSentimentAnalysis").getOrCreate()

# Read the data as a single column
data = spark.read.option("header", "true").text("movie.csv")


# Use regular expression to extract text and label
data = data.withColumn("text", regexp_extract(data["value"], '^"(.+?)"', 1))
data = data.withColumn("label", regexp_extract(data["value"], '(\d)$', 1))

# Drop the original value column
data = data.drop("value")

# Show the data
# data.show(truncate=False)

data.show()
data.describe().show()

# Data preprocessing
tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stop_remove = StopWordsRemover(inputCol="token_text", outputCol="stop_tokens")
hashing_tf = HashingTF(inputCol="stop_tokens", outputCol="hash_token")
idf = IDF(inputCol="hash_token", outputCol="idf_token")

# Set up model
pos_neg_to_num = StringIndexer(inputCol="label", outputCol="indexedLabel")
lr = LogisticRegression(featuresCol="idf_token", labelCol="indexedLabel")

pipeline = Pipeline(stages=[pos_neg_to_num, tokenizer, stop_remove, hashing_tf, idf, lr])

# Train and test
train_data, test_data = data.randomSplit([0.7, 0.3])
trained_model = pipeline.fit(train_data)
results = trained_model.transform(test_data)

# Evaluation
eval = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="indexedLabel")
auc = eval.evaluate(results)
print(f"AUC: {auc}")
