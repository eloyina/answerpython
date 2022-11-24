#this code applies pyspark
#
# import findspark
# findspark.init()
# import pyspark
# from pyspark.sql import SparkSession
# from pyspark.sql import Row
# from pyspark.sql import functions
# from pyspark.sql.types import *
# from pyspark.sql.functions import *
# from pyspark.sql.functions import col, udf
# from pyspark.sql.functions import isnan, when, count, col
# from pyspark.sql.functions import *

# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.regression import LinearRegression
# from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.ml.regression import DecisionTreeRegressor

#create a query in pyspark
file= spark.read.csv('D:/python/linearregression.csv', header=True, inferSchema=True)
file.show(3)
