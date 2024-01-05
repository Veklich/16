
# Преобразуем колонки в вектор
from pyspark.ml.linalg import Vectors, VectorUDT

cols_to_vector = F.udf(lambda l: Vectors.dense(l), VectorUDT())
df.withColumn('features', cols_to_vector(F.array(*['feature1', 'feature2'])))

# Разбиваем на трейн и тест
train, test = df.randomSplit([0.7, 0.3])

# Обучаем модель
from pyspark.ml.regression import RandomForestRegressor

lr = RandomForestRegressor(labelCol='target')
lr = lr.fit(train)
train_predictions = lr.transform(train)
test_predictions = lr.transform(test)

# Импортируем библиотеки
from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric

# Создаём спарк-сессию
spark = SparkSession.builder \
    .appName('Spark_app') \
    .getOrCreate()

#Считываем данные
# Ниже для тех, у кого хоть раз были просрчоки больше 60 дней, ставим в таргет 1.
# Загружаем данные
data = spark.read.csv("application_record.csv",  encoding = 'utf-8')
record = spark.read.csv("credit_record.csv", encoding = 'utf-8')

# Добавляем срок кредита к параметрам выдачи кредита
begin_month = record.groupBy("ID").agg(F.min(F.col("MONTHS_BALANCE")).alias("begin_month")).withColumn("begin_month", F.col("begin_month") * -1)
new_data = data.join(begin_month, on="ID", how="left")

# # Больше 60, то это просрочка, ставим - Yes, если просрочка есть за срок кредита,то так же ставим Yes
record = spark.read.csv("/content/application_record.csv", header=True)

# Создаем новый столбец 'dep_value' и устанавливаем его значение в зависимости от условий
record = record.withColumn('dep_value', F.lit(None))
record = record.withColumn('dep_value', F.when(record['STATUS'] == '2', 'Yes').otherwise(record['dep_value']))
record = record.withColumn('dep_value', F.when(record['STATUS'] == '3', 'Yes').otherwise(record['dep_value']))
record = record.withColumn('dep_value', F.when(record['STATUS'] == '4', 'Yes').otherwise(record['dep_value']))
record = record.withColumn('dep_value', F.when(record['STATUS'] == '5', 'Yes').otherwise(record['dep_value']))

# Группируем по 'ID' и считаем количество записей
cpunt = record.groupby('ID').agg(F.count('dep_value').alias('dep_count'))

# Устанавливаем значения 'dep_value' в зависимости от количества записей
cpunt = cpunt.withColumn('dep_value', F.when(cpunt['dep_count'] > 0, 'Yes').otherwise('No'))

# Джойним все данные вместе
new_data = new_data.join(cpunt.select('ID', 'dep_value'), on="ID", how="inner")

# Заменяем значения 'Yes' и 'No' на 1 и 0 в столбце 'dep_value'
new_data = new_data.withColumn('target', F.when(new_data['dep_value'] == 'Yes', 1).otherwise(F.when(new_data['dep_value'] == 'No', 0)))

# Удаляем столбец 'dep_value'
new_data = new_data.drop('dep_value')

# Выводим первые строки нового датасета с помощью PySpark
new_data.show()

# Оставим только часть признаков
selected_features = ['AMT_INCOME_TOTAL', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN']
selected_target = ['target']

# Выбираем только заданные признаки и целевую переменную
dataset = new_data.select(selected_features + selected_target)

# Преобразуем целевую переменную в числовой формат
dataset = dataset.withColumn(selected_target[0], dataset[selected_target[0]].cast('double'))

# Преобразуем данные векторный формат для обучения модели
vector_assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
input_data = vector_assembler.transform(dataset)

# Разделим выборку на обучающую и тестовую
(train_data, test_data) = input_data.randomSplit([0.7, 0.3], seed=42)

# Применяем StringIndexer для преобразования категориальных признаков в численные
string_indexer = StringIndexer(inputCol="CODE_GENDER", outputCol="CODE_GENDER_INDEX")
model = string_indexer.fit(train_data)
indexed_train_data = model.transform(train_data)

# Применяем OneHotEncoder для преобразования численных индексов в бинарные признаки
onehot_encoder = OneHotEncoder(inputCols=["CODE_GENDER_INDEX"], outputCols=["CODE_GENDER_ONEHOT"])
onehot_model = onehot_encoder.fit(indexed_train_data)
encoded_train_data = onehot_model.transform(indexed_train_data)

# Повторяем преобразования для тестовых данных
indexed_test_data = model.transform(test_data)
encoded_test_data = onehot_model.transform(indexed_test_data)

# Собираем численные признаки в один вектор
vector_assembler = VectorAssembler(inputCols=["AMT_INCOME_TOTAL", "CNT_CHILDREN"], outputCol="numerical_features")
assembled_train_data = vector_assembler.transform(train_data)
assembled_test_data = vector_assembler.transform(test_data)

# Применяем MinMaxScaler для масштабирования численных признаков
scaler = MinMaxScaler(inputCol="numerical_features", outputCol="scaled_features")
scaler_model = scaler.fit(assembled_train_data)
scaled_train_data = scaler_model.transform(assembled_train_data)
scaled_test_data = scaler_model.transform(assembled_test_data)

# Объединяем преобразованные данные
X_train = scaled_train_data.join(encoded_train_data, on="ID", how="inner")
X_test = scaled_test_data.join(encoded_test_data, on="ID", how="inner")

# Создаем экземпляр модели логистической регрессии
lr = LogisticRegression(featuresCol='features', labelCol='target')

# Обучаем модель на обучающем датасете
lr_model = lr.fit(X_train)

# Вычисляем оценку модели на обучающем и тестовом датасетах
evaluator = BinaryClassificationEvaluator(labelCol="target")

train_score = evaluator.evaluate(lr_model.transform(X_train))
test_score = evaluator.evaluate(lr_model.transform(X_test))

print(f'Оценка модели на обучающем датасете: {train_score}, на тестовом датасете: {test_score}')