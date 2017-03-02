from sklearn import metrics
def evaluate(data, model):
  scaler, (_, _), (testX, testY), _ = data
  prediction = model.predict(testX)

  mean_absolute_error = metrics.mean_absolute_error(testY, prediction)
  mean_squared_error = metrics.mean_squared_error(testY, prediction)
  r2_score = metrics.r2_score(testY, prediction)

  return mean_absolute_error, mean_squared_error, r2_score
