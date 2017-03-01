from sklearn import metrics
def evaluate(data, model):
  scaler, (_, _), (testX, testY), _ = data
  prediction = model.predict(testX)

  mean_absolute_error = metrics.mean_absolute_error(testY, prediction)
  mean_squared_error = metrics.mean_squared_error(testY, prediction)
  r2_score = metrics.r2_score(testY, prediction)

  print("Mean Absolute Error: {}".format(mean_absolute_error))
  print("Mean Squared Error: {}".format(mean_squared_error))
  print("R2 Score: {}".format(r2_score))
  print
