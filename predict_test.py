import pandas as pd
from keras.models import load_model

# Load model
model = load_model("model/trained_model.h5")

# Load scaling data
scaling_df = pd.read_csv("model/scaling.csv")


# Unscales data
def unscale(scaling, field_name, value):
  return (value - scaling[field_name][1]) / scaling[field_name][0]


# Load sample person to test model on
X = pd.read_csv("model/test_person.csv").values
prediction = model.predict(X)[0]

# Print predictions
gender = "Female" if round(prediction[0]) == 0 else "Male"
age = unscale(scaling_df, 'age', prediction[1])
print("The predicted gender is: {}".format(gender))
print("The predicted age is {}".format(age))