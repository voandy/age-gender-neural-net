import pandas as pd
import numpy as np
from keras.models import load_model

# Load model and scaling data
model = load_model("model/trained_model.h5")
scaling_df = pd.read_csv("model/scaling.csv")

# Slice scaling data into X and Y
scaling_X_df = scaling_df.iloc[:,3:]
scaling_Y_df = scaling_df.iloc[:,1:3]


# Scales data
def scale(scaling, field_name, value):
  return (value * scaling[field_name][0]) + scaling[field_name][1]


# Unscales data
def unscale(scaling, field_name, value):
  return (value - scaling[field_name][1]) / scaling[field_name][0]


# Questionnaire
X = []

print("MUSIC PREFERENCES\n")
X.append(int(input("Pop: Don't enjoy at all 1-2-3-4-5 Enjoy very much: \n")))
X.append(int(input("Opera: Don't enjoy at all 1-2-3-4-5 Enjoy very much: \n")))

print("MOVIE PREFERENCES\n")
X.append(int(input("Horror movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much: \n")))
X.append(int(input("Romantic movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much: \n")))
X.append(int(input("Sci-fi movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much: \n")))
X.append(int(input("Documentaries: Don't enjoy at all 1-2-3-4-5 Enjoy very much: \n")))

print("HOBBIES & INTERESTS\n")
X.append(int(input("History: Not interested 1-2-3-4-5 Very interested: \n")))
X.append(int(input("Politics: Not interested 1-2-3-4-5 Very interested: \n")))
X.append(int(input("Cars: Not interested 1-2-3-4-5 Very interested: \n")))
X.append(int(input("Art: Not interested 1-2-3-4-5 Very interested: \n")))
X.append(int(input("Gardening: Not interested 1-2-3-4-5 Very interested: \n")))
X.append(int(input("Celebrity lifestyle: Not interested 1-2-3-4-5 Very interested: \n")))
X.append(int(input("Theatre: Not interested 1-2-3-4-5 Very interested: \n")))

print("PHOBIAS\n")
X.append(int(input("Darkness: Not afraid at all 1-2-3-4-5 Very afraid of: \n")))
X.append(int(input("Spiders: Not afraid at all 1-2-3-4-5 Very afraid of: \n")))
X.append(int(input("Ageing: Not afraid at all 1-2-3-4-5 Very afraid of: \n")))

print("HEALTH HABITS\n")
X.append(int(input("Smoking habits: 0 Never smoked - 1 Tried smoking - 2 Former smoker - 3 Current smoker: \n")))
X.append(int(input("Drinking: 0 Never - 1 Social drinker - 2 Drink a lot: \n")))
X.append(int(input("I live a very healthy lifestyle.: Strongly disagree 1-2-3-4-5 Strongly agree: \n")))

print("PERSONALITY TRAITS, VIEWS ON LIFE & OPINIONS\n")
X.append(int(input("I look at things from all different angles before I go ahead.: Strongly disagree 1-2-3-4-5 "
                   "Strongly agree: \n")))
X.append(int(input("I damaged things in the past when angry.: Strongly disagree 1-2-3-4-5 Strongly agree: \n")))
X.append(int(input("I always try to vote in elections.: Strongly disagree 1-2-3-4-5 Strongly agree: \n")))
X.append(int(input("I try to give as much as I can to other people at Christmas.: Strongly disagree 1-2-3-4-5 "
                   "Strongly agree: \n")))
X.append(int(input("I cry when I feel down or things don't go the right way.: Strongly disagree 1-2-3-4-5 "
                   "Strongly agree: \n")))
X.append(int(input("How much time do you spend online?: 0 No time at all - 1 Less than an hour a day - "
                   "2 Few hours a day - 3 Most of the day: \n")))

print("SPENDING HABITS")
X.append(int(input("I save all the money I can: Strongly disagree 1-2-3-4-5 Strongly agree: \n")))
X.append(int(input("I spend a lot of money on my appearance.: Strongly disagree 1-2-3-4-5 Strongly agree: \n")))

# Convert answer to dataframe and scale for our model
X_df = pd.DataFrame(np.array(X), scaling_X_df.columns.values).T
for col in X_df:
  X_df[col][0] = scale(scaling_X_df, X_df[col].name, X_df[col][0])
  print(X_df[col][0])
  print(X_df[col].name)

# Convert scaled answer to numpy array to feed into our model
X_np = X_df.values
prediction = model.predict(X_np)[0]

# Print predictions
gender = "Female" if round(prediction[0]) == 0 else "Male"
age = unscale(scaling_Y_df, 'age', prediction[1])
print("The predicted gender is: {}".format(gender))
print("The predicted age is {}".format(age))