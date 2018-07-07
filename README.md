# Neural Network Age/Gender Predictor

![Diagram](https://i.imgur.com/W51g1mQ.png)

This is a simple neural network based on Keras and TensorFlow. It trains a model to predict a person's age and gender given their response to a number of lifestyle questions. E.g. what film genres they like, what they fear or what their interests are.

The training data was obtained from an open dataset on kaggle.com. https://www.kaggle.com/miroslavsabo/young-people-survey/home

Since the dataset consists purely of responses from Slovakian youths aged between 15 and 30 this model's accuracy in making predictions about the general population is greatly limited. E.g. the model rarely predicts an age that is less than 15 or greater than 30.

Even when evaluated against a subset of testing data from the same dataset the model's accuracy leaves a lot to be desired. Training a four-layer neural net over 100 epochs produces a model whose mean squared error is ~0.08 (for output variables scaled between 0 and 1).

![Diagram](https://i.imgur.com/A2YIl6j.png)

This is no more accurate than a standard multiple linear regression model might perform on the same dataset. 

Attempts to retrain the model on more serious hardware proved fruitless. With the neural netword expanded to 7 layers and trained over 100,000 epochs the mean squared error remained the same ~0.09.

Conclusion: Lifestyle factors such as interests and phobias are not good predictors of age and gender. There are simply too many outliers and too much overlap between the interests of the different age groups and genders. 

Note: The height and weight fields were intentionally excluded from the model as they are intutively much better predictors of age and gender (it would be like cheating).
