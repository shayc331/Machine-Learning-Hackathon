# Machine-Learning-Hackathon
This project was made during The Hebrew University's Machine Learning Hackaton.
Completed within less than 35 hours by a group of 4 programmers.

Project's Goal: predicting revenues and vote_averages of movies
Given data sets of movies with different kinds of features (numeric, free text, Json) we analysed and adjusted the data, and trained 4 models for
each feature to predict (LinearRegression, Lasso, Ridge, RandomForest).
From those we chose the best model (according to RMSE loss function) for each feature, and used Lasso model for achiving lower loss in each model.

Repo content:
1) sav files with the trained model ready to used
2) data sets for training and testing (zip)
3) Python code
