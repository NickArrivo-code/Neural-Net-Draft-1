# Neural-Net-Draft-1
Using the created scores from MLB-Hitter-Ratings to predict OPS through a Neural Net

I am using the Keras package in Python as part of the Tensorflow platform

This is a trial to see how effective the rating scores can be to predict future success for a player. 

For this trial, I am using power score, contact score, and discipline score to predict OPS. 
I think that this model could be improved upon using more data, which I am working to collect.
I am also interested to see how well a model could be trained with using the predictor features that went into creating the scores instead
of the scores themselves.

My mean squared error is ~0.004206. I am worried about overfitting the data to the training set and am looking to use an early stopping technique for future models.

![image](https://github.com/NickArrivo-code/Neural-Net-Draft-1/assets/137729712/5c5a6e5c-fdb4-4b7e-97db-682e1a5f71ce)

This is the plot of the predicted vs actual values for the test set. Clearly, it can be improved upon, but does provide some value as the mean squared error is relatively small. 
