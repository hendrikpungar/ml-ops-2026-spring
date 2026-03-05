Model performance explanations


REGRESSION MODEL - Predicting taxi fare

I trained a linear regression model to predict the total fare amount using three features: trip distance, passenger count, and pickup location.

The model works very well with an R² of 0.89, meaning it explains 89% of fare variation. Trip distance does most of the heavy lifting here since taxi meters charge by distance.

Feature weights (coefficients):
- trip_distance: 3.14 - Each additional mile adds about $3.14 to the predicted fare
- DOLocationID: 0.0085 - Drop-off zone has a tiny effect on price
- PULocationID: 0.0005 - Pickup zone has almost no effect

The intercept (5.34) represents the base fare before distance is added - basically the flag drop fee.

The MAE of $2.89 means predictions are off by less than $3 on average - excellent for taxi fares that can range from $5 to $50+.

Why doesn't it predict perfectly? Because fares also depend on things we didn't include - time of day surcharges, tolls, tips, and traffic. But for a simple 3-feature model, it does the job.


CLASSIFICATION MODEL - Predicting payment type

This one was trickier. I tried to predict whether someone pays with credit card or cash using fare amount, trip distance, and drop-off location.

Results: 62% accuracy. Sounds okay, but here's the catch - about 60% of people pay with credit card anyway. So if we just guessed "credit card" every time, we'd be right 60% of the time. Our model is barely better than that.

The confusion matrix tells the real story. The model correctly identified 4547 credit card payments but wrongly labeled 2753 cash payments as credit card. It's basically biased toward predicting credit card.

Why is this so hard to predict? Because how someone pays is a personal choice. It depends on whether they carry cash, their habits, maybe cultural preferences - none of which we can see from trip data. The trip itself  doesn't really determine how someone wants to pay.
