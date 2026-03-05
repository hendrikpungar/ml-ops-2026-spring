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


FEATURE ADDITION ANALYSIS

I added passenger_count as a 4th feature to the regression model to see if it improves predictions.

a) How can you ensure that the improvement is not due to randomness or data leakage?

To verify the improvement is real and not just luck:

1. Cross-validation: I used 5-fold cross-validation instead of a single train/test split. This trains and tests the model 5 times on different data portions. If the improvement is consistent across all folds, it's likely real.

2. Statistical significance: Compare the mean R² scores and their standard deviations. If the confidence intervals don't overlap, the improvement is statistically meaningful.

3. Check for data leakage: Make sure the new feature (passenger_count) is available at prediction time and doesn't contain information about the target. Passenger count is recorded when the trip starts, before the fare is calculated, so no leakage here.

4. Use a holdout test set: Keep some data completely separate that wasn't used during any training or cross-validation.

b) What risks exist if this feature cannot be reliably generated in production?

If passenger_count is unreliable in production:

1. Missing values: If drivers don't always record passenger count, the model will fail or need a fallback strategy (like using a default value, which hurts accuracy).

2. Inconsistent recording: Different drivers might estimate passengers differently. Some might count children, others might not.

3. Self-reported data: Passengers might lie about how many people are in the cab to avoid surcharges.

4. Model degradation: If the feature quality in production is worse than in training data, model performance will drop without warning.

5. Dependency risk: If the system collecting passenger count goes down, the entire prediction pipeline breaks.

Mitigation: Build models that can gracefully handle missing features, or ensure the feature has strict quality controls in the data collection process.


LINK TO GITHUB:
https://github.com/hendrikpungar/ml-ops-2026-spring.git