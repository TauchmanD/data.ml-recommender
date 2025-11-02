# Recommender systems
## Part 1
a. DONE
b. Pearson correlation function is implemented in the `similarity_functions.py`
c. prediction function is implemented in the `recommender.py`
d. Popularity-Discounted, Significance-Weighted Pearson similarity. was implemented in the `similarity_functions.py` 
- Why "Significance-Weighted Pearson similarity"? It’s useful because it down-weights blockbuster items and shrinks tiny-overlap correlations, so my user–user similarities reflect true shared taste and produce steadier recommendations.
e. The user-based collaborative filtering that is generating the groups is implemented in the `recommender.py` in the function `get_predictions_for_group`, the average aggregation method is implemented in the `group_aggregation_functions.py`
The Least Misery aggregation method is implemneted in the `group_aggregation_functions.py`, both methods are tested in the `group_aggregation_pipeline.py`




# Working dataset acknoledgement:
> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>
