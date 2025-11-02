# Recommender Systems

## Part 1

- **a.** DONE
- **b.** Pearson correlation function is implemented in `similarity_functions.py`.
- **c.** Prediction function is implemented in `recommender.py`.
- **d.** Popularity-Discounted, Significance-Weighted Pearson similarity is implemented in `similarity_functions.py`.
  - *Why "Significance-Weighted Pearson similarity"?*  
    It’s useful because it down-weights blockbuster items and shrinks tiny-overlap correlations, so user–user similarities reflect true shared taste and produce steadier recommendations.
- **e.** User-based collaborative filtering for generating groups is implemented in `recommender.py` (`get_predictions_for_group`).  
  The average aggregation method is implemented in `group_aggregation_functions.py`.  
  The Least Misery aggregation method is implemented in `group_aggregation_functions.py`.  
  Both methods are tested in `group_aggregation_pipeline.py`.

---

### Working Dataset Acknowledgment

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.  
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.  
<https://doi.org/10.1145/2827872>
