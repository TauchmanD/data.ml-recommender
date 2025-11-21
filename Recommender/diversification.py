from typing import Iterable, Optional, Sequence as Seq
import numpy as np
import pandas as pd

from Recommender.metric_functions import get_disagreements_based_on_order, get_overall_satisfaction
from .sequence import Sequence
from .group_aggregation_functions import get_group_agg_func


MOVIE_EMBEDDINGS: Optional[pd.DataFrame] = None

def load_genre_embedding(path: str = "ml-latest-small/movies.csv"):
    
    global MOVIE_EMBEDDINGS
    if MOVIE_EMBEDDINGS is not None:
        return MOVIE_EMBEDDINGS
    
    movies = pd.read_csv(path)

    all_genres = set()
    for g_string in movies["genres"]:
        if pd.isna(g_string):
            continue

        for genre in str(g_string).split("|"):
            if genre != "(no genres listed)":
                all_genres.add(genre)
    
    all_genres = sorted(all_genres)

    print(all_genres)

    rows = []
    for _, row in movies.iterrows():
        g_str = str(row["genres"])
        genres = [] if g_str == "nan" else g_str.split("|")
        vec = [1.0 if gen in genres else 0.0 for gen in all_genres]
        rows.append(vec)

    emb = pd.DataFrame(
        rows,
        index=movies["movieId"].astype(int),
        columns=all_genres,
        dtype=float,
    )

    vals = emb.to_numpy()
    norms = np.linalg.norm(vals, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    vals = vals / norms
    emb.iloc[:, :] = vals

    MOVIE_EMBEDDINGS = emb
    return emb


def movie_similarity(movie_a: int, movie_b: int, embeddings: pd.DataFrame) -> float:
    if movie_a not in embeddings.index or movie_b not in embeddings.index:
        return 0.0
    
    va = embeddings.loc[movie_a].to_numpy()
    vb = embeddings.loc[movie_b].to_numpy()
    return float(np.dot(va, vb))


def max_similarity_to(movie_id: int, others: Seq[int], embeddings: pd.DataFrame) -> float:
    if not others:
        return 0.0
    return max(movie_similarity(movie_id, other, embeddings) for other in others)


def seq_mmr_diversify(
        group_scores: pd.Series,
        history_items: Iterable[int],
        embeddings: pd.DataFrame,
        k: int,
        lambda_rel: float = 0.7,
        alpha_hist: float = 0.5,
        candidate_factor: int = 3
        ):
    
    if k <= 0:
        return []
    
    group_scores = group_scores.sort_values(ascending=False)

    pool_size = min(len(group_scores), candidate_factor * k)
    candidates = list(group_scores.index[:pool_size])

    max_rel = float(group_scores.iloc[0]) if pool_size > 0 else 1.0
    if max_rel == 0.0:
        max_rel = 1.0
    rel_norm = group_scores / max_rel

    history = list(dict.fromkeys(int(m) for m in history_items))
    selected: list[int] = []

    while candidates and len(selected) < k:
        best_item = None
        best_score = -1e9

        for m in candidates:
            rel = float(rel_norm[m])
            red_curr = max_similarity_to(m, selected, embeddings)
            red_hist = max_similarity_to(m, history, embeddings)

            redundancy = (1.0 - alpha_hist) * red_curr + alpha_hist * red_hist
            score = lambda_rel * rel - (1.0 - lambda_rel) * redundancy

            if score > best_score:
                best_score = score
                best_item = m

        selected.append(best_item)
        candidates.remove(best_item)

    return selected


class SIAADiversification(Sequence):
    def __init__(
            self,
            preferences: pd.DataFrame,
            b: float = 0.5,
            k: int = 10,
            lambda_rel: float = 0.7,
            alpha_hist: float = 0.5,
            candidate_factor: int = 3
    ):
        super().__init__(preferences)
        self.b = float(b)
        self.k = k

        self.lambda_rel = lambda_rel
        self.alpha_hist = float(alpha_hist)
        self.candidate_factor = int(candidate_factor)

        self.avg_agg = get_group_agg_func("dictator")
        self.avg_agg.set_dictator(91)
        self.repeat = False

        self.previous_recommendations: Optional[pd.DataFrame] = None

        self._embeddings = load_genre_embedding()

    def __next__(self) -> pd.DataFrame:
        if self.previous_recommendations is None:
            prefs_avail = self._available_prefs()
            first_full = self.avg_agg(prefs_avail)

            base_scores = first_full.iloc(0)
            selected_ids = seq_mmr_diversify(
                base_scores,
                [],
                self._embeddings,
                self.k,
                self.lambda_rel,
                self.alpha_hist,
                self.candidate_factor
            )
            top_k_cols = selected_ids
            
            first_round = first_full.columns[: self.k]
            self.recommended_items.update(top_k_cols)
            self._history = first_round.copy()
            self.previous_recommendations = first_round.copy()
            return first_round
        

        overall_sat = get_overall_satisfaction(self.preferences, self._history)
        penalties, _ = get_disagreements_based_on_order(self.preferences, self.previous_recommendations)
        per_user_disagreements = penalties.sum(axis=1)
        user_dis = per_user_disagreements
        if isinstance(per_user_disagreements, pd.DataFrame):
            user_dis = per_user_disagreements.iloc[:, 0]
        else:
            user_dis = per_user_disagreements

        w = (1.0 - self.b) * (1.0 - overall_sat) + self.b * user_dis
        w = w.reindex(self.preferences.index).fillna(0.0)

        prefs_avail = self._available_prefs()
        weighted_prefs = prefs_avail.mul(w, axis=0)
        group_scores = weighted_prefs.sum(axis=0).sort_values(ascending=False)

        if group_scores.empty:
            raise StopIteration
        
        selected_ids = seq_mmr_diversify(
            group_scores,
            self.recommended_items,
            self._embeddings,
            self.k,
            self.lambda_rel,
            self.alpha_hist,
            self.candidate_factor
        )
        top_k_cols = selected_ids

        full_new_round = group_scores.to_frame().T
        new_round = full_new_round.loc[:, top_k_cols]

        self.recommended_items.update(top_k_cols)

        round_idx = self._history.shape[0] + 1
        new_round.index = [f"siaa_{round_idx}"]

        self._history = pd.concat([self._history, new_round], axis=0, join="outer")
        self.previous_recommendations = new_round.copy()
        return new_round
        



    def _available_prefs(self) -> pd.DataFrame:
        """Return preferences restricted to movies that have NOT been recommended yet."""
        if self.repeat:
            return self.preferences

        if not self.recommended_items:
            return self.preferences
        cols = [c for c in self.preferences.columns if c not in self.recommended_items]
        return self.preferences[cols]


if __name__ == "__main__":
    print(max_similarity_to(14, [16,17,18], load_genre_embedding()))