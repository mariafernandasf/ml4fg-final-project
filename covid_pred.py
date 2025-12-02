import numpy as np
import pandas as pd

def aggregate_pca_by_individual(adata, pca_key="X_pca", agg="mean"):
    X = adata.obsm[pca_key]
    sample_ids = adata.obs["sampleID"].values

    individuals = np.unique(sample_ids)
    aggregated = []
    labels = []

    for person in individuals:
        idx = np.where(sample_ids == person)[0]
        person_cells = X[idx]

        if agg == "mean":
            vec = person_cells.mean(axis=0)
        elif agg == "median":
            vec = np.median(person_cells, axis=0)
        elif agg == "mean_var":
            vec = np.concatenate([person_cells.mean(axis=0),
                                  person_cells.var(axis=0)])
        else:
            raise ValueError("Unknown agg")

        aggregated.append(vec)
        labels.append(person)

    return np.array(aggregated), np.array(labels)
