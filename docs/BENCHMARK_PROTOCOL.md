# Benchmark protocol

## Primary task
Predict the dataset's `condition` label from the review text. The configured condition list is a project choice inherited from the original notebook and should be documented if changed.

## Primary metric
Macro-F1 is the model-selection metric because it gives each class equal weight. Also report accuracy, weighted-F1, macro recall, per-class recall, and confusion matrices.

## Data policy
- Raw datasets are local-only and ignored by Git.
- Record SHA-256 hashes for provenance.
- Keep a fixed final test set untouched for final reporting.
- Check exact review overlap across train/test and document it.
- Any deduplication, filtering, or label mapping must be identical across model comparisons.

## Model comparison
Classical baselines use TF-IDF fitted inside each pipeline and one classifier per pipeline. Compare all models on the same evaluation rows. Train-fraction curves use the same evaluation set and fixed random seed. For a publication-quality learning curve, repeat each fraction across multiple seeds and report mean and standard deviation.

## Deep learning
DistilBERT should use the same train/evaluation examples and label mapping as the classical baselines. The current starter script is intentionally a first implementation; adapt it to the shared split before treating direct score differences as a strict benchmark.

## Reporting
Never report a score without recording the model, dataset hash, split strategy, seed, and package versions. Don't interpret classification performance as evidence that a drug is safe or effective.
