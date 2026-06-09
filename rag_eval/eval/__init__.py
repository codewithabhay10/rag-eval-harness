"""eval — synthetic test-set generation, RAGAS scoring, retrieval metrics, ablation.

Phase 2 (done): build_testset.py + review_testset.py produce a labeled QA set;
ragas_eval.py + retrieval_metrics.py + run.py score a config (RAGAS faithfulness /
answer relevancy / context precision / context recall, plus Recall@k and MRR) and
save a traceable results file.

Phase 3 (todo): ablation.py sweeps configs and builds the comparison tables.
"""
