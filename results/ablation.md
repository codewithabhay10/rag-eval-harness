
#### Ablation: retrieval strategy
| Config | Faithfulness | Context Precision | Context Recall |
|--------|--------------|-------------------|----------------|
| Dense only | 0.148 | 0.667 | 0.630 |
| Hybrid | 0.200 | 0.796 | 0.750 |

#### Ablation: reranker
| Config | Faithfulness | Context Precision |
|--------|--------------|-------------------|
| Reranker off | 0.148 | 0.667 |
| Reranker on | 0.219 | 0.665 |

#### Ablation: chunk size
| Chunk size | Faithfulness | Context Precision | Context Recall |
|------------|--------------|-------------------|----------------|
| 256 | 0.361 | 0.767 | 0.833 |
| 512 | 0.148 | 0.667 | 0.630 |
| 1024 | 0.296 | 0.445 | 0.310 |

#### Ablation: agentic vs vanilla
| Config | Faithfulness | Context Recall |
|--------|--------------|----------------|
| Vanilla | 0.148 | 0.630 |
| Agentic | 0.213 | 0.750 |