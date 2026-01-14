# Semantic Routing Memory (SRM)

**A Lightweight, Sustainable Long-Term Memory Layer for Agentic AI Systems**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Semantic Routing Memory (SRM)** is a novel memory architecture designed to solve the "Memory Wall" problem in long-lived autonomous agents. Instead of storing massive dense embedding matrices in hot memory (RAM), SRM uses **Vector Quantization (VQ)** to compress semantic locations into discrete routing codes.

By acting as a low-latency semantic filter, SRM replaces brute-force scans with efficient routing codes, drastically reducing the overhead of long-term memory retrieval in agentic systems.


---

## 🚀 Key Features

* **⚡ Extreme Compression:** Reduces hot memory footprint by **~50-600x** compared to dense baselines depending on configuration.
* **🧠 Semantic Continuity:** Maintains high recall using multi-probe routing ($m$-search) and lightweight reranking.
* **🛡️ Drift Robustness:** Static routing proves operationally more stable than naive adaptive updates under severe semantic drift.
* **🔌 Plug-and-Play:** Compatible with any embedding model (S-BERT, OpenAI, Cohere).
* **⚙️ Production Ready:** Includes unit tests, CI/CD workflow, and modular package structure.

---

## 📊 The "Memory Wall" Benchmark (1 Million Items)

We pushed SRM to the limit with **1,000,000 vectors** ($d=768$). The results demonstrate why linear dense storage is unsustainable for long-term agents.

![Memory Scaling 1M](benchmarks/scalability_memory_wall_benchmark/srm_memory_wall_scaling_1M_three_lines.png)
*Figure 1: Memory footprint scaling from 1k to 1M items.*

| System | Memory Usage (1M items) | Note |
| :--- | :--- | :--- |
| **Dense Index (Flat)** | **~2.93 GB** | Linear explosion. Unsustainable for edge/hot memory. |
| **SRM (Standard)** | ~43.15 MB | Python list overhead. Efficient but optimized further. |
| **SRM (Packed uint32)** | **~4.57 MB** | **~640x Reduction.** The optimized v0.2 routing layer. |

---

## 📦 Installation

Clone the repository and install dependencies:
```
git clone [https://github.com/AhmetYSertel/Semantic-Routing-Memory-System.git](https://github.com/AhmetYSertel/Semantic-Routing-Memory-System.git)
cd Semantic-Routing-Memory-System
pip install -r requirements.txt
```
💡 Quick Start
```python
import numpy as np
from srm import SemanticRoutingMemory, SRMConfig

# 1. Configuration
config = SRMConfig(
    d=768,          # Embedding dimension
    K=256,          # Codebook size (Sweet Spot for Recall/Compression)
    m=8,            # Multi-probe search count
    adaptive_update=False
)

# 2. Initialize SRM
memory = SemanticRoutingMemory(config)

# 3. Add Data (Simulated)
X_train = np.random.rand(10_000, 768).astype(np.float32)
ids = [f"mem_{i}" for i in range(10_000)]

# Fit & Index
memory.fit_codebook(X_train)
memory.add_items(X_train, ids=ids)

# 4. Query
query_vec = np.random.rand(768).astype(np.float32)
result = memory.query(query_vec, top_k=5)
print(f"Retrieved IDs: {result['ids']}")
```

🔬 Engineering Insights: Why K=256? (The Sweet Spot)
Through extensive parameter sweeping, we identified $K=256$ as the inflection point where semantic resolution maximizes before the routing layer becomes unnecessarily large. Increasing $K$ beyond 256 yields diminishing returns on Recall@10 while linearly increasing the routing layer size. Static vs. Adaptive Routing. Our drift benchmarks (benchmarks/drift_benchmark.py) revealed a counter-intuitive insight: Static routing is often safer.Under strict candidate budgets, naive online updates cause "bucket collapse" (centroids migrating into dense clusters), reducing recall. Static high-dimensional Voronoi cells provide better long-term stability.

🧪 Reproducibility
The repository includes a comprehensive benchmark suite.

Run Unit Tests: pytest tests/test_srm.py

Run 1M Scaling Benchmark: python benchmarks/benchmark.py

Run Drift Stress Test:python benchmarks/drift_benchmark.py

📜 Citation
If you use SRM in your research, please cite:

```
@article{sertel2026srm,
  title={Semantic Routing Memory (SRM): A Lightweight Long-Term Memory Layer for Agentic and Interactive Language Model Systems},
  author={Sertel, Ahmet Yiğit},
  journal={arXiv preprint},
  year={2026}
}
```


📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
