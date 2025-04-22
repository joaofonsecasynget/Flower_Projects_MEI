
### 4.3 Comparative Analysis, Reflection, and Future Work

Both approaches showed relevant strengths: the decision tree model ensured direct interpretability and structural consistency, while the linear regression approach integrated post-hoc explainability techniques (SHAP and LIME) and demonstrated stable convergence. Table 1 summarizes their key differences and shared characteristics. More detailed configurations are described in Sections 4.1 and 4.2.

**Table 1: Summary of Key Differences Between Both Strategies**

| Feature                        | Decision Tree-Based Approach / Linear Regression with Explainability    |
|--------------------------------|---------------------------------------------------------------------------|
| **Model**                      | DecisionTreeRegressor (depth=10) / Linear Regression (Adam)              |
| **Clients**                    | Same dataset, independent preprocessing, containerized / Partitioned dataset, containerized |
| **Rounds**                     | 5 / 5                                                                     |
| **Training Strategy**          | Feature importance shared and aggregated / FedAvg aggregation             |
| **Explainability**             | Tree structure + feature importance / SHAP (global), LIME (local, fixed)  |
| **Monitoring**                 | JSON logs, structural similarity / RMSE, SHAP & LIME visualizations       |
| **RMSE (Client 1)**            | 66,234.26 / 235,232.00                                                    |
| **RMSE (Client 2)**            | 61,409.84 / 235,231.31                                                    |
| **Convergence**                | Structural similarity 0.95–0.97 / Stable RMSE after round 3               |
| **Training Time per Round**    | Not specified / 6–7 seconds                                               |
| **Privacy**                    | Only feature importance shared / Only model weights shared                |

Future work should include testing hybrid models combining decision trees and linear regression, incorporating additional aggregation strategies, expanding to more clients and data sources, introducing differential privacy and encryption techniques, and quantifying the convergence of explanations — for example, assessing the stability of SHAP and LIME across rounds.

Both strategies confirmed the feasibility of interpretable federated learning and the value of explainability for trustworthy AI systems. However, reducing data heterogeneity impact and reinforcing robustness of interpretability remain open challenges.
