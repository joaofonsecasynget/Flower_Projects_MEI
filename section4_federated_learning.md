### 4 Implementation and Evaluation of Federated Learning Strategies for Predictive Modeling

Building on the discussion in the previous section, *Strategies for Implementing Federated Learning and Explainability*, two distinct approaches were explored for predicting real estate prices in California, both implemented using the Flower federated learning framework.

The analysis conducted so far has helped to understand the benefits and challenges of implementing interpretable models in a federated learning context. However, several questions remain open, particularly regarding how to improve model stability, reduce discrepancies between clients, and optimize the interpretability of predictions without compromising performance. In this section, we describe two distinct strategies — a Decision Tree-based model and a Linear Regression model with integrated explainability — and reflect on their effectiveness based on empirical evaluation. The analysis conducted helped to identify strengths, limitations, and points for improvement.

---

#### 4.1 Decision Tree-Based Approach

The first implemented approach used a DecisionTreeRegressor from scikit-learn to predict real estate prices, leveraging its inherent interpretability and ease of analyzing the decision structure. The model was configured with a maximum depth of 10 and a fixed random state of 42 for reproducibility. Training was conducted over 5 federated rounds using two clients with independently preprocessed versions of the same dataset.

Each client computed the importance of features locally using the feature_importances_ attribute, which were then sent to the server for aggregation. These aggregated importances were redistributed to all clients, influencing the reconstruction of the decision trees in the next round. The model thus evolved iteratively, prioritizing the most relevant features while maintaining privacy, as raw data never left the clients.

The system maintained detailed logs per round, tracking RMSE, tree structure, and feature importance. Performance varied slightly between clients, with RMSE values of 66,234.26 (Client 1) and 61,409.84 (Client 2). Structural similarity scores ranged from 0.95 to 0.97, indicating consistent convergence across rounds. The interpretability of this approach was further enhanced by visualization of tree structures and analysis of feature relevance.

While this setup demonstrated effective collaboration and explainability, discrepancies in performance due to data heterogeneity revealed the need for improved aggregation strategies and possibly more advanced ensemble models.

---

#### 4.2 Linear Regression with Explainability Approach

The second approach employed a linear regression model implemented using PyTorch's nn.Linear module, optimized with Adam and using Mean Squared Error (MSE) as the loss function. The training process was structured and reproducible, relying on Docker containers to isolate environments and a centralized Flower server to coordinate model aggregation using FedAvg.

The training began with a client randomly selected to provide the initial model parameters. Across five rounds, the server distributed these parameters to the clients, who then trained locally, computed RMSE and loss, and generated explanations using SHAP and LIME. Updated weights were sent back to the server for aggregation.

The model demonstrated stable convergence, with RMSE values decreasing gradually from 235,232.42 in the first round to 235,231.31 in the final round, showing consistent improvement across all rounds. The progression was steady, with values of 235,232.14, 235,231.88, and 235,231.59 in rounds two, three, and four respectively, indicating a stable learning process.

To ensure comparability across rounds, SHAP explanations were generated using fixed random seeds, while LIME used a fixed instance selected at the start of training. This approach enabled precise tracking of how feature attributions evolved over time and clear identification of the most impactful features—median income and proximity to the ocean.

This approach succeeded in embedding interpretability into the federated pipeline but faced challenges regarding explanation robustness across clients and overall predictive accuracy in the presence of data heterogeneity.

---

#### 4.3 Comparative Analysis, Reflection, and Future Work

Both approaches showed relevant strengths: the decision tree model ensured direct interpretability and structural consistency, while the linear regression approach integrated post-hoc explainability techniques (SHAP and LIME) and demonstrated stable convergence. Table 1 summarizes their key differences and shared characteristics. More detailed configurations are described in Sections 4.1 and 4.2.

**Table 1: Summary of Key Differences Between Both Strategies**

| Feature                              | Decision Tree-Based Approach                     | Linear Regression with Explainability         |
|--------------------------------------|--------------------------------------------------|-----------------------------------------------|
| **Model**                             | DecisionTreeRegressor (max_depth: 10, random_state: 42) | PyTorch nn.Linear with Adam Optimizer         |
| **Clients**                           | 2 (same dataset, independent preprocessing)      | 2 (partitioned dataset, containerized setup)  |
| **Data Partitioning**                | Random split with seed 42 (80% train, 20% test, 10% validation) | Random split with seed 42 (80% train, 20% test, 10% validation) |
| **Batch Size**                       | 32                                               | 32                                             |
| **Rounds**                            | 5                                                | 5                                             |
| **Training Strategy**                 | Feature importance shared; trees rebuilt each round | FedAvg aggregation of model weights        |
| **Explainability**                    | Tree structure + feature importance              | SHAP (local, fixed seed), LIME (local, fixed instance) |
| **Monitoring**                        | Tree structure logs, JSON metrics, similarity index | Loss, RMSE, visualizations, reports         |
| **RMSE (Client 1)**                   | 66,234.26                                        | 235,231.31                                    |
| **RMSE (Client 2)**                   | 61,409.84                                        | 235,231.31                                    |
| **Structure Convergence**            | Structural similarity: 0.95–0.97                 | RMSE stabilized from round 3 onward           |
| **Privacy**                           | Data stays local, only feature importance shared | Data stays local, only model weights shared   |
| **Training Time per Round**          | ~1.16 seconds                                    | ~6.73 seconds                                 |
| **Loss Function**                    | Not applicable (feature-based split)             | Mean Squared Error (MSE)                      |

Future work should include testing hybrid models combining decision trees and linear regression, incorporating additional aggregation strategies, expanding to more clients and data sources, introducing differential privacy and encryption techniques, and quantifying the convergence of explanations — for example, assessing the stability of SHAP and LIME across rounds.

Both strategies confirmed the feasibility of interpretable federated learning and the value of explainability for trustworthy AI systems. However, reducing data heterogeneity impact and reinforcing robustness of interpretability remain open challenges.
