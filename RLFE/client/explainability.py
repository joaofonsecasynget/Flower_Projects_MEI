import shap
import lime
import lime.lime_tabular
import numpy as np
import torch

class ModelExplainer:
    def __init__(self, model, feature_names, device="cpu"):
        self.model = model
        self.feature_names = feature_names
        self.device = device
        self.lime_explainer = None
        self.shap_explainer = None

    def predict_fn(self, x):
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            preds = self.model(x_tensor).cpu().numpy().reshape(-1)
        return preds

    def explain_lime(self, X_train, instance):
        if self.lime_explainer is None:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=self.feature_names,
                mode="regression"
            )
        exp = self.lime_explainer.explain_instance(instance, self.predict_fn, num_features=10) # Mostrar apenas as 10 features mais importantes
        return exp

    def explain_shap(self, X_train, n_samples=100):
        if self.shap_explainer is None:
            background = shap.sample(X_train, n_samples, random_state=42) if X_train.shape[0] > n_samples else X_train
            self.shap_explainer = shap.KernelExplainer(self.predict_fn, background)
        shap_values = self.shap_explainer.shap_values(X_train)
        return shap_values
