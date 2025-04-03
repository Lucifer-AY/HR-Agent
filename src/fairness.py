import pandas as pd
import xgboost as xgb
import shap
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(filename='data/logs/fairness.log', level=logging.INFO)

# Expanded synthetic data with protected attributes
data = pd.DataFrame({
    'skill_similarity': [0.8, 0.6, 0.9, 0.4, 0.7],
    'experience_match': [1.0, 0.8, 1.0, 0.5, 0.9],
    'education_match': [1, 0, 1, 0, 1],
    'gender': [0, 1, 0, 1, 0],  # 0: Male, 1: Female
    'age': [25, 35, 28, 40, 30],
    'label': [9, 7, 10, 5, 8]
})

# Train model
X = data.drop(['label', 'gender', 'age'], axis=1)
y = data['label']
model = xgb.XGBRegressor()
model.fit(X, y)

# Bias auditing with AI Fairness 360
dataset = BinaryLabelDataset(
    df=data.drop('label', axis=1).assign(label=(data['label'] > 7).astype(int)),
    label_names=['label'],
    protected_attribute_names=['gender', 'age']
)
privileged_groups = [{'gender': 0}, {'age': lambda x: x <= 35}]
unprivileged_groups = [{'gender': 1}, {'age': lambda x: x > 35}]
metric = BinaryLabelDatasetMetric(dataset, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
logging.info(f"Disparate Impact (Gender): {metric.disparate_impact():.2f}")
logging.info(f"Statistical Parity Difference (Gender): {metric.statistical_parity_difference():.2f}")

# Explainability with SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X)
shap.summary_plot(shap_values, X, show=False)
plt.savefig('data/logs/shap_summary.png')
logging.info("SHAP summary plot saved.")

# Example usage
if __name__ == "__main__":
    print(f"Disparate Impact: {metric.disparate_impact():.2f}")
    plt.show()