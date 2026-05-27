# SalivaryFlowRateXerostomiaML

This repository contains the de-identified analysis dataset and Python code used for the study:

**Medication-Class Exposure and Chair-Side Salivary Functional Markers for Risk Stratification of Objective Salivary Hypofunction**

The project evaluates objective salivary hypofunction using demographic, systemic disease, medication-class, oral clinical, and chair-side salivary functional variables. The analyses include traditional regression models, incremental machine-learning prediction models, and SHAP-based model interpretation.

## Study overview

Objective salivary hypofunction is clinically relevant in aging and medically complex populations, but the relative contributions of systemic disease burden, xerogenic medication exposure, and chair-side salivary functional markers remain unclear.

This study analyzed:

- **Unstimulated salivary flow rate (UFR)**
- **Stimulated salivary flow rate (SFR)**
- **UFR-defined hyposalivation**, defined as UFR < 0.1 mL/min
- **SFR-defined hyposalivation**, defined as SFR < 0.7 mL/min

The main analytic framework included adjusted regression analyses, incremental gradient boosting models, and SHAP-based interpretation to assess the relative importance of systemic-medication variables and salivary functional markers.

## Repository contents

| File | Description |
|---|---|
| `Analysis_dataset_Saliva_Xerostomia_xerogenic_medication.xlsx` | De-identified analysis dataset used for the manuscript. |
| `Salivary Flow Rate and ML.py` | Main Python analysis script for salivary flow outcomes, hyposalivation prediction, and SHAP-based interpretation. |
| `AUC and ML.py` | Python script for AUROC-based model comparison, DeLong testing, and related machine-learning performance analyses. |

## Main variables

The dataset includes variables related to:

- Demographics: age and sex
- Salivary flow outcomes: UFR and SFR
- Flow-rate-defined hyposalivation outcomes
- Systemic disease burden
- Current medication class count
- High-confidence xerogenic medication class count
- Individual medication classes
- Oral and salivary clinical variables, including salivary pH and salivary buffer capacity

High-confidence xerogenic medication classes included:

- Antidepressive agents
- Hypnotics and sedatives
- Anti-allergic agents
- Urological agents
- Anticonvulsants
- Antihypertensive agents

## Machine-learning framework

Four prespecified incremental models were constructed:

| Model | Description |
|---|---|
| Model 1 | Demographic model: age + sex |
| Model 2 | Systemic-medication burden model: Model 1 + systemic disease category count + current medication class count + high-confidence xerogenic medication class count |
| Model 3 | Practical oral-systemic model: Model 2 + visual analog scale, sticky saliva, dental calculus, tongue coating, and halitosis |
| Model 4 | Full salivary oral-systemic model: Model 3 + salivary pH and salivary buffer capacity |

Continuous salivary flow outcomes were evaluated using regression models. Binary hyposalivation outcomes were evaluated using gradient boosting classification models. For classification tasks, repeated stratified 5-fold cross-validation with five repeats was used, and out-of-fold predicted probabilities were used for performance estimation and pairwise model comparisons.

## Key performance metrics

Regression performance was assessed using:

- Coefficient of determination (R²)
- Mean absolute error (MAE)
- Root mean squared error (RMSE)

Classification performance was assessed using:

- Area under the receiver operating characteristic curve (AUROC)
- Area under the precision-recall curve (AUPRC)
- Brier score
- DeLong test for pairwise AUROC comparison

## Software

The analyses were performed in Python. The manuscript analysis used:

- Python 3.12.13
- statsmodels 0.14.6
- SciPy 1.16.3
- scikit-learn 1.6.1
- XGBoost 3.2.0

## Data availability

The dataset provided in this repository is de-identified and intended to support transparency and reproducibility of the published analyses. The data should be used only for academic and research purposes.

## Code availability

The Python scripts in this repository reproduce the statistical analyses, machine-learning modeling, SHAP-based interpretation, and model-performance comparisons described in the manuscript.

## Citation

If you use this dataset or code, please cite the associated manuscript:

> Lee Y-H, Jeon S, Kim D-H, Kim T-S, Noh Y-K. Medication-Class Exposure and Chair-Side Salivary Functional Markers for Risk Stratification of Objective Salivary Hypofunction. Manuscript under review.

## Contact

For questions regarding the dataset, code, or manuscript, please contact:

**Yeon-Hee Lee, DDS, PhD**  
Email: omod0209@gmail.com

**Yung-Kyun Noh, PhD**  
Email: nohyung@hanyang.ac.kr
