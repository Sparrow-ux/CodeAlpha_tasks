# Health Prediction Dashboard

![Health Prediction UI](file:///C:/Users/Deepak%20raaja/.gemini/antigravity-ide/brain/af4349a8-3ff5-40ad-8535-a4760a9c9988/health_prediction_ui_1781703173535.png)

## 📖 Overview

**Health Prediction Dashboard** is a lightweight, interactive Streamlit application that enables users to obtain quick probability estimates for two common health conditions:

- **Heart Disease** (using an XGBoost classifier)
- **Diabetes** (using an SVM classifier)

The app loads pre‑trained models and a scaler from the repository, renders a clean three‑tab UI, and displays model‑specific metrics alongside predictions.

## ✨ Features

- **Three‑tab layout** – Heart Disease, Diabetes, and an informational *About Us* page.
- **Real‑time input validation** using Streamlit widgets (number inputs, sliders, select boxes).
- **Probability output** with human‑readable percentages.
- **Model performance summary** (accuracy, precision, recall, F1‑score) shown on the *About Us* tab.
- **Dark‑mode friendly design** with modern glass‑morphism styling (see screenshot above).

## 🛠️ Installation

```bash
# Clone the repo (adjust the path to your workspace)
git clone <repo‑url>
cd Codealpha_Health_Prediction

# Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The repository already contains the serialized model files (`heart_disease_model.pkl`, `diabetes_model.pkl`, `scaler_for_diabeties.pkl`). Ensure they remain in the project root.

## 🚀 Usage

```bash
streamlit run app.py
```

Open the displayed local URL (usually `http://localhost:8501`) in a browser. Navigate between tabs to:

- **Heart Disease** – input age, gender, chest‑pain type, etc., then click *Predict Heart Disease*.
- **Diabetes** – provide pregnancy count, glucose level, BMI, etc., then click *Predict Diabetes*.
- **About Us** – read model metrics and application credits.

## 🧠 Model Details

| Condition | Model Type | Accuracy | Precision | Recall | F1‑Score |
|-----------|------------|----------|-----------|--------|---------|
| Heart Disease | XGBoost Classifier | 84 % | 84.37 % | 84.37 % | 84.37 % |
| Diabetes | SVM Classifier | 71.48 % | 57.78 % | 74.54 % | 65.78 % |

The models were trained on publicly available datasets (UCI Heart Disease and Pima Indians Diabetes). They are intended **only** as a decision‑support tool; always consult a qualified healthcare professional for medical advice.

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Add new health condition predictors.
- Improve the UI/UX (e.g., theming, layout).
- Update model training pipelines with more recent data.

Please fork the repository, create a feature branch, and submit a pull request. Ensure that any new dependencies are reflected in `requirements.txt`.

## 📄 License

This project is released under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Streamlit team for the fantastic interactive framework.
- The authors of the original datasets (UCI Machine Learning Repository).
- **Deepak Raaja H J** – creator and maintainer of the dashboard.
