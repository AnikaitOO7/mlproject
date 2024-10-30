# 📊 Student Performance Indicator - Machine Learning Project

This project analyzes how students' academic performance is influenced by various factors such as **gender**, **ethnicity**, **parental education**, **lunch type**, and **test preparation course**. The aim is to predict **math scores** using **machine learning models**, while implementing **CI/CD pipelines** and **MLOps principles** for seamless workflow.

---

## 🏗️ Project Structure

```plaintext
├── .github/               # GitHub workflows for CI/CD
│   └── workflows/
│       └── ci.yml         # CI/CD pipeline configuration
├── notebook/              # Jupyter Notebooks for EDA and Model Training
│   ├── data/              # Dataset files
│   ├── EDA_Student_Performance.ipynb
│   └── Model_Training.ipynb
├── src/                   # Source code folder
│   ├── components/        # Core components (data ingestion, transformation, etc.)
│   ├── pipeline/          # Train and predict pipelines
│   ├── exception.py       # Custom exception handling
│   ├── logger.py          # Logging setup
│   └── utils.py           # Utility functions
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── setup.py               # Setup file for packaging
```
