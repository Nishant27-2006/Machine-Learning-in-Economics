Machine Learning Models for Economic Forecasting: A Case Study on GDP Growth Prediction
Overview
This repository contains the research paper, code, and datasets used in the study "Machine Learning Models for Economic Forecasting: A Case Study on GDP Growth Prediction." The study explores the application of advanced machine learning models—Linear Regression, Random Forest, and Gradient Boosting—in predicting GDP growth. The research highlights the effectiveness of these models in handling complex, non-linear relationships within economic data, with a focus on improving the accuracy and reliability of economic forecasts.

Table of Contents
Project Structure
Installation
Usage
Results
Figures
Contributing
License
Project Structure
plaintext
.
├── data/
│   ├── GDP_Data_Cleaned.xlsx
│   ├── Inflation_Data_Cleaned.xlsx
│   └── SP500_Monthly_Data_Cleaned.xlsx
├── code/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── scenario_analysis.py
├── figures/
│   ├── actual_vs_predicted_gdp_growth.png
│   ├── feature_importance_random_forest.png
│   └── gdp_forecast_arima.png
├── paper/
│   ├── research_paper.pdf
│   └── references.bib
└── README.md
data/: Contains the cleaned datasets used in the study.
code/: Python scripts for data preprocessing, model training, evaluation, and scenario analysis.
figures/: Output figures generated from the analysis, used in the research paper.
paper/: The final research paper in PDF format along with the bibliography file.
Installation
To run the code and reproduce the results, you'll need to set up your Python environment. Follow these steps:

Clone the repository:


git clone https://github.com/yourusername/economic-forecasting-ml.git
cd economic-forecasting-ml
Create a virtual environment and activate it:


python3 -m venv venv
source venv/bin/activate
Install the required dependencies:


pip install -r requirements.txt
Usage
Once the environment is set up, you can run the analysis scripts:

Data Preprocessing:


python code/data_preprocessing.py
This script processes the raw data and generates cleaned datasets.

Model Training:


python code/model_training.py
This script trains the Linear Regression, Random Forest, and Gradient Boosting models on the cleaned data.

Evaluation:


python code/evaluation.py
This script evaluates the performance of the trained models using MAE, MSE, and R-squared metrics.

Scenario Analysis:


python code/scenario_analysis.py
This script performs scenario analysis using the Random Forest model to predict GDP growth under different economic conditions.

Results
The results of the analysis, including model performance metrics and scenario analysis, are detailed in the research paper located in the paper/ directory. Key findings include the superior accuracy of the Random Forest model in predicting GDP growth and its effectiveness in providing actionable insights for economic forecasting.

Figures
The following key figures generated from the analysis are available in the figures/ directory:

actual_vs_predicted_gdp_growth.png: Comparison of actual vs. predicted GDP growth using the Random Forest model.
feature_importance_random_forest.png: Feature importance analysis from the Random Forest model.
gdp_forecast_arima.png: GDP forecast using the ARIMA model.
These figures are included in the final research paper.

Contributing
Contributions to this project are welcome. If you have any suggestions or improvements, feel free to submit a pull request or open an issue.

License
This project is licensed under the MIT License. See the LICENSE file for details.

