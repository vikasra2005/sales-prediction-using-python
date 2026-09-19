# Sales Prediction Using Python

This project analyzes the `Advertising.csv` dataset and builds a sales prediction model based on advertising spend across TV, Radio, and Newspaper channels.

## Project Objective

- Explore the relationship between advertising spend and sales.
- Clean and prepare the dataset.
- Train and compare machine learning models.
- Save the best model for future sales prediction.
- Generate a business-friendly summary and visual insights.

## Dataset

The dataset is stored in `data/Advertising.csv` and contains the following columns:

- `TV`
- `Radio`
- `Newspaper`
- `Sales`

## Project Structure

- `data/Advertising.csv` – source dataset
- `src/data_loader.py` – dataset loading and cleaning
- `src/exploratory_analysis.py` – plots and visual analysis
- `src/model_pipeline.py` – model training and evaluation
- `src/predictor.py` – prediction interface for new ad spend inputs
- `main.py` – full workflow entry point
- `outputs/` – charts and reports
- `models/` – trained models and evaluation results

## How to Run

1. Open a terminal in the project folder.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the full project:
   ```bash
   python main.py
   ```

## Output

The workflow creates:

- sales distribution and correlation charts in `outputs/`
- a model comparison file in `models/model_comparison.csv`
- a trained model in `models/best_sales_model.joblib`
- a summary report in `outputs/sales_analysis_report.txt`

## Business Use

This project helps understand which advertising channel contributes most to sales and provides a quick estimate of future sales based on marketing spend.
