import os
import sys
from pathlib import Path

project = Path(r'c:\Users\prajju\OneDrive\Desktop\code alpha project\Sales Prediction Using Python')
os.chdir(project)
sys.path.insert(0, str(project))

from src.data_loader import load_dataset
from src.exploratory_analysis import generate_eda_report
from src.model_pipeline import train_and_evaluate_model


df = load_dataset(project / 'data' / 'Advertising.csv')
print(df.head().to_string(index=False))
print('SHAPE:', df.shape)

generate_eda_report(df, project / 'outputs')
result = train_and_evaluate_model(df, project / 'models')
print(result['model_comparison'].to_string(index=False))
print('BEST_MODEL:', result['best_model'])
