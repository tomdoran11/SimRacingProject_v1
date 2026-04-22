# SimRacingProject_v1
Final year project using machine learning to evaluate sim racing performance using python and scikit-learn.

# iRacing Telemetry Machine Learning System
- Developed in Python 3.12

# Features 
- Telemetry data preprocessing and feature extraction
- Sector and lap predictions
- Linear Regression model for baseline
- A Random Forest model for each sector
- Performance evaluation using MAE and R-squared metrics

# Requirements
- Python 3.12 (tested)
- pandas
- numpy
- scikit-learn

# How to run the code
- Run the main 'model_v3', ensuring model is reading correct, preprocessed data

# Additional notes
- babymodel_v1.py and load_data was used for the first created system, based on fake data for practice
- model_v3 is the final version and the core (as of 22/04/2026) and was tested with silverstone_20laps_p911gt3_v1.csv
- Data is stored in '/data'
- Models and other code is included in '/src'
- 



