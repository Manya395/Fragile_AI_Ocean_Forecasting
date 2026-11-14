FRAGILE AI FOR OCEAN FORECASTING - STARTER PROJECT

1. Put fragile_ai_ocean_forecasting_dataset.csv in the data/ folder.
2. Open this folder in VS Code.
3. Create and activate a virtual environment:
   python -m venv venv
   venv\Scripts\activate
4. Install packages:
   pip install -r requirements.txt
5. Run:
   python main.py

The script predicts next-day SST using the previous 7 consecutive daily observations
at the same latitude/longitude. It uses a chronological split and only training-period
medians for missing-value imputation. It then runs Gaussian-noise, random-missingness,
and sensor-dropout experiments.

Check outputs/ and models/. Do not claim results until you have run and checked them.
The script is a baseline research implementation; verify dataset provenance and
scientific assumptions with your project guide before final submission.
