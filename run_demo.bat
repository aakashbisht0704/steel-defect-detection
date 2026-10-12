@echo off
echo ==============================================================
echo  Starting Industrial Surface Defect Inspection Web Dashboard
echo ==============================================================
call venv\Scripts\activate.bat
streamlit run app.py
pause
