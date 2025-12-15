# Multicriteria Load Control – Forecasting, Optimization and Real-Time Control

## Overview
This repository contains the complete data-processing, forecasting, optimization,
and real-time control pipeline developed for the **Multicriteria Load Control**
project in the EI2525 Electric Power Engineering course at KTH Royal Institute of Technology.

The project was examined by **Prof. Lina Bertling Tjernberg** and supervised by
**Prof. Nathaniel Taylor**.

The objective of the project is the development of an intelligent load-control
system for a domestic electric water heater, aiming to:
- minimize electricity cost,
- increase self-consumption of local PV generation,
- and respect comfort and technical constraints.

The system is designed to run on a **Raspberry Pi 5** with an
**Arduino-based TRIAC controller**.

---

## Repository Structure

### Part A – Load Forecasting
- `partA.ipynb`
- `Active_power_import_prediction.ipynb`
- `LoadForecast.py`

These scripts generate day-ahead and short-term load forecasts using
multi-output machine learning models.
Pre-trained models (`days_prediction.joblib`, `hours_prediction.joblib`) are provided and can be used. However, `hours_prediction.joblib` model is 2GB and it is not possible to upload it via GitHub

The training notebook (`Active_power_import_prediction.ipynb`) illustrates
the full training process but relies on a **large dataset that is not included**
in the repository.

---

### Part B – Data Preparation and Profiling
- `partB.ipynb`
- `OtherForecasts.py`

This part processes water usage profiles and external API data
(weather, solar, electricity prices) in order to construct the full set of
inputs required by the optimization problem.

The file `water_data.csv` is required for this step but is not included
due to its large size. A compressed version is provided for reference.

---

### Part C – Day-Ahead Optimization
- `PartOptimization-1.ipynb`
- `Optimization.py`

This part formulates and solves the day-ahead optimization problem,
producing the optimal heating schedule for the next day
(e.g. `triac_schedule.csv`).

---

### Real-Time Control
- `RealTimeController.ipynb`
- `RealTimeController.py`

The real-time controller monitors PV export and household demand
and performs online adjustments of the heating schedule by reallocating
future consumption when excess solar generation is available.
No re-optimization is performed in real time; instead, the controller
acts as a supervisory layer on top of the day-ahead schedule.

---

### Hardware Control (Arduino)
- `Arduino/Arduino.ino`
- `Arduino/test_TRIAC.ino`

The Arduino code handles zero-cross detection and TRIAC firing.
It receives duty-cycle commands from the Raspberry Pi and performs
low-level power modulation of the heating element.
No forecasting or optimization logic is implemented on the Arduino side.

---

## Execution Timing
For correct day-ahead operation, the notebooks must be executed
**approximately 5 minutes before the day changes**.
This ensures that the optimization schedule for the next day is available
before midnight.

---

## Credits
This project is the personal work of:

- **Iraklis Bournazos**
- **Hadrien Guillaud**
- **Dio Damara**

All code was developed within the scope of the
EI2525 Electric Power Engineering Project at KTH Royal Institute of Technology.
