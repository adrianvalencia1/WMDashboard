# WMDashboard

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Dependencies](#dependencies)

## Introduction
This project is a web-based dashboard application with tools for financial planning built using the Dash framework by Plotly. It allows users to visualize and interact with data through a user-friendly interface. The dashboard currently features a Monte Carlo simulation and a portfolio optimization tool.

## Installation
To run this dashboard locally, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/adrianvalencia1/WMDashboard.git
    cd master
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. **Run the application:**
    ```sh
    python app.py
    ```

2. **Open your web browser and navigate to:**
    ```
    http://127.0.0.1:8050/
    ```

## Project Structure
```
WMDashboard/
│
├── app.py
├── data/
│   └── sample_data.csv
├── README.md
├── requirements.txt
├── utils/
│   └── util.py
│   └── cwutil.py
│   └── inflation.py
│   └── navbar.py
│   └── portfoliographs.py
└── pages/
    └── home.py
    └── montecarlo.py
    └── portfolio.py
```

- `app.py`: The main application file.
- `data/`: Contains data files.
- `utils/`: Contains utility functions for data processing.
- `pages/`: Contains layout and callback definitions for the Dash application.

## Dependencies
Dependencies are listed in the `requirements.txt` file. You can install them using the command:
```sh
pip install -r requirements.txt
```

---

Feel free to reach out if you have any questions or need further assistance.