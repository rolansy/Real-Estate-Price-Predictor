# Real Estate Price Prediction

A machine learning project to predict real estate prices based on various features.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Repository Contents](#repository-contents)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/real-estate-price-prediction.git
    ```
2. Navigate to the project directory:
    ```sh
    cd real-estate-price-prediction
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Jupyter Notebook:
    ```sh
    jupyter notebook Real\ Estate\ Price\ Prediction.ipynb
    ```
2. Follow the instructions in the notebook to train the model and make predictions.

## Features

- Data preprocessing
- Model training
- Model evaluation
- Price prediction

## Repository Contents

| File/Folder                  | Description                                      |
|------------------------------|--------------------------------------------------|
| `Real Estate Price Prediction.ipynb` | Jupyter Notebook for the project.               |
| `Real Estate Price Prediction.py`   | Python script for the project.                  |
| `columns.json`               | JSON file containing column names.               |
| `requirements.txt`           | List of dependencies required for the project.   |
| `README.md`                  | This README file.                                |

## Code Excerpt

```python
import json
columns={
    'data_columns':[col.lower() for col in x.columns]
}
with open ('columns.json','w') as f:
    f.write(json.dumps(columns))
