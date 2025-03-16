# Student Performance Prediction

This project aims to predict student performance using machine learning techniques, leveraging a dataset from the UCI Machine Learning Repository.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Understanding the factors that influence student performance can help educators and institutions implement strategies to improve educational outcomes. This project explores various attributes affecting student performance and applies machine learning models to predict academic success.

## Dataset

The dataset used in this project is sourced from the UCI Machine Learning Repository and includes:

- **Demographic Information:** Age, gender, etc.
- **Social Features:** Parental education level, family background, etc.
- **Academic Features:** Study time, previous grades, etc.

For more details, refer to the [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance).

## Installation

To set up the project locally:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/muhammedfayiz122/student_performance_prediction.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd student_performance_prediction
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installation, you can run the main script to preprocess the data, train the models, and evaluate their performance:

```bash
python main.py
```

Ensure that the dataset is placed in the appropriate directory as specified in the code.

## Models Used

The project explores several machine learning models, including:

- **Linear Regression**
- **Decision Trees**
- **Support Vector Machines**
- **Random Forests**

Each model's performance is evaluated to determine the most effective approach for this dataset.

## Evaluation Metrics

The models are assessed using the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

These metrics provide a comprehensive understanding of each model's predictive capabilities.

## Results

The results section summarizes the performance of each model, highlighting the most accurate and reliable predictors of student performance.

## Contributing

Contributions to this project are welcome. Feel free to fork the repository, make improvements, and submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).

