# Testbed for Spark and `lightgbm`

## Synthethic Data Generation

**Notebook**: [create_time_series_training_data.ipynb](./jupyter_notebooks/create_time_series_training_data.ipynb)

The notebook generates synthetic time series data with specific characteristics and prepares it for time series modeling. Here's a summary of its functionality:

1. **Synthetic Data Generation**:
   - Creates a time series dataset with `N_SAMPLES` samples and `N_FEATURES` features.
   - Adds time-dependent components to the features, including trends, weekly and monthly seasonality, and autocorrelation.
   - Uses `make_classification` to generate a binary classification target with class imbalance and label noise.

2. **Time Series Splitting**:
   - Implements a 4-fold time series split using `TimeSeriesSplit` to ensure temporal order is preserved in training and testing sets.

3. **Visualization**:
   - Plots sample features to illustrate trends, seasonality, and autocorrelation.
   - Visualizes the train-test splits for each fold.
   - Displays the positive class ratio for training and testing sets in each fold.

4. **Data Export**:
   - Saves the entire dataset to a Parquet file.
   - Exports training and testing data for each fold to separate Parquet files.

5. **Output**:
   - Prints summary statistics for each fold, including the number of samples and the positive class ratio.
   - Saves a visualization of the data and splits as an image file.

The notebook is designed for creating and preparing time series data for machine learning experiments, particularly for tasks involving temporal dependencies.

**Overview of synthetic data generation**
![Overview of synthetic data](images/synthetic_timeseries_visualization.png)

**Generaed synthetic data sets**
![Generated synthetic data sets](images/synthetic_time_series_data.png)

## `lightgbm` and `pyspark` Integration
**Notebook**: [spark_lightgbm_classification.ipynb](./jupyter_notebooks/spark_lightgbm_integration.ipynb)

The notebook `spark_lightgbm_classification.ipynb` demonstrates a distributed approach to training and evaluating LightGBM classification models using Apache Spark. It is designed to handle large-scale datasets efficiently by leveraging Spark's parallel processing capabilities. The notebook performs the following tasks:

1. **Library Imports**: It imports essential libraries for data manipulation (`pandas`), distributed computing (`pyspark`), machine learning (`lightgbm`), and evaluation (`sklearn`).

2. **Directory Setup**: Paths for data and results are defined. The results directory is cleaned up to ensure no residual files interfere with the current run.

3. **Spark Session Initialization**: A Spark session is created with specific configurations for memory management and fault handling. This session enables distributed data processing and parallel execution of tasks.

4. **Data Preparation**: The notebook retrieves unique fold IDs and prepares training and testing datasets for each fold by reading parquet files. These datasets are split into features and target labels.

5. **Model Training**:
   - A function `train_lightgbm_model` is defined to encapsulate the training process for a LightGBM classification model.
   - The function reads training and testing data for a specific fold, initializes a `LGBMClassifier` with provided hyperparameters, and trains the model using the `fit` method.
   - During training, evaluation metrics are logged, and early stopping is applied to prevent overfitting.
   - After training, the model generates predictions and calculates various classification metrics, including accuracy, precision, recall, F1 score, and AUC-ROC.
   - A detailed classification report is printed for each fold, and the results (including metrics and hyperparameters) are saved as parquet files in the results directory.

6. **Hyperparameter Tuning**: A hyperparameter grid is defined, and all possible combinations of parameters are generated. This allows for systematic exploration of the model's performance under different configurations.

7. **Parallel Processing**: Spark's RDDs are used to parallelize the training process across multiple parameter combinations. Each combination is processed independently, leveraging Spark's distributed computing capabilities.

8. **Result Aggregation**: The results from all parameter combinations are read from the parquet files, concatenated into a single DataFrame, and summarized. Key metrics are described to provide insights into the model's performance.

9. **Spark Session Termination**: The Spark session is stopped after all tasks are completed to release resources.

### Run-time Screenshots

**Spark Stages**:
![Spark Stages](images/spark_app_stages_overview.png)

**Spark Event Timeline**:
![Spark Event Timeline](images/spark_app_events.png)

**Spark Executor Metrics**:
![Spark Executor Metrics](images/spark_app_executor_metrics.png)

**Spark Executor Overview**:
![Spark Executor Overview](images/spark_executor_overview.png)

**Standalone Spark Cluster Overview**:
![Spark Cluster Overview](images/standalone_spark_cluster_utilization_overview.png)

**Standalone Spark Cluster Executor Utiliazaton**:
![Spark Cluster Executor Utilization](images/spark_worker_utilization.png)

**Successful Spark Job**:
![Successful Spark Job](images/spark_app_successful_completion.png)

**Hyperparameter Results Files**:
![Hyperparameter Results Files](images/hyperparameter_result_files.png)

**Summary of Hyperparameter Tuning Metrics**:
![Summary of Hyperparameter Tuning Metrics](images/hyperparameter_test_metrics_summary.png)



