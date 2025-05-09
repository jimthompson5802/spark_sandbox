FROM quay.io/jupyter/all-spark-notebook

RUN conda install -c conda-forge lightgbm pyarrow


# Start JupyterLab without authentication
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]