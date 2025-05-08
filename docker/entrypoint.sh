#!/bin/bash
set -e

# Activate conda environment
source ${CONDA_HOME}/bin/activate spark-env

# Default values
SPARK_MASTER_HOST=${SPARK_MASTER_HOST:-spark-master}
SPARK_MASTER_PORT=${SPARK_MASTER_PORT:-7077}
SPARK_MASTER_WEBUI_PORT=${SPARK_MASTER_WEBUI_PORT:-8080}
SPARK_WORKER_WEBUI_PORT=${SPARK_WORKER_WEBUI_PORT:-8081}
SPARK_WORKER_CORES=${SPARK_WORKER_CORES:-1}
SPARK_WORKER_MEMORY=${SPARK_WORKER_MEMORY:-1g}

case "$1" in
  master)
    shift
    echo "Starting Spark master"
    exec ${SPARK_HOME}/bin/spark-class org.apache.spark.deploy.master.Master \
      --ip ${SPARK_MASTER_HOST} \
      --port ${SPARK_MASTER_PORT} \
      --webui-port ${SPARK_MASTER_WEBUI_PORT} \
      "$@"
    ;;
  worker)
    shift
    echo "Starting Spark worker"
    exec ${SPARK_HOME}/bin/spark-class org.apache.spark.deploy.worker.Worker \
      --webui-port ${SPARK_WORKER_WEBUI_PORT} \
      --cores ${SPARK_WORKER_CORES} \
      --memory ${SPARK_WORKER_MEMORY} \
      spark://${SPARK_MASTER_HOST}:${SPARK_MASTER_PORT} \
      "$@"
    ;;
  *)
    echo "Usage: $(basename "$0") {master|worker} [options]"
    exit 1
    ;;
esac