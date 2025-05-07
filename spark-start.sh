#!/bin/bash

# Script to start Spark in different modes (master or worker)

# Default to master if not specified
NODE_TYPE=${SPARK_NODE_TYPE:-"master"}
SPARK_MASTER_HOST=${SPARK_MASTER_HOST:-"spark-master"}
SPARK_MASTER_PORT=${SPARK_MASTER_PORT:-"7077"}
SPARK_WORKER_CORES=${SPARK_WORKER_CORES:-"1"}
SPARK_WORKER_MEMORY=${SPARK_WORKER_MEMORY:-"1g"}

if [ "$NODE_TYPE" == "master" ]; then
    echo "Starting Spark master node..."
    exec $SPARK_HOME/sbin/start-master.sh --host $SPARK_MASTER_HOST --port $SPARK_MASTER_PORT --webui-port 8080 -h $SPARK_MASTER_HOST
elif [ "$NODE_TYPE" == "worker" ]; then
    echo "Starting Spark worker node..."
    # Wait for master to be available
    echo "Waiting for Spark master to be available..."
    until nc -z $SPARK_MASTER_HOST $SPARK_MASTER_PORT; do
        echo "Waiting for Spark master..."
        sleep 1
    done
    # Start worker
    exec $SPARK_HOME/sbin/start-worker.sh spark://$SPARK_MASTER_HOST:$SPARK_MASTER_PORT \
        --cores $SPARK_WORKER_CORES \
        --memory $SPARK_WORKER_MEMORY
else
    echo "Unknown node type: $NODE_TYPE"
    echo "Use 'master' or 'worker'"
    exit 1
fi

# Keep container running
tail -f /dev/null