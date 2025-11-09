#!/bin/bash

KAFKA_HOME="/data/opt/kafka_2.13-3.5.2"
KAFKA_BROKER="localhost:9092"

# Check Kafka connection
timeout 1 bash -c "echo > /dev/tcp/localhost/9092" 2>/dev/null || {
    echo "Error: Kafka not running on ${KAFKA_BROKER}"
    exit 1
}

# Clean up topics (both data and control)
echo "Cleaning Kafka topics..."
${KAFKA_HOME}/bin/kafka-topics.sh --bootstrap-server ${KAFKA_BROKER} --delete --topic vector-queries 2>/dev/null
${KAFKA_HOME}/bin/kafka-topics.sh --bootstrap-server ${KAFKA_BROKER} --delete --topic svfusion-control 2>/dev/null

# Recreate topics to avoid first-message latency
echo "Creating Kafka topics..."
${KAFKA_HOME}/bin/kafka-topics.sh --bootstrap-server ${KAFKA_BROKER} --create --topic vector-queries --partitions 1 --replication-factor 1 --if-not-exists 2>/dev/null
${KAFKA_HOME}/bin/kafka-topics.sh --bootstrap-server ${KAFKA_BROKER} --create --topic svfusion-control --partitions 1 --replication-factor 1 --if-not-exists 2>/dev/null

# Run producer
cd build && ./streaming_producer "$@"
