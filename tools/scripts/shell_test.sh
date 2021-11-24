#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

PORT_ARG_NAME="--tcp_port "
PORT_ARG=$PORT_ARG_NAME$PORT
echo $PORT_ARG


TOTAL_PY_ARGS="$PORT_ARG $PY_ARGS"
echo $TOTAL_PY_ARGS
