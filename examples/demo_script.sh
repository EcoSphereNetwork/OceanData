#!/bin/bash
# Demonstration of the OceanData CLI

set -e

echo "User:"
oceandata whoami

MODEL=$(oceandata register DemoModel demo.json 1.0 | tail -n 1 | cut -d"'" -f2)

oceandata evaluate "$MODEL" dataset123
sleep 1

oceandata results "$MODEL"
