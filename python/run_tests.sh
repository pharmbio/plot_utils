#!/bin/bash

# Create test output directories for plots
mkdir -p test_output/clf test_output/reg

python -m pytest 
