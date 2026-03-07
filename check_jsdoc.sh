#!/bin/bash
find frontend/src -name "*.ts" -o -name "*.tsx" | xargs grep -L "/\*\*"
