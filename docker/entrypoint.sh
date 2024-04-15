#!/bin/bash
uvicorn serving.app.main:app --host 0.0.0.0 --port 9090