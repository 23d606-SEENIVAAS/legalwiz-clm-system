#!/bin/bash
# Azure App Service startup script for LegalWiz CLM Backend
cd routes
gunicorn main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120
