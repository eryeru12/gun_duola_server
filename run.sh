#!/bin/bash

apt update -y
apt install python3-pip python3-venv -y
apt install libglib2.0-0 libmagic1 libgl1-mesa-glx -y
gunicorn -c gunicorn_config.py app:app
