#!/bin/bash

apt install libgl1-mesa-glx
gunicorn -c gunicorn_config.py app:app
