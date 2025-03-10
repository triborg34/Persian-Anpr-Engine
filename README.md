Persian ANPR Engine (YOLO)

This is the engine for the Persian Automatic Number Plate Recognition (ANPR) system, using YOLO for real-time detection of cars, license plates, and characters. It is capable of detecting any form of car.

Features

YOLO-based detection for:

Vehicles

License plates

Plate characters

Supports various car types and plate formats

Optimized for real-time performance

Easily integratable with a backend via Uvicorn & FastAPI

Can be used for CCTV systems with RTSP streams

Installation

Prerequisites

Python 3.8+

YOLO model & weights

Torch & OpenCV

FastAPI (for API integration)

Steps

Clone the repository:

git clone <engine_repo_url>
cd <engine_repo_name>

Install dependencies:

pip install -r requirements.txt

Download YOLO model weights (if not included in the repo):

wget <yolo_weights_url> -O models/yolo_weights.pt

Run the detection script:

python detect.py --source test_images/

Usage

The engine processes images and videos to detect vehicles, license plates, and characters.

It can be integrated with FastAPI and Uvicorn for real-time processing.

It supports CCTV systems via RTSP streams for continuous monitoring.

License

This project is open-source under the MIT License.

Persian ANPR Backend (Uvicorn & FastAPI)

This is the backend for the Persian Automatic Number Plate Recognition (ANPR) system. It serves as the bridge between the Flutter UI and the ANPR engine, using FastAPI and Uvicorn for high-performance API handling.

Features

FastAPI-based backend for handling ANPR requests

Uvicorn ASGI server for real-time processing

Handles image uploads for YOLO-based detection

Connects to PocketBase for data storage

REST API support for integration with Flutter UI

Can be used for CCTV systems with RTSP streams

Installation

Prerequisites

Python 3.8+

FastAPI & Uvicorn

PocketBase (for database)

YOLO Engine (for detection)

Steps

Clone the repository:

git clone <backend_repo_url>
cd <backend_repo_name>

Install dependencies:

pip install -r requirements.txt

Run the API server:

uvicorn main:app --host 0.0.0.0 --port 8000

License

This project is open-source under the MIT License.
