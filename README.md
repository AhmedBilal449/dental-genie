DentalGenie

A web application that uses a custom-built YOLOv8 machine learning model to detect diseases in dental X-rays. It provides the user with AI assistance using Gemini Flash-2.0. It allows detection of impacted teeth, caries, deep caries, and periapical lesions.
Built in With

    Python

    Shell

Requirements (Prerequisites)

To run this project, you need to have the following installed:

    Python 3.12 or higher (Download Python)

Installation

To install the project, follow these steps:

    Clone the repository to your local machine:

git clone https://github.com/AhmedBilal449/dental-genie

Navigate to the project directory:

cd dental-genie

Create a virtual environment and activate it:

    For Linux/Mac:

python -m venv venv
source venv/bin/activate

For Windows:

    python -m venv venv
    venv\Scripts\activate

Install the project dependencies:

    pip install -r requirements.txt

Run the Project

    Go to the runs folder and check which models are available.

    Provide the relative path for each model in the inference.py.

    Run the project:

    python app.py

    Once the application runs, Ctrl-click the address the application provides.

    Upload any X-ray images you have to the upload box or use images from the xray folder, and click "Analyze."

File Structure

The project is structured as follows:
    
    | .gitignore
    | README.md
    | requirements.txt
    | app.py
    | cocotoyoloconvertor.py
    | heirconversion.py
    | inference.py
    | llm_integration.py
    | train.sh
    | yolov8n.pt
    |
    +---project
    | | val_0.png
    | | val_1.png
    | | val_2.png
    | | val_3.png
    | | val_4.png
    | | val_5.png
    | | val_6.png
    | | val_7.png
    | | val_8.png
    | | val_9.png
    |
    +---runs
    |
    +---detect
    Directory and File Descriptions

    .gitignore: Contains the list of files and directories to be ignored by Git.

    README.md: This file with project information.

    requirements.txt: Lists all the dependencies for the project.

    app.py: Main application file that runs the project.

    cocotoyoloconvertor.py: Converts YOLO annotations to a specific format.

    heirconversion.py: Handles conversion between image formats or model inputs.

    inference.py: Contains the inference logic for running the YOLOv8 model.

    llm_integration.py: Integrates the project with Gemini Flash-2.0 for AI assistance.

    train.sh: Script for training the YOLOv8 model.

    yolov8n.pt: Trained YOLOv8 model file.

    project: Contains sample validation images (val_0.png, etc.).

    runs: Contains output and logs from model runs.

    detect: Holds detection-related files or models.
