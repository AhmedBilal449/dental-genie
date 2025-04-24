**DentalGenie**

A web application that uses a custom built Yolov8 machine learning model to detect diseases in dental x-rays. Provides the user with AI assistance as well using Gemini Flash-2.0. Allows detection of impacted teeth, caries, deep caries and pericapal lesions

Built in With

    Python
    Shell


Requirements (Prerequisites)

To run this project, you need to have the following installed:

    Python 3.12 or higher (https://www.python.org/downloads/)

Installation

To install the project, follow these steps:

    Clone the repository to your local machine from https://github.com/AhmedBilal449/dental-genie

    Create a virtual environment and activate it python -m venv venv

    Activate the virtual environment source venv/bin/activate (Linux/Mac), venv\Scripts\activate (Windows)

    Install the project dependencies pip install -r requirements.txt (this may take a few minutes)


Run the project

    Go to runs and check which models are available.
    Give the relative path for each model in inference.
    Run app.py
    Ctrl click the Address the application provides you.
    Upload any x-rays you have to the upload box or get some from the xray folderand click analyse
    

File Structure

    |   .gitignore
    |   README.md
    |   requirements.txt
    |   app.py
    |   cocotoyoloconvertor.py
    |   heirconversion.py
    |   inference.py
    |   llm_integration.py
    |   train.sh
    |   yolov8n.pt
    |
    +---project
    |   |   val_0.png
    |   |   val_1.png
    |   |   val_2.png
    |   |   val_3.png
    |   |   val_4.png
    |   |   val_5.png
    |   |   val_6.png
    |   |   val_7.png
    |   |   val_8.png
    |   |   val_9.png
    |
    +---runs
    |
    +---detect


