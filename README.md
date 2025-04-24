DentalGenie

A web application that uses a custom built Yolov8 machine learning model to detect diseases in dental x-rays. Provides the user with AI assistance as well using Gemini Flash-2.0. Allows detection of impacted teeth, caries, deep caries and pericapal lesions

    Django
    Chart.js
    Tailwind CSS

Requirements (Prerequisites)

To run this project, you need to have the following installed:

    Python 3.6 or higher (https://www.python.org/downloads/)

Installation

To install the project, follow these steps:

    Clone the repository to your local machine git clone https://github.com/olumide-moore/AirQualityAnalysis.git

    cd AirQualityAnalysis

    Create a virtual environment and activate it python -m venv venv

    Activate the virtual environment source venv/bin/activate (Linux/Mac)

    venv\Scripts\activate (Windows)

    Install the project dependencies pip install -r requirements.txt (this may take a few minutes)

Setup Database

    Apply the migrations python manage.py migrate

    python manage.py makemigrations

    Create a superuser python manage.py createsuperuser (follow the prompts to create a superuser)

Run the project

    Run the project python manage.py runserver

Open your browser and navigate to http://127.0.0.1:8000/ Login with the superuser credentials created in step 6.
Usage

To access populated data, options include:

    Plume sensor type with IDs 47, 24, 11
    Zephyr sensor type with IDs 60, 62
    Dates between 2024-01-01 and 2024-01-20

Running the tests

To run the tests, run the following command:

 python manage.py test

This will run all the tests in the directories and display the results in the terminal. Optionally, you can run a specific test file by running the following command:

 python manage.py test project.tests.<test_file_name>

File Structure

The project is structured as follows:

|   manage.py
|   README.md
|   requirements.txt
|   
+---authentication   
|   |   tests.py
|   |   urls.py
|   |   views.py
|   |   
|   +---migrations
|           
+---project
|   |   models.py
|   |   settings.py
|   |   urls.py
|   |   
|   +---documentations
|   |       aqi_calculator.html
|   |       data_processor.html
|   |       sensors_metadata.html
|   |       sensor_data_fetcher.html
|   |       
|   +---services
|   |       aqi_calculator.py
|   |       data_processor.py
|   |       sensors_metadata.py
|   |       sensor_data_fetcher.py
|   |       
|   +---static
|   |   +---css
|   |   |       style.css
|   |   |       
|   |   +---icons
|   |   |       
|   |   +---js
|   |           base.js
|   |           compare.js
|   |           home.js
|   |           inputs.js
|   |           
|   +---templates
|   |   |   aqiguide.html
|   |   |   base.html
|   |   |   compare.html
|   |   |   correlationguide.html
|   |   |   home.html
|   |   |   inputs.html
|   |   |   
|   |   +---authentication
|   |           login.html
|   |           signup.html
|   |           
|   +---tests_integration
|   |       
|   +---tests_unit
|   |       
|   +---views
|           compare.py
|           home.py
|           
+---routers
        db_routers.py
        

Directory and File Descriptions

    /authentication: Contains all the authentication logic, including user login and registration.
        urls.py: URL configurations for authentication views.
        views.py: Defines the views for the authentication app.
        /migrations: Database migration files for the authentication app.

    /project: Main project.
        models.py: Defines the database models for the project.
        settings.py: Configuration settings for the Django project.
        urls.py: URL configurations for the project.
        /documentations: Contains HTML files for the documentation pages.
        /services: Contains the business logic for the project.
        /static: Contains static files such as CSS, JavaScript, and image files.
        /templates: Contains HTML templates for the project.
        /tests_integration: Contains integration tests for the project.
        /tests_unit: Contains unit tests for the project.
        /views: Contains the views for the project.

    /routers: Contains the database router for the project.
