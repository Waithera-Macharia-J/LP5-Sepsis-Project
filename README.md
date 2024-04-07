# Deploying Machine Learning Models using FastAPI: A CRISP-DM Framework Approach

[![GitHub stars](https://img.shields.io/github/stars/WachiraChris/LP-Six)](https://github.com/WachiraChris/LP-Six/stargazers)
[![GitHub license](https://img.shields.io/github/license/WachiraChris/LP-Six)](https://github.com/WachiraChris/LP-Six/blob/main/LICENSE)

In the rapidly evolving landscape of data science and machine learning, deploying models into real-world applications is a crucial step to turn insights into tangible value. **FastAPI**, a modern web framework for building APIs with Python, provides an efficient and user-friendly way to deploy machine learning models. In this article, we'll walk through the deployment process using FastAPI while adhering to the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework, a widely recognized methodology for data science projects.

## Understanding the CRISP-DM Framework

CRISP-DM consists of six major phases, providing a structured approach to guide data science projects:

- :chart_with_upwards_trend: **Business Understanding**
- :mag: **Data Understanding**
- :wrench: **Data Preparation**
- :computer: **Modeling**
- :bar_chart: **Evaluation**
- :rocket: **Deployment**

Each phase builds upon the insights gained from the previous ones, fostering a well-organized and iterative workflow. When deploying a machine learning model, we'll integrate these phases into the FastAPI development process.

### Phase 1: Business Understanding

We will be using the [sepsis dataset](https://www.kaggle.com/datasets/chaunguynnghunh/sepsis) from Kaggle. Our goal is to develop a machine learning model that predicts whether a patient in the ICU will develop sepsis. The aim is to achieve high accuracy and deploy the model using FastAPI.

### Phase 2: Data Understanding

The dataset includes columns such as:
1. ID: Unique number to represent patient ID
2. PRG: Plasma glucose
3. PL: Blood Work Result-1 (mu U/ml)
4. PR: Blood Pressure (mm Hg)
5. SK: Blood Work Result-2 (mm)
6. TS: Blood Work Result-3 (mu U/ml)
7. M11: Body mass index (weight in kg)/(height in m)^2
8. BD2: Blood Work Result-4 (mu U/ml)
9. Age: Patient's age (years)
10. Insurance: If a patient holds a valid insurance card
11. Sepssis: Target (Positive: patient in ICU will develop sepsis, Negative: otherwise)

Detailed descriptions of the columns can be found in the [data description](#link-to-your-data-description).

### Phase 3: Data Preparation

After conducting extensive exploratory data analysis, including univariate and multivariate analysis, the results can be found in the [analysis notebook](https://github.com/WachiraChris/LP-Six/blob/main/src/LP6.ipynb).

### Phase 4: Modeling

Seven different machine learning models were trained and evaluated to select the most suitable one. For more information, refer to the [evaluation notebook](https://github.com/WachiraChris/LP-Six/blob/main/src/LP6.ipynb).

### Phase 5: Evaluation

Seven models were trained and evaluated to select the best one. Details are available in this [notebook](https://github.com/WachiraChris/LP-Six/blob/main/src/LP6.ipynb).

### Phase 6: Deployment

Let's dive into the deployment process using FastAPI:

1. **Set Up Your Environment**
    Before you start, make sure you have Python 3.7 or later installed. Create a virtual environment to isolate project dependencies:
    ```bash
    python -m venv myenv
    ./myenv/Scripts/activate
    ```

2. **Install Dependencies**
    Install FastAPI and Uvicorn (ASGI server):
    ```bash
    pip install fastapi uvicorn
    ```
    Additional libraries might be required depending on your model and data processing needs.

3. **Build Your Machine Learning Model**
    Create and train your machine learning model using libraries like Scikit-learn, TensorFlow, or PyTorch. Save the trained model using serialization techniques like pickle.

4. **Create a FastAPI App**
    Define your FastAPI app in a Python file (e.g., `main.py`):
    ```python
    from fastapi import FastAPI

    app = FastAPI()
    ```

5. **Define API Endpoints**
    Define endpoints for interacting with your machine learning model:
    ```python
    @app.post("/predict/")
    async def predict(data: dict):
        # Use your machine learning model to make predictions
        # For example: result = MyModel.predict(data)
        return {"prediction": result}
    ```

6. **Run the App Locally**
    Run your FastAPI app locally for testing:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
    Access the Swagger documentation and interact with your API at http://127.0.0.1:8000/docs.

7. **Deploy to Production**
    To deploy in a production environment, use a production-ready ASGI server like Gunicorn or Hypercorn. Install the server and start the app:
    ```bash
    pip install gunicorn
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
    ```
    Configure a reverse proxy (e.g., Nginx) to handle incoming requests and forward them to the Gunicorn server.

## Conclusion

Deploying machine learning applications becomes manageable and efficient with tools like FastAPI. Its simplicity, automatic documentation generation, and performance make it an excellent choice for deploying machine learning models. By following the steps outlined in this guide, you can create, deploy, and serve your machine learning app to users, allowing them to interact with your models and benefit from data-driven insights.
