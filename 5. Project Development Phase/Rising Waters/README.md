**Rising Waters - Flood Prediction System**

Project Overview:

Rising Waters is a Machine Learning based web application that predicts the likelihood of floods using historical rainfall and weather data. The system leverages a trained Machine Learning model integrated with a Flask web application to provide users with instant flood risk predictions through a simple and interactive interface.
The project aims to support early disaster preparedness by helping individuals and authorities identify potential flood conditions before they become critical.


**Features:**
- Machine Learning based flood prediction
- Instant flood risk analysis
- Simple and responsive web interface
- User friendly input form
- Fast prediction results
- Attractive UI with modern design
- Flask based backend integration
- Real time prediction using trained model


**Technologies Used**
- Python
- Flask
- HTML5
- CSS3
- JavaScript
- Scikit-learn
- NumPy
- Pandas

**Input Parameters**

The prediction model uses the following environmental parameters:
- Temperature
- Humidity
- Cloud Cover
- Annual Rainfall
- January - February Rainfall
- March - May Rainfall
- June - September Rainfall
- October - December Rainfall
- Average June Rainfall
- Subdivision Rainfall

  
**Machine Learning Model**
- Algorithm: Random Forest Classifier
- Data Preprocessing: StandardScaler
- Model Storage: Pickle (.save)
- Framework: Scikit-learn

The model is trained using historical weather and rainfall datasets to classify whether flood conditions are likely to occur based on user-provided inputs.

**Project Structure**

Rising-Waters/
│

├── 1. Brainstorming & Ideation/

├── 2. Requirement Analysis/

├── 3. Project Design Phase/

├── 4. Project Planning Phase/

├── 5. Project Development Phase/

│   ├── Dataset/

│   ├── Model/

│   ├── Notebook/

│   ├── static/

│   │   ├── css/

│   │   ├── images/

│   │   └── js/

│   ├── templates/

│   ├── app.py

│   ├── requirements.txt

│   └── trained_model.save
│
├── 6. Project Testing/

├── 7. Project Documentation/

└── 8. Project Demonstration/


**How to Run**

Clone the Repository

git clone https://github.com/AP24110010739/Rising-Waters.git

Install Dependencies

pip install -r requirements.txt

Run the Application

python app.py

Open in Browser

http://127.0.0.1:5000


**Output**
After entering the required weather and rainfall parameters, the system predicts one of the following outcomes:
✅ No Flood Chance
⚠️ Flood Chance Detected

The prediction is generated instantly using the trained Machine Learning model.

**Objectives**
- Predict flood occurrence using Machine Learning.
- Provide early warning based on environmental conditions.
- Improve disaster preparedness.
- Reduce potential loss of life and property.
- Deliver an easy-to-use web-based prediction system.

  
**Future Enhancements**
- Integration with live weather APIs
- GIS-based flood visualization
- SMS and Email alert notifications
- Mobile application support
- Multi-region flood prediction
- IoT sensor integration for real-time monitoring
- Cloud deployment with continuous model updates
  
Developed By
KHUSHWANTH MEDARAMETLA
B.Tech - Computer Science and Engineering
SRM University AP
Internship Project
