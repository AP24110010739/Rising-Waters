"""
Locust Performance Test — Rising Water Flood Prediction App
============================================================
Tests all major routes: login, home, predict, history, analysis.
Run with:  locust -f locustfile.py --host http://127.0.0.1:5001
"""

from locust import HttpUser, task, between, SequentialTaskSet


class FloodAppUser(SequentialTaskSet):
    """Simulates a real user flow: register → login → browse → predict → history."""

    def on_start(self):
        """Register and login before running tasks."""
        import time
        # Create a unique user for this simulated session
        self.username = f"locust_user_{id(self)}_{int(time.time())}"
        self.email = f"{self.username}@test.com"
        self.password = "Test@1234"

        # Step 1: Register
        self.client.post("/register", data={
            "username": self.username,
            "email": self.email,
            "password": self.password,
            "confirm_password": self.password
        }, name="/register")

        # Step 2: Login
        self.client.post("/login", data={
            "login_id": self.username,
            "password": self.password
        }, name="/login")

    @task
    def load_home_page(self):
        """Test 1: Load the home/landing page."""
        self.client.get("/", name="/  (Home Page)")

    @task
    def load_predict_form(self):
        """Test 2: Load the prediction input form."""
        self.client.get("/predict", name="/predict (GET Form)")

    @task
    def submit_valid_prediction(self):
        """Test 3: Submit valid weather values for flood prediction."""
        self.client.post("/predict", data={
            "Temp": "29.5",
            "Humidity": "78.2",
            "Cloud Cover": "42",
            "ANNUAL": "3248.6",
            "Jan-Feb": "73.4",
            "Mar-May": "386.2",
            "Jun-Sep": "2122.8",
            "Oct-Dec": "666.1"
        }, name="/predict (POST Valid)")

    @task
    def submit_extreme_prediction(self):
        """Test 4: Submit extreme weather values."""
        self.client.post("/predict", data={
            "Temp": "45.0",
            "Humidity": "99.0",
            "Cloud Cover": "100",
            "ANNUAL": "9999.9",
            "Jan-Feb": "500.0",
            "Mar-May": "800.0",
            "Jun-Sep": "5000.0",
            "Oct-Dec": "2000.0"
        }, name="/predict (POST Extreme)")

    @task
    def load_analysis_page(self):
        """Test 5: Load the EDA analysis dashboard."""
        self.client.get("/analysis", name="/analysis (Charts)")

    @task
    def load_history_page(self):
        """Test 6: Load the prediction history log."""
        self.client.get("/history", name="/history (Logs)")


class WebsiteUser(HttpUser):
    """Locust user definition with realistic wait times."""
    tasks = [FloodAppUser]
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
