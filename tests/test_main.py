import json
import pytest
from fastapi.testclient import TestClient
from src import main

client = TestClient(main.app)


@pytest.fixture
def employee_file(tmp_path, monkeypatch):
    file_path = tmp_path / "employees.json"

    employees = [
        {"name": "Arun", "salary": 30000},
        {"name": "Priya", "salary": 45000},
        {"name": "Rahul", "salary": 60000},
    ]

    file_path.write_text(json.dumps(employees))
    monkeypatch.setattr(main, "EMPLOYEE_FILE", file_path)

    return file_path


# Test 1: Health endpoint
def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# Test 2: Get employee list
def test_get_employees(employee_file):
    response = client.get("/employees")

    assert response.status_code == 200
    assert len(response.json()) == 3
    assert response.json()[0]["name"] == "Arun"


# Test 3: Create employee successfully
def test_create_employee(employee_file):
    response = client.post("/employees", json={"name": "Karthik", "salary": 50000})

    assert response.status_code == 200
    assert response.json()["employee"]["name"] == "Karthik"


# Test 4: Employee not found
def test_employee_not_found(employee_file):
    response = client.get("/employees/Sakthi")

    assert response.status_code == 404
    assert response.json() == {"detail": "Employee not found"}


# Test 5: Invalid salary
def test_invalid_salary(employee_file):
    response = client.post("/employees", json={"name": "Ravi", "salary": 0})

    assert response.status_code == 422


# Test 6: Empty employee name
def test_empty_name(employee_file):
    response = client.post("/employees", json={"name": "", "salary": 50000})

    assert response.status_code == 422
