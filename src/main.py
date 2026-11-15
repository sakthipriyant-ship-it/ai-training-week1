from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import json
from pathlib import Path

app = FastAPI()

# Location of employees.json
EMPLOYEE_FILE = Path(__file__).resolve().parent.parent / "config" / "employees.json"


class Employee(BaseModel):
    name: str = Field(min_length=1)
    salary: float = Field(gt=0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Name cannot be empty")
        return value


# Read employee data
def read_employees():
    with open(EMPLOYEE_FILE, "r") as file:
        return json.load(file)


# Save employee data
def save_employees(employees):
    with open(EMPLOYEE_FILE, "w") as file:
        json.dump(employees, file, indent=4)


# GET /health
@app.get("/health")
def health():
    return {"status": "healthy"}


# GET /employees
@app.get("/employees")
def get_employees():
    return read_employees()


# GET /employees/{name}
@app.get("/employees/{name}")
def get_employee(name: str):
    employees = read_employees()

    for employee in employees:
        if employee["name"].lower() == name.lower():
            return employee

    raise HTTPException(status_code=404, detail="Employee not found")


# POST /employees
@app.post("/employees")
def create_employee(employee: Employee):
    employees = read_employees()

    new_employee = employee.model_dump()
    employees.append(new_employee)

    save_employees(employees)

    return {"message": "Employee added successfully", "employee": new_employee}
