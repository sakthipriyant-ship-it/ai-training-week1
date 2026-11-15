import csv
import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import openpyxl
import pymupdf


logging.basicConfig(
    level=logging.ERROR,
    format="%(levelname)s - %(message)s"
)

folder = Path(__file__).parent

def read_employees(path: Path) -> List[dict]:
    with open(path, "r") as file:
        employees = list(csv.DictReader(file))

    for emp in employees:
        emp["salary"] = int(emp["salary"])

    return employees

def find_employee(name: str) -> Optional[dict]:
    for emp in employees:
        if emp["name"].lower() == name.lower():
            return emp
    return None

def write_json(employees: List[dict]) -> None:
    with open(folder / "employees.json", "w") as file:
        json.dump(employees, file, indent=4)

def write_high_earners(employees: List[dict]) -> int:
    high = [emp for emp in employees if emp["salary"] > 50000]

    with open(folder / "high_earners.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["name", "department", "salary"]
        )
        writer.writeheader()
        writer.writerows(high)

    return len(high)

def create_excel(employees: List[dict]) -> int:
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    sheet.append(["Name", "Department", "Salary"])

    for emp in employees:
        sheet.append([
            emp["name"],
            emp["department"],
            emp["salary"]
        ])

    workbook.save(folder / "salary.xlsx")

    workbook = openpyxl.load_workbook(folder / "salary.xlsx")
    return workbook.active.max_row - 1

def read_pdf(path: Path) -> str:
    pdf = pymupdf.open(path)
    text = pdf[0].get_text()
    pdf.close()
    return text[:100]

def read_file(path: Union[int, str]) -> Optional[dict]:
    try:
        with open(path, "r") as file:
            return json.load(file)

    except FileNotFoundError:
        logging.error(f"File not found: {Path(path).name}")

    except json.JSONDecodeError:
        logging.error(f"Invalid JSON file: {Path(path).name}")

    return None

employees = read_employees(folder / "employees.csv")

write_json(employees)

count = write_high_earners(employees)

print(f"Rows read: {len(employees)} | High earners written: {count}")

rows = create_excel(employees)

print(f"salary.xlsx created and read back: {rows} rows")

pdf_path = folder / "Week1_Assessment.pdf"

if pdf_path.exists():
    pdf_text = read_pdf(pdf_path)

read_file(folder / "missing.txt")