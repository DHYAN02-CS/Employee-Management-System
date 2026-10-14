
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, cast
import mysql.connector
import os
from dotenv import load_dotenv
from openai import OpenAIError

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
load_dotenv(BASE_DIR / ".env")

from rag import EmployeeRAG

app = FastAPI(title="Employee Management API")

# Allow requests from your index.html served by Live Server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"]
)


# MySQL connection
def get_connection():
    try:
        return mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "Dhyan1825@"),
            database="employee_db"
        )
    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=503,
            detail="Database connection failed. Check MySQL and credentials."
        ) from e


# Data received from index.html
class Employee(BaseModel):
    emp_name: Annotated[str, Field(min_length=1, max_length=100)]
    department: Annotated[str, Field(min_length=1, max_length=50)]
    salary: Annotated[
        Decimal,
        Field(ge=0, max_digits=10, decimal_places=2)
    ]


class RAGQuestion(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=1000)]
    top_k: Annotated[int, Field(ge=1, le=10)] = 5


class RAGAnswer(BaseModel):
    answer: str
    source_employee_ids: list[int]


rag = None


def get_rag() -> EmployeeRAG:
    global rag
    if rag is None:
        try:
            rag = EmployeeRAG()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
    return rag


def invalidate_rag_index() -> None:
    if rag is not None:
        rag.clear()


# CREATE: Receive employee details from HTML
@app.post("/employees", status_code=201)
def add_employee(employee: Employee):
    name = employee.emp_name.strip()
    department = employee.department.strip()

    if not name or not department:
        raise HTTPException(
            status_code=422,
            detail="Name and department cannot be blank."
        )

    db = get_connection()
    cursor = db.cursor()

    try:
        query = """
            INSERT INTO employees (employee_name, department, salary)
            VALUES (%s, %s, %s)
        """
        cursor.execute(
            query, (name, department, employee.salary)
        )

        emp_id = cursor.lastrowid
        db.commit()
        invalidate_rag_index()

        return {
            "message": "Employee added successfully",
            "employee": {
                "emp_id": emp_id,
                "emp_name": name,
                "department": department,
                "salary": employee.salary
            }
        }

    except mysql.connector.Error as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail="Unable to add employee."
        ) from e
    finally:
        cursor.close()
        db.close()


# READ: Get all employees for the HTML table
@app.get("/employees")
def get_employees():
    db = get_connection()
    cursor = db.cursor(dictionary=True)

    try:
        cursor.execute(
            "SELECT employee_id AS emp_id, employee_name AS emp_name, "
            "department, salary FROM employees ORDER BY employee_id"
        )
        return {"employees": cursor.fetchall()}

    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=500, detail="Unable to fetch employees."
        ) from e
    finally:
        cursor.close()
        db.close()


# READ: Get one employee
@app.get("/employees/{emp_id}")
def get_employee(emp_id: int):
    db = get_connection()
    cursor = db.cursor(dictionary=True)

    try:
        cursor.execute(
            "SELECT employee_id AS emp_id, employee_name AS emp_name, "
            "department, salary FROM employees WHERE employee_id = %s",
            (emp_id,)
        )
        employee = cursor.fetchone()

        if employee is None:
            raise HTTPException(
                status_code=404, detail="Employee not found."
            )

        return employee

    except HTTPException:
        raise
    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=500, detail="Unable to fetch employee."
        ) from e
    finally:
        cursor.close()
        db.close()


# UPDATE: Receive edited details from index.html
@app.put("/employees/{emp_id}")
def update_employee(emp_id: int, employee: Employee):
    name = employee.emp_name.strip()
    department = employee.department.strip()

    if not name or not department:
        raise HTTPException(
            status_code=422,
            detail="Name and department cannot be blank."
        )

    db = get_connection()
    cursor = db.cursor()

    try:
        cursor.execute(
            """
            UPDATE employees
            SET employee_name = %s, department = %s, salary = %s
            WHERE employee_id = %s
            """,
            (name, department, employee.salary, emp_id)
        )

        if cursor.rowcount == 0:
            cursor.execute(
                "SELECT employee_id FROM employees WHERE employee_id = %s",
                (emp_id,)
            )
            if cursor.fetchone() is None:
                raise HTTPException(
                    status_code=404, detail="Employee not found."
                )

        db.commit()
        invalidate_rag_index()

        return {
            "message": "Employee updated successfully",
            "emp_id": emp_id
        }

    except HTTPException:
        db.rollback()
        raise
    except mysql.connector.Error as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail="Unable to update employee."
        ) from e
    finally:
        cursor.close()
        db.close()


# DELETE: Delete employee from MySQL
@app.delete("/employees/{emp_id}")
def delete_employee(emp_id: int):
    db = get_connection()
    cursor = db.cursor()

    try:
        cursor.execute(
            "DELETE FROM employees WHERE employee_id = %s",
            (emp_id,)
        )

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404, detail="Employee not found."
            )

        db.commit()
        invalidate_rag_index()

        return {
            "message": "Employee deleted successfully",
            "emp_id": emp_id
        }

    except HTTPException:
        db.rollback()
        raise
    except mysql.connector.Error as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail="Unable to delete employee."
        ) from e
    finally:
        cursor.close()
        db.close()


@app.post("/ask", response_model=RAGAnswer)
def ask_employee_assistant(question: RAGQuestion):
    question_text = question.question.strip()
    if not question_text:
        raise HTTPException(
            status_code=422, detail="Question cannot be blank."
        )

    db = get_connection()
    cursor = db.cursor(dictionary=True)

    try:
        cursor.execute(
            "SELECT employee_id AS emp_id, employee_name AS emp_name, "
            "department, salary FROM employees ORDER BY employee_id"
        )
        employees = cast(list[dict[str, Any]], cursor.fetchall())
    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=500, detail="Unable to retrieve employee records."
        ) from e
    finally:
        cursor.close()
        db.close()

    try:
        answer, source_ids = get_rag().answer(
            question_text, employees, question.top_k
        )
    except (OpenAIError, RuntimeError) as e:
        raise HTTPException(
            status_code=502, detail="The RAG assistant is unavailable."
        ) from e

    return {"answer": answer, "source_employee_ids": source_ids}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(INDEX_FILE)


@app.get("/index", include_in_schema=False)
def index():
    return FileResponse(INDEX_FILE)


@app.get("/index.html", include_in_schema=False)
def index_html():
    return FileResponse(INDEX_FILE)
