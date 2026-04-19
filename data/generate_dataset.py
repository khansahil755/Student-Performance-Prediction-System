"""
Columns: attendance, internal, assignment, previous, result
Rules: High marks -> Pass, Medium -> Average, Low -> Fail
"""
import csv
import random

random.seed(42)
rows = []

for _ in range(120):
    attendance = random.randint(50, 100)
    internal = random.randint(0, 30)
    assignment = random.randint(0, 20)
    previous = random.randint(0, 100)
    total_score = (attendance * 0.2) + (internal * 1.5) + (assignment * 1.5) + (previous * 0.5)
    if total_score >= 70:
        result = "Pass"
    elif total_score >= 45:
        result = "Average"
    else:
        result = "Fail"
    rows.append([attendance, internal, assignment, previous, result])

with open("student_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["attendance", "internal", "assignment", "previous", "result"])
    w.writerows(rows)
print("Generated student_data.csv with", len(rows), "rows")
