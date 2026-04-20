import csv
import os
import random

random.seed(42)

out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "student_data.csv")
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

with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["attendance", "internal", "assignment", "previous", "result"])
    w.writerows(rows)
print("Wrote", out_path, "with", len(rows), "rows")
