import numpy as np
marks = np.array([[70,80,90],
                  [60,75,85],
                  [88,92,79],
                  [55,65,70]])
print("Shape: ",marks.shape)

print("Student averages:",np.round(marks.mean(axis=1),2))
print("Subject averages:",marks.mean(axis=0))

weights = np.array([0.3, 0.3, 0.4])
print("Weighted scores:",np.dot(marks,weights))