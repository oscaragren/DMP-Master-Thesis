import numpy as np

v = np.array([0, 1, 0])
u = np.array([1, 0, 0])

#print(np.degrees(np.arccos(np.dot(v, u)/(np.linalg.norm(v)*np.linalg.norm(u)))))

v1 = np.array([0, 1, 0])
u1 = np.array([1, 0.5, 0])

print(180 - np.degrees(np.arccos(np.dot(v1, u1)/(np.linalg.norm(v1)*np.linalg.norm(u1)))))
#print(np.degrees(np.arcsin(np.dot(v1, u1)/(np.linalg.norm(v1)*np.linalg.norm(u1)))))

v2 = np.array([0, 1, 0])
u2 = np.array([0, -1, 0])

print(180 - np.degrees(np.arccos(np.dot(v2, u2)/(np.linalg.norm(v2)*np.linalg.norm(u2)))))
#print(np.degrees(np.arcsin(np.dot(v2, u2)/(np.linalg.norm(v2)*np.linalg.norm(u2)))))

v3 = np.array([0, 1, 0])
u3 = np.array([1, -0.5, 0])

print(180 - np.degrees(np.arccos(np.dot(v3, u3)/(np.linalg.norm(v3)*np.linalg.norm(u3)))))
#print(np.degrees(np.arccos(np.dot(v3, u3)/(np.linalg.norm(v3)*np.linalg.norm(u3)))))
#print(np.degrees(np.arcsin(np.dot(v3, u3)/(np.linalg.norm(v3)*np.linalg.norm(u3)))))

v3 = np.array([0, 1, 0])
u3 = np.array([1, -1, 0])

print(180 - np.degrees(np.arccos(np.dot(v3, u3)/(np.linalg.norm(v3)*np.linalg.norm(u3)))))