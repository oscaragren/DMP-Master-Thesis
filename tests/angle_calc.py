import math

base = [0, -1, 0]

def get_sflex(arm_vector):
    return math.degrees(math.atan2(arm_vector[2], -arm_vector[1]))

print(get_sflex([0, -1, 0]))
print(get_sflex([0, 0, 1]))
print(get_sflex([0, 1, 0]))

def get_sabd(arm_vector):
    return math.degrees(math.atan2(arm_vector[0], -arm_vector[1]))

print(get_sabd([0, -1, 0]))
print(get_sabd([1, 0, 0]))
print(get_sabd([0, 1, 0]))