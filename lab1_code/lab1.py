def OR(x1, x2):
    return 1 if x1 + x2 >= 1 else 0

def AND(x1, x2):
    return 1 if x1 + x2 == 2 else 0

def XOR(x1, x2):
    return OR(x1, x2) - AND(x1, x2)

print("x1 x2 | OR AND XOR")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(x1, x2, "  |", OR(x1, x2), AND(x1, x2), XOR(x1, x2))