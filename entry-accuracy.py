import sys
different_lines = 0                                           
with open(sys.argv[1]) as a, open('validation/REFERENCE.csv') as b:
    for i, line in enumerate(a, 1):
        other_line = b.readline()
        if line != other_line:
            different_lines += 1
print(i, (i - different_lines) / i)
