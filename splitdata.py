import sys

file = sys.argv[1]
with open(file) as f:
    lines = f.readlines()
    lines = [l.strip() for l in lines]
    split = int(len(lines) * 0.9)
with open(file[:-4] + '-train.tsv', 'w') as f:
    for line in lines[:split]:
        print(line, file=f)
with open(file[:-4] + '-dev.tsv', 'w') as f:
    for line in lines[split:]:
        print(line, file=f)
