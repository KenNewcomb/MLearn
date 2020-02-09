with open('mnist_train.csv', 'r') as f:
    csv = f.readlines()
    part1 = csv[:30000]
    part2 = csv[30000:]

with open('mnist_train_part1.csv', 'w') as f:
    for line in part1:
        f.write(line)

with open('mnist_train_part2.csv', 'w') as f:
    for line in part2:
        f.write(line)
