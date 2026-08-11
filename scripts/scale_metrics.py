import sys

#Quick min-max scaling based on control methods
#Arguments: unscaled scores, control scores, output scaled scores

controlFile = open(sys.argv[2])
controlFile.readline()
mmDict = {}
for line in controlFile:
    tokens = line.strip().split(",")
    data_metric = tokens[0] + "," + tokens[1]
    vals = [float(x) for x in tokens[2:]]
    mmDict[data_metric] = (min(vals), max(vals))
controlFile.close()

outFile = open(sys.argv[3], "w")
inFile = open(sys.argv[1])
print(inFile.readline().strip(), file=outFile)
for line in inFile:
    tokens = line.strip().split(",")
    data_metric = tokens[0] + "," + tokens[1]
    print(data_metric, end="", file=outFile)
    if data_metric in mmDict:
        minV, maxV = mmDict[data_metric]
    else:
        print("Unable to find", data_metric)
        minV = 0
        maxV = 1
    if maxV == minV:
        print("No variation for", data_metric)
        print("", *["0.0"]*(len(tokens)-2), sep=",", file=outFile)
        continue
    for val in tokens[2:]:
        scaled = (float(val) - minV) / (maxV - minV)
        if scaled < 0: #Cap values at range of control methods
            print("scaled value", val, "is", scaled, "for", data_metric)
            scaled = 0
        if scaled > 1:
            print("scaled value", val, "is", scaled, "for", data_metric)
            scaled=1
        print(",", scaled, sep="", end="", file=outFile)
    print(file=outFile)
outFile.close()
inFile.close()

