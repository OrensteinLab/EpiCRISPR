from csv import reader
import csv

FIXED_LEN = 32

"This file parse bed and bed narrowpeak files and creates epigenetic files"


# Reads BED file and returns a dictionary of the locations with epigenetic marker for each chromosome
def parseBed(filename):
    dictionary = {}
    with open(filename) as f:
        lines = f.readlines()
        firstLine = lines[0].split('\t')
        chrom = firstLine[0][3:]
        start = int(firstLine[1])
        end = int(firstLine[2])
        for i in range(1, len(lines)):
            lineList = lines[i].split('\t')
            newChrom = lineList[0][3:]
            newStart = int(lineList[1])
            newEnd = int(lineList[2])
            if chrom != newChrom:
                if chrom in dictionary.keys():
                    dictionary[chrom].append((start, end))
                else:
                    dictionary[chrom] = [(start, end)]
                chrom = newChrom
                start = newStart
                end = newEnd
            elif chrom == newChrom and newStart == end:
                end = newEnd
            elif chrom == newChrom and end != newStart:
                if chrom in dictionary.keys():
                    dictionary[chrom].append((start, end))
                else:
                    dictionary[chrom] = [(start, end)]
                start = newStart
                end = newEnd
    return dictionary


# Creates the epigenetic file
def createEpigentics(dictionary, file, output):
    dict = {}
    with open(file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        next(csv_reader)
        line = 1
        for row in csv_reader:
            chromNum = row[0]
            # the start and end locations refers to only 20nt of guideRNA. If looking for epigenetic 0f optimal length
            # of 32 the new locations are presented here. For another number of nt location should be changed.
            start = int(row[1]) + 13
            end = int(row[2]) - 15
            temp = ""
            for tup in dictionary[chromNum]:
                # entire guide has epigenetic marker
                if start >= tup[0] and end <= tup[1]:
                    temp += "1" * FIXED_LEN
                    dict[line] = temp
                    line += 1
                    break
                # when part of the guide has epigenetic mark
                elif start < tup[0] < end <= tup[1]:
                    temp += "0" * (int(tup[0]) - int(start) + 1)
                    temp += "1" * (FIXED_LEN - len(temp))
                    dict[line] = temp
                    line += 1
                    break
                # the guide doesn't have epigenetic marker
                elif tup[0] < start < tup[1] <= end:
                    temp += "1" * (int(tup[1]) - int(start) + 1)
                    temp += "0" * (FIXED_LEN - len(temp))
                    dict[line] = temp
                    line += 1
                    break
            if temp == "":
                dict[line] = "0" * FIXED_LEN
                line += 1
            else:
                temp = ""
    # writing to epigenetic file
    with open(f'{output}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'epigenetics'])
        for k in dict.keys():
            writer.writerow([k, dict[k]])


if __name__ == '__main__':
    dic = parseBed(".bed")  # input bed file name
    createEpigentics(dic, "T_data.csv", 'epi_file')  # gives as input file of gRNAs with efficiency values
                                                                 # output to specific file name
