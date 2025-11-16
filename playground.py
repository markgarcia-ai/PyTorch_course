# Read lines -> Lowercase -> Filter -> Count
"""
lines = read_lines("logs.txt")
filtered = filter_keyword(lines, "error")
counter = count_items(filtered)

for c in counter:
    print("Matches:", c)
"""

import sys

class FileReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.lines = []

    def read_lines(self):
        with open(self.file_path, 'r') as file:
            self.lines = file.readlines()
        return self.lines

class KeywordFilter:
    def __init__(self, lines, keyword):
        self.lines = lines
        self.keyword = keyword

    def filter_keyword(self):
        return [line for line in self.lines if self.keyword in line.lower()]

class Counter(items):
    def __init__(self,items):
        self.items = items

    def count(self)
        return len(self.items)


if __name__ == "__main__":
    reader = FileReader("logs.txt")
    lines = reader.read_lines()
    filter = KeywordFilter(lines, "error")
    filtered = filter.filter_keyword()
    counter = Counter(filtered)
    print(counter.count())
    print(filtered)
    print(lines)
    print("Matches:", counter.count())
    sys.exit(0)