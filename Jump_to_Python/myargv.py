import sys

numbers = sys.argv[1:]

result = 0
print(numbers)
print(type(numbers))
for number in numbers:
    result += int(number)
print(result)