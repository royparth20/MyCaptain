def fibonacci(a, b):
    return a+b


input_number = int(input("Enter Number of length Fibonacci Series : "))
n1 = 0
n2 = 1
if input_number > 0:
    print("Series : ", end=" ")
    print(n1, end=" ")
for i in range(input_number-1):
    print(n2, end=" ")
    x = fibonacci(n1, n2)
    n1, n2 = n2, x
