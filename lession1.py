list1 = [int(x) for x in input("Enter List").split()]
print(list1)
list2= []
for x in list1:
    if x >=0:
        list2.append(x)
print(list2)