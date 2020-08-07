list1 = [1, 2, 3, 4, 5, 6]
print(list1)
list2 = [1, 2, 3, "Parth Roy", 5, [1515, "dasd"]]
print(list2)
tuple1 = tuple(list1)
print(tuple1)
tuple2 = tuple(list2)
print(tuple2)
tuple3 = ("Parth", "Roy", 1234, 489, [2, 3, 4, 5, "Roy"])
print(tuple3[1])
print(tuple3[4])
print(tuple3[0:-1])
dict1 = {"Name": "Parth Roy", "Address": "Surat,Gujarat",
         1: "Temp123", "list": [20, 10, "ABCXYZ"]}
print(dict1)

dict1.pop("Address")
print("Remove Address : ")
print(dict1)

del dict1["list"]
print("Remove list : ")
print(dict1)
