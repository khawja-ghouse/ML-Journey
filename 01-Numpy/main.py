import numpy as np


# my_list = [1,2,3,4]

# my_list = my_list * 2
# print(my_list)
# print(type(my_list))
# print(np.__version__)

array = np.array([1,2,3,4])
array = array * 2
print(array.ndim)

twoD_array = np.array([[1,2,3],
                       [4,5,6]])

print(twoD_array.ndim)
print(twoD_array.shape)

threeD_array = np.array([[[1,2,3],[4,5,6],[7,8,9]],
                         [[11,22,33],[44,55,66],[77,88,99]],
                         [[111,222,333],[444,555,666],[777,888,999]]])

print(threeD_array.ndim)
print(threeD_array.shape)
print(threeD_array.size)

# Row selection
# array[start:end:step]
# print(threeD_array[0:1:])

# Column Selection
# array[row selection,column]
print(threeD_array[:,1:1])