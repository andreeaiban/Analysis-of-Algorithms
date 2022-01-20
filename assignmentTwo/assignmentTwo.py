# Design & Analysis of Algorithms (CSCI 323)
# Winter Session 2022
# Assignment 2: This program uses sort algorithms, then compares and contrasts their time complexitity
# @Andreea Ibanescu

import math
import random
import time
import pandas as pd
import matplotlib.pyplot as plt

# native sort


def native_sort(arr):
    arr.sort()

# https://www.geeksforgeeks.org/bubble-sort/


def bubble_sort(arr):
    n = len(arr)
    # Traverse through all array elements
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n-i-1):
            # traverse the array from 0 to n-i-1 Swap if the element found is greater than the next element
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

# https://www.geeksforgeeks.org/selection-sort/


def selection_sort(arr):
    # Traverse through all array elements
    for i in range(len(arr)):
        # Find the minimum element in remaining unsorted array
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[min_idx] > arr[j]:
                min_idx = j
        # Swap the found minimum element with the first element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

# https://www.geeksforgeeks.org/insertion-sort/


def insertion_sort(arr):
    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):
        key = arr[i]
        # Move elements of arr[0..i-1], that are greater than key, to one position ahead of their current position
        j = i-1
        while j >= 0 and key < arr[j] :
                arr[j + 1] = arr[j]
                j -= 1
        arr[j + 1] = key

# https://www.geeksforgeeks.org/cocktail-sort/


def cocktail_sort(a):
    n = len(a)
    swapped = True
    start = 0
    end = n - 1
    while swapped == True:
        # reset the swapped flag on entering the loop because it might be true from a previous iteration.
        swapped = False
        # loop from left to right same as the bubble. sort
        for i in range(start, end):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                swapped = True
        # if nothing moved, then array is sorted.
        if swapped == False:
            break
        # otherwise, reset the swapped flag so that it can be used in the next stage
        swapped = False
        # move the end point back by one, because item at the end is in its rightful spot
        end = end - 1
        # from right to left, doing the same  comparison as in the previous stage
        for i in range(end - 1, start - 1, -1):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                swapped = True

        # increase the starting point last stage would have moved the next smallest number to its rightful spot
        start = start + 1

# https://www.geeksforgeeks.org/shellsort/


def shell_sort(arr):
    gap = len(arr) // 2  # initialize the gap
    while gap > 0:
        i = 0
        j = gap
        # check the array in from left to right till the last possible index of j
        while j < len(arr):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j += 1
            # now, we look back from ith index to the left
            # we swap the values which are not in the right order.
            k = i
            while k - gap > -1:

                if arr[k - gap] > arr[k]:
                    arr[k - gap], arr[k] = arr[k], arr[k - gap]
                k -= 1

        gap //= 2

# https://www.geeksforgeeks.org/merge-sort/


def merge_sort(arr):
    if len(arr) > 1:
        # Finding the mid of the array
        mid = len(arr)//2
        # Dividing the array elements
        L = arr[:mid]
        # into 2 halves
        R = arr[mid:]
        # Sorting the first half
        merge_sort(L)
        # Sorting the second half
        merge_sort(R)
        i = j = k = 0
        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
# https://www.geeksforgeeks.org/quick-sort/


def quick_sort(arr):
    quick_sort_real(0, len(arr) - 1, arr)


# https://www.geeksforgeeks.org/quick-sort/
def partition(start, end, array):
    pivot_index = start
    pivot = array[pivot_index]

    while (start < end):

        while start < len(array) and array[start] <= pivot:
            start += 1

        while array[end] > pivot:
            end -= 1
        if start < end:
            array[start], array[end] = array[end], array[start]
    array[end], array[pivot_index] = array[pivot_index], array[end]
    #  print("partition", end)
    return end


def quick_sort_real(start, end, array):
    if start < end:
        p = partition(start, end, array)
        # print("partition", start, end)
        quick_sort_real(start, p - 1, array)
        quick_sort_real(p + 1, end, array)


# https://www.geeksforgeeks.org/heap-sort/


def heapify(arr, n, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1
    r = 2 * i + 2
    # See if left child of root exists and is
    # greater than root
    if l < n and arr[largest] < arr[l]:
        largest = l
    # See if right child of root exists and is greater than root
    if r < n and arr[largest] < arr[r]:
        largest = r
    # Change root, if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # swap
        # Heapify the root.
        heapify(arr, n, largest)


def heap_sort(arr):
    n = len(arr)

    # Build a maxheap.
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)

    # One by one extract elements
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # swap
        heapify(arr, i, 0)


# https://www.geeksforgeeks.org/counting-sort/


def count_sort(arr):
    max_element = int(max(arr))
    min_element = int(min(arr))
    range_of_elements = max_element - min_element + 1
    # Create a count array to store count of individual
    # elements and initialize count array as 0
    count_arr = [0 for _ in range(range_of_elements)]
    output_arr = [0 for _ in range(len(arr))]
    # Store count of each character
    for i in range(0, len(arr)):
        count_arr[arr[i] - min_element] += 1
    # Change count_arr[i] so that count_arr[i] now contains actual position of this element in output array
    for i in range(1, len(count_arr)):
        count_arr[i] += count_arr[i - 1]
        # Build the output character array
    for i in range(len(arr) - 1, -1, -1):
        output_arr[count_arr[arr[i] - min_element] - 1] = arr[i]
        count_arr[arr[i] - min_element] -= 1
        # Copy the output array to arr, so that arr now contains sorted characters
    for i in range(0, len(arr)):
        arr[i] = output_arr[i]
    return arr

    # https://www.geeksforgeeks.org/bucket-sort-2/


def bucket_sort(arr):
    bucket_sort_helper(arr, int(len(arr)/2))


# https://www.geeksforgeeks.org/bucket-sort-2/
def bucket_sort_helper(arr, no_of_buckets):
    max_ele = max(arr)
    min_ele = min(arr)

    # range(for buckets)
    rge = (max_ele - min_ele) / no_of_buckets

    temp = []

    # create empty buckets
    for i in range(no_of_buckets):
        temp.append([])

    # scatter the array elements into the correct bucket
    for i in range(len(arr)):
        diff = (arr[i] - min_ele) / rge - int((arr[i] - min_ele) / rge)

        if diff != 0 or arr[i] == min_ele:
            temp[int((arr[i] - min_ele) / rge)].append(arr[i])

        # append the boundary elements to the lower array
        else:
            temp[int((arr[i] - min_ele) / rge) - 1].append(arr[i])
    # Sort each bucket individually
    for i in range(len(temp)):
        if len(temp[i]) != 0:
            temp[i].sort()
    # Gather sorted elements to the original array
    k = 0
    for lst in temp:
        if lst:
            for i in lst:
                arr[k] = i
                k = k + 1

# https://www.geeksforgeeks.org/radix-sort/ TWO FUNCTIONS USED FOR RADIX SORT: counting & radix_sort


def counting(arr, exp1):
    n = len(arr)
    # The output array elements that will have sorted arr
    output = [0] * (n)
    # initialize count array as 0
    count = [0] * (10)
    # Store count of occurrences in count[]
    for i in range(0, n):
        index = arr[i] // exp1
        count[index % 10] += 1
    # Change count[i] so that count[i] now contains actual position of this digit in output array
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build the output array
    i = n - 1
    while i >= 0:
        index = arr[i] // exp1
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
    # Copying the output array to arr[],so that arr now contains sorted numbers
    i = 0
    for i in range(0, len(arr)):
        arr[i] = output[i]


def radix_sort(arr):
    # Find the maximum number to know number of digits
    max1 = max(arr)
    # Do counting sort every num  instead of passing number, exp is passed. exp is 10^i, i is current number
    exp = 1
    while max1 / exp > 1:
        counting(arr, exp)
        exp *= 10


# Make a list


def random_list(max_int, size):
    arr = [random.randint(0, max_int) for i in range(size)]
    return arr


def plot_times_line_graph(dict_searches):
    for search in dict_searches:
        x = dict_searches[search].keys()
        y = dict_searches[search].values()
        plt.plot(x, y, label=search)
    plt.legend()
    plt.title("Run Time of Search Algorithms")
    plt.xlabel("Number of Elements")
    plt.ylabel("Time for 100 Trials (ms)")
    plt.savefig("search_graph.png")
    plt.show()


def plot_times_bar_graph(dict_searches, sizes, searches):

    search_num = 0
    plt.xticks([j for j in range(len(sizes))], [str(size) for size in sizes])

    for search in searches:
        search_num += 1
        d = dict_searches[search.__name__]
        x_axis = [j + .05 * search_num for j in range(len(sizes))]
        y_axis = [d[i] for i in sizes]
        plt.bar(x_axis, y_axis, width=.05, alpha=.25, label=search.__name__)
    plt.legend()
    plt.title("Run Time of Search Algorithms")
    plt.xlabel("Number of Elements")
    plt.ylabel("Time for 100 Trials (ms)")
    plt.savefig("search_graph.png")
    plt.show()


# function to check if array is sorted

def check_sort(arr):
    if all(arr[i] <= arr[i + 1] for i in range(len(arr)-1)):
        print("Yes, List is sorted.")
    else:
        print("No, List is not sorted.")

# main function

def main():
    max_int = 100
    trials = 5
    dict_sorts = {}
    sorts = [native_sort, bubble_sort, selection_sort, insertion_sort, cocktail_sort, shell_sort, merge_sort,quick_sort, heap_sort, count_sort,bucket_sort]
    sizes = [100,200,300]

    for sort in sorts:
        dict_sorts[sort.__name__] = {}
    for size in sizes:
        for sort in sorts:
            dict_sorts[sort.__name__][size] = 0
        for trial in range(1, trials):
            array = random_list(max_int, size)
            for sort in sorts:
                arr_copy = array.copy()
                start_time = time.time()
                sort(arr_copy)
                end_time = time.time()
                net_time = end_time - start_time

                if trial == trials - 1:
                    dict_sorts[sort.__name__][size] /= trials

                dict_sorts[sort.__name__][size] += 1000 * net_time
                print(sort.__name__)
                check_sort(arr_copy)
                # print(arr_copy, sort.__name__)


    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = pd.DataFrame.from_dict(dict_sorts).T
    print(df)
    plot_times_bar_graph(dict_sorts, sizes, sorts)


if __name__ == '__main__':
    main()

