# Sorting Algorithms

## Why study sorting?
- its a common problem in real world applications
- its a prereq for a lot of other algo's, meaning a lot of times data needs to be sorted.
- the basic algorithms of other things sometimes mimics searches we learn
- it also lends itself to analysis easily
- Often times, the sort itself dictates its time complexity

## Sorting Terminology

- Inversion - means data is out of order
- Swa -, process of moving things around
- In-place - uses data itself and nothing else
- Stable sort - order of data is intact
- Sentinel - a marker, not part of the actual data
- Offline - all our data is here upfront
- Online - data comes in chunks
- Comparison based - sorts based detecting inversion

# Classic Sorting Algos

## Bogo Sort
- takes data, creates random permutation
- check is perm is sorted
- if its sorted stop, if not repeat process
- is a Las Vegas Algorithm - a randomized algorithm whose output is correct, but runtime is not guaranteed.

Best case - linear, goes through elements once
Worst case - could never stop
Average - n!/2

### Code for Bogo Sort | [Source](https://www.geeksforgeeks.org/bogosort-permutation-sort/)
```python
def bogoSort(a):
    n = len(a)
    while (is_sorted(a)== False):
        shuffle(a)
  
def is_sorted(a):
    n = len(a)
    for i in range(0, n-1):
        if (a[i] > a[i+1] ):
            return False
    return True
  
def shuffle(a):
    n = len(a)
    for i in range (0,n):
        r = random.randint(0,n-1)
        a[i], a[r] = a[r], a[i]

```

## Bubble Sort
- iterate through an array swapping adjacent elements that are inverted
- repeat process, each time stopping one element less
- is a Decrease and Conquer by one Algo
- Popular Optimization: come back to it

### Code for Bubble Sort | [Source](https://www.geeksforgeeks.org/bubble-sort/)
```python
def bubbleSort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1] :
                arr[j], arr[j+1] = arr[j+1], arr[j]

```

### Time Complexity for Bubble Sort
- Worst and Average - O(n<sup>2</sup>)

### Proof (Mathematical Induction)

1. Base case: n is 1 or 0, clearly the array is sorted

2. Inductive Hypothesis: assume bubble sort works for n = k-1

3. clearly bubble moves the largest elemenet to the end following each comparison of two adjacent elements, large of two will be on the end.

4. After the largest of that step is at the end, we rely on the “inductive hypothesis” that Bubble Sort will correctly sort the remaining k-1 elements

### Analysis of iterative implementation
- Comparisons: (n-1) + (n-2) ... + 1 = O(n<sup>2</sup>)

Swaps: 
    - Worst - O(n<sup>2</sup>), makes one swap for every comparison
    - Best - O(n), array is sorted and makes one pass
    - Average -  only swaps for half the comparison, still O(n<sup>2</sup>)

## Cocktail Sort
### Overview
- Similar to bubble, but goes in both directions left to right, right to left
- moves largest to last pos, by swapping adjacent elements left to right
- moves smallest to first, by swapping right to left
- first and last no longer have to be considered
- come back to this
- advantage is it works well with very small and very large numbers

### Code for Cocktail Sort | [Source](https://www.geeksforgeeks.org/cocktail-sort/)
```python
def cocktailSort(a):
    n = len(a)
    swapped = True
    start = 0
    end = n-1
    while (swapped == True):
        swapped = False
        for i in range (start, end):
            if (a[i] > a[i + 1]) :
                a[i], a[i + 1]= a[i + 1], a[i]
                swapped = True
        if (swapped == False):
            break
        swapped = False
        end = end-1
        for i in range(end-1, start-1, -1):
            if (a[i] > a[i + 1]):
                a[i], a[i + 1] = a[i + 1], a[i]
                swapped = True
        start = start + 1

```

## Selection Sort
### Overview
- finds index of smallest element of unsorted portion, swaps it with current index. its now in its final place after this. decrease unsorted portion.
- repeat process.
- only one swap per iteration, unlike bubble sort that could do one for every element in a pass.
- is a "Decrease by one and conquer"

### Selection Sort Code | [Source](https://www.geeksforgeeks.org/selection-sort/)
```python
# A here is an array
for i in range(len(A)):
    min_idx = i
    for j in range(i+1, len(A)):
        if A[min_idx] > A[j]:
            min_idx = j
    A[i], A[min_idx] = A[min_idx], A[i]

```
### Analysis
- O(n<sup>2</sup>) comparisons for best, average and worst
- worst and average swaps are O(n)
- best case for swaps are no swaps

## Insertion Sort
- array is divided into sorted(left) and unsorted (right)
- inserts current index into sorted portion, shifting elements over. 

- Does 2 comparisons instead of 1.

- Trick: put a really small number in front of the list, to save a comparison against index.

- Optimization: use a binary search, instead of linear to find the sorted area it needs to be. However you still need to shift elements to make space, so it might not really be an optimization

### Code for Insertion Sort | [Source](https://www.geeksforgeeks.org/insertion-sort/)
```python
def insertionSort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j] :
                arr[j + 1] = arr[j]
                j -= 1
        arr[j + 1] = key

```

### Analysis
- Worst case - O(n<sup>2</sup>)
- Average case - O(n<sup>2</sup>)
- Best case - O(n)
- there are no swaps here, but you do some insertion

## Shell Sort
### Overview 
- Sorts in discontinous chunks,
- Intervals are some number like every 3rd or 5th number.
- Optimal Sizes to use for the gaps? Research is ongoing, nothing in conclusive
- Could be the worst sort there is

### Code for Shell Sort | [Source](https://www.geeksforgeeks.org/shellsort/)
```python
def shellSort(arr):
    n = len(arr)
    gap = n//2
  
    # Do a gapped insertion sort for this gap size.
    # The first gap elements a[0..gap-1] are already in gapped order
    # keep adding one more element until the entire array is gap sorted
    while gap > 0:
        for i in range(gap,n):
            # add a[i] to the elements that have been gap sorted
            # save a[i] in temp and make a hole at position i
            temp = arr[i]
            # shift earlier gap-sorted elements up until the correct location for a[i] is found
            j = i
            while  j >= gap and arr[j-gap] >temp:
                arr[j] = arr[j-gap]
                j -= gap
            # put temp (the original a[i]) in its correct location
            arr[j] = temp
        gap //= 2
```

### Analysis
- Difficult to analyze
- Depends on gaps and other considerations
- estimates include n<sup>2</sup>, n<sup>3/2</sup>, n<sup>4/3</sup>

## Merge Sort
### Overview
- classic divide and conquer
- divide the array into two parts roughly,
left to mid, mid to right
- sort halfs recursively,
- merge when single elements have been placed
- works better when chunks are sorted kinda
- out of place algo, needs aux space O(n)

### Code for Merge Sort | [Source](https://www.geeksforgeeks.org/merge-sort/)
```python
def mergeSort(arr):
    if len(arr) >1:
        mid = len(arr)//2 #Finding the mid of the array
        L = arr[:mid] # Dividing the array elements 
        R = arr[mid:] # into 2 halves
        mergeSort(L) # Sorting the first half
        mergeSort(R) # Sorting the second half
        # Merge the two halves...
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i+=1
            else:
                arr[k] = R[j]
                j+=1
            k+=1
        while i < len(L):
            arr[k] = L[i]
            i+=1
            k+=1
        while j < len(R):
            arr[k] = R[j]
            j+=1
            k+=1
```


### Analysis
- T(1 or 0) = 0, nothing to sort
- There are two sub problems, left and sub arrays, half size of the previous
- merging lists always takes n computations
- worst, best and average case - O(n log n)

### Master Theorem (Copied from slides, not simplified)
General method for recurrences for describe a divide and conquer algorithm

T(n) = **a**T(n/**b**) + **f(n)**
>There are **a** sub-problems<br>
Each sub-problem is of size **n/b**<br>
f(n) is the work involved in dividing the problem into <br>subproblems and/or later combining the solutions into a master solution<br>
Assume that f(n) = O(nc)

Three cases:
- Case 1: logba < c : T(n) = Θ(nc)
- Case 2: logba = c : T(n) = Θ(nc logbn)
- Case 3: logba > c : T(n) = Θ(n ^ logba) 


### Analysis with Master Theorem (copied from slides, not simplified)
> The time complexity of Merge Sort can be described by the recurrence<br>
T(1) = 0, T(n) = 2T(n/2) + n-1<br>
Applying the Master Theorem, a = 2, b = 2, c = 1 <br>
logba = log22 = 1 = c, so Case 2 of the Master Theorem is satisfied.<br>
Therefore, T(n) = Θ(n1 log2n) = Θ(n logn)

Note that this is true in the best, average, and worst cases because f(n) = Θ(n) 

### Merge Sort Analysis (Solve by Recurrence) (copied from slides)
>To solve the recurrence, assume n = 2k and k = lg n (lg denotes log2)<br>
   Use a domain transformation<br>
   S(k) = T(n) = 2T(n/2) + n = 2S(k-1) + 2k, S(0) = T(1) = 0
	If T(n) = S(k), then T(n/2) = S(k-1) because 2k = n, so 2k-1 = n/2<br>
Use a range transformation<br>
      R(k) = S(k)/2k = 1/2k  x (2S(k-1) + 2k) = R(k-1) + 2k/2k =  R(k-1) + 1 <br>
	2S(k-1) / 2k  = S(k-1) / 2k-1 = R(k-1)<br>
Using telescoping – adding up the equations (see next slide for details)<br>
 	R(k) – R(0) = k, but R(0) = 0, so R(k) = k<br>
S(k) = R(k) x 2k  = k x 2k<br>
T(n) = S(k) = lg n x 2lg n = lg n x n = Θ(n log n)<br>  
Note: a^logan = n^logaa = n1 = n<br>

### Merge Sort by Telescoping (copied from slides)
>R(k) = R(k-1) + 1<br>
R(k) - R(k-1) = 1<br>
R(k-1) - R(k-2) = 1<br>
R(k-2) - R(k-3) = 1<br>
R(k-3) - R(k-4) = 1<br>
…<br>
R(1) - R(0) = 1<br>
	Add all the k equations together<br>
	The right term on the left cancels with the left term on the next line<br>
	R(k) - R(0) = k (there are k equations, each with 1 on right hand side)<br>
	R(0) = S(0) / 20 = T(1) = 0 <br>
R(k) - 0 = k so R(k) = k  <br>



## Stooge Sort (Educational purposes only)
### Overview
- sort first 2/3 of the elements
- sort last 2/3
- sort first 2/3 to be 

### Code for Stooge sort | [Source](https://www.geeksforgeeks.org/stooge-sort/)
```python
def stoogesort(arr, low, high):
    if low >= high:
        return
    if arr[low]>arr[high]:
        arr[low], arr[high] = arr[high], arr[low] 
    if high-low + 1 > 2:
        third = (int)((high-low + 1)/3)
        # Recursively sort first 2/3 elements
        stoogesort(arr, low, high-third)
        # Recursively sort last 2/3 elements
        stoogesort(arr, low + third, high)
        # Recursively sort first 2/3 elements again to confirm
        stoogesort(arr, low, high-third))
```

### Master Theorem for Stooge Sort (copied from slides)
>Recurrence is T(n) = 3T(n * 2/3) + 1 = 3T(n / (3/2)) + 1<br>
There are three sub-problems, each is ⅔ the size of the original<br>
Each sub-problem requires 1 unit of work, in addition to solving the problem itself<br>
a = 3, b = 3/2, c = 0<br>
logba = log3/23 = 2.7 vs. c = 0 <br>
note: log3/23 = ln(3) / ln(3/2)<br>
Case 3 T(n) = Θ(n^ log3/23) = Θ(n2.7) <br>
## Quick Sort
### Overview
- choose a pivot x i position k
- ideally the pivot is in the middle
- how to choose a pivot?
    - Median of three (first, middle and last)
    - Randomized pivot
- Partition the array based on the pivot, smaller elements to the left, bigger to the right. 
- repeat till base case swap, 
- how to get smaller parts to the left and larger to the right? 
### Code for quick sort | [Source](https://www.geeksforgeeks.org/quick-sort/)
```python
def partition(arr,low,high):
    i = ( low-1 )     	
    pivot = arr[high] 	
    for j in range(low , high):
        if   arr[j] < pivot:
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i]
  
    arr[i+1],arr[high] = arr[high],arr[i+1]
    return ( i+1 )
  
def quickSort(arr,low,high):
    if low < high:
        pi = partition(arr,low,high)
        quickSort(arr, low, pi-1)
        quickSort(arr, pi+1, high)

```
### Analysis
- Can't use master theorem here
- Worst - n<sup>2</sup>
- Best - n log n
- Average - n log n

## Binary Search Tree (BST) Sort
### Overview
- Build a binary search tree
    - start with an empty BST, insert elements in order
- InOrder Traversal (L,M,R) will give elements in order

### Analysis
Building the tree:
- best, n log n, height is n
- average, log n
- worst, n<sup>2</sup>
Going through the tree takes at least N
Time Complexity is dominated by the building of the tree

### Quick and BST connection
- pivot is like root
- partition by smaller and larger       

## HeapSort 
### Overview
its a tree, but follows a min or max heap property. root is either largest or smallest in tree, and children are smaller of larger than parent.

### Heapsort Code | [Source](https://www.geeksforgeeks.org/heap-sort/)
```python
def heapify(arr, n, i):
    largest = i 
    l = 2 * i + 1 	
    r = 2 * i + 2 	
    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r
    if largest != i:
        arr[i],arr[largest] = arr[largest],arr[i] 
        heapify(arr, n, largest)
  
def heapSort(arr):
    n = len(arr)
    for i in range(n, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i] 
        heapify(arr, i, 0)
```

### Heapsort Analysis
- buildheap takes O(n)
- heapsort removing max and moving to end takes O(1)
- restoring heap is O(log n)
- repeating this process n times brings our time to n log n.
- O(n log n) is the worst, best and average case time. 

## BST and Quicksort Average case Analysis (come back to this)