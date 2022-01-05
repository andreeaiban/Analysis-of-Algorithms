# Lecture 3

### All sorts we've covered so far have been comparison based

## <mark>Decision tree model for comparison-based sorts</mark>
### Overview
- bottom of tree are called leafs, terminal vertices and external vertices
- n! permutations of a tree
- It is possible to arrive to the same decision taking different routes

### Analysis
- each non-leaf corresponds to a comparison question. we go left of right to answer a question
- there are at least n! leaves
    - n! possible permutations of elements
    - we say at least because some permutations may appear more than once, because we can get to it with different decisions
    - you need at least N log N time to sort stuff at average and best.
    - If no sorting is done, you still need N time to check to see if things are in the right spot.

## <mark>Non-comparison based sorts
- we can actually get better times than n log n, if we use sorts that don't use decision-tree comparisons!
- bucket sort uses a comparison kinda. don't tell anyone. 
---

## <mark>Counting Sort</mark>
### Overview
- We DO NOT compare elements, recording frequencies only
- Init a frequency array to 0
    - one cell for each possible value in range
- count the frequency of each element in the input
- output one copy of the number for each frequency
    - if 5 appears 4 times, output 4 5's
- you have a sorted list
- works only with a fixed range, large ranges make take too much space. theres also an issue with decimals
- if you have decimals, use powers of 10 to make them whole numbers and use sort

### Code for Counting Sort | [Source](https://www.geeksforgeeks.org/counting-sort/)
```python
def countSort(arr):
  
    # 256 is the number of difference possible characters
    output = [0 for i in range(256)]
    count = [0 for i in range(256)]
    ans = ["" for _ in arr]
    for i in arr:
        count[ord(i)] += 1
    # Change count[i] so that it contains actual position of this character in output array
    for i in range(256):
        count[i] += count[i-1]
    # Build the output character array
    for i in range(len(arr)):
        output[count[ord(arr[i])]-1] = arr[i]
        count[ord(arr[i])] -= 1
    # Copy the output array to arr, so that arr contains sorted characters
    for i in range(len(arr)):
        ans[i] = output[i]
    return ans 
```

### Analysis
- initializing the zero-array is O(range of elements) or m
- counting the frequencies of each element in the range is O(n)
- moving the data to make room is O(n)
- finally, outputting the data is O(n)
- total time is really O(n+m), but we can really say O(n)
---

## <mark>Radix Sort</mark>
### Overview
- Perform counting sort, but digit by digit
- Starts from one end to the other, neither is wrong
- We will use least significant to most, AKA right to left
- Requires another stable sort, so we dont lose order on sorts we did earlier. almost never used on its own.
- Digit doesn't have to be a number, it can be an character, or numbers in other bases

### Code for Radix Sort | [Source](https://www.geeksforgeeks.org/radix-sort/)
```python
def radixSort(arr):
  
    # Find the maximum number to know number of digits
    max1 = max(arr)
  
    # Do counting sort for every digit. 
    # Instead of passing digit number, exp is passed. exp is 10^i where i is current digit number
    exp = 1
    while max1/exp > 0:
        countingSort(arr,exp)
        exp *= 10
```

### Analysis
- Time for counting sort = n, and number of digits=d is O(dn) = O(n)
- keep in mind d is a constant
- Time could be n<sup>2</sup> if d is some how dependent on n
---

## <mark>Bucket Sort</mark>
### Overview
- Create empty buckets that will hold slices of data
    - bucket 0 holds 0-99
    - bucket 1 holds 100-199 and so forth
- place data into buckets and sort them
- concatenate buckets to get your sorted array
- when buckets get small enough, do an insertion on each cause its fast.
- works well if you can get small buckets. 
- its a hybrid of sorts that kinda uses a comparison, to place stuff in buckets

### Bucket Sort Code | [Source](https://www.geeksforgeeks.org/bucket-sort-2/)
```python
def bucketSort(x):
    arr = []
    slot_num = 10 # 10 means 10 slots, each slot's size is 0.1
    for i in range(slot_num):
        arr.append([])
          
    # Put array elements in different buckets 
    for j in x:
        index_b = int(slot_num * j) 
        arr[index_b].append(j)
      
    # Sort individual buckets 
    for i in range(slot_num):
        arr[i] = insertionSort(arr[i])
          
    # concatenate the result
    k = 0
    for i in range(slot_num):
        for j in range(len(arr[i])):
            x[k] = arr[i][j]
            k += 1
    return x
```

### Analysis
- worst case: 
    - all n elements use one bucket, and sorted using something that is n<sup>2<s/up>
    - Sorting the buckets, you chose a sort thats like n<sup>2</sup> time or n log n. might as well just sort normally.
- average case: 
    - you get uniformly distributd buckets that get sorted with insertion sorts.
    - you need b insertion sorts for b number of buckets
    - picking a b that is dependent on the number of elements will lead to overall time becoming linear AKA O(n).
    - if you pick a static b or something not dependent, you can end up with something thats n<sup>2</sup>
- pick the right number of buckets, its the only thing preventing this from going to n<sup>2</sup>
---

# <mark>Order Statistics</mark>
 **"i<sup>th</sup> order statistic"** is the i<sup>th</sup> smallest element in a set of n elements. <br>
Thus,<br> 
- “1<sup>st</sup> order statistic” is the same as “minimum”<br>
- “n<sup>th</sup> order statistic” is the same as “maximum”<br>
- “(n/2)<sup>th</sup> order statistic” is the same as “median” or 50<sup>th</sup> percentile<br>
- Need special handling depending on whether n is odd or even<br>
- “(n/4)<sup>th</sup> order statistic” is the same as 25<sup>th</sup> percentile<br>
- “(3n/4)<sup>th</sup> order statistic” is the same as 75<sup>th</sup> percentile<br>

## How do we get min's, max's and median's without sorting?
--- 

## <mark>Calculating Min's and Max's
### Linear Approach - Overview

come back to this
### Code for linear 
```python
def getMin(arr, n): 
    mi = arr[0] 
    for i in range(1,n): 
        mi = min(mi, arr[i]) 
    return mi 
def getMax(arr, n): 
    ma = arr[0] 
    for i in range(1,n): 
        ma = max(ma, arr[i]) 
    return ma 
```
### Analysis
- we know this take n time
- still only a linear approach

### Recursive Approach 1
```python
def getMin(arr, n):
    if (n==1):
        return a[0]
    else   
    	return min(arr[n-1], getMin(arr, n-1))
```
## Analysis
- So this is actually a "Decrease by one and Conquer" design
- T(0) = T(1) = 0; T(n) = 1 + T(n-1) 
- still N time needed to compute

### Recursive Approach 2
```python
def getMin(arr, start, end):
    if (start == end):
        return a[start]
    else:
        mid = (start + end) // 2
    	return min(getMin(arr, start, mid), getMin(arr, mid+1, end))
```
- Uses "Divide and Conquer" design
- T(0) = T(1) = 0; T(n) = 1 + 2T(n/2)
- Master Theorem: a = 2, b = 2, c = 0; log<sub>2</sub>s = 1 > 0 so T(n) = O(n<sup><sup>log<sub>2</sub>2</sup></sup>) = O(n)  
- Same as iterative and recursive Decrease-and-Conquer approaches 

### Simultaneous Approach
```python
def getMinMax(arr, n):
   mi = arr[0]
   ma = arr[0]
   for i in range(1, n): 
      if arr[i] < mi: 
         mi = arr[i] 
      elif arr[i] > ma: 
         ma = arr[i] 
   return mi, ma 

```
- find both at the same time
- only does second comparison half the time, because if its less than the min it wont be higher than the max. 
- Worst:
    - T(n) = 2(n-1), means it goes through the array twice
- Best:
    - T(n) = n-1, means it goes through the array once
- Average: 
    - T(n) = 3(n-1)/2, each pointer will probably cross each other a bit to find their value. you're doing an interation and a bit more.

## <mark>Calculating the "i<sup>th</sup> order statistic"</mark>
- sort the data!
    - which ones to use? 
- modify a sort to iteratively finds the next min or max
    - seletion, bubble, heap
- quick select
    - variation of quick sort, partitions and checks partition point and only focuses on one half. the value you want is only greater or less than your pivot.
    - can do really bad if you get unlucky with pivots
- median of medians
    - Pseudo-median (approximation) can be used in Quick Sort and Quick Select
    - Can be used recursively to find actual median


## <mark>Quick Select</mark>
### Code | [Source](https://www.geeksforgeeks.org/quickselect-algorithm/)
```python
# Use the partition algorithm from Quick Sort - not repeated here 

def kthSmallest(arr, l, r, k): 
    if (k > 0 and k <= r - l + 1): 
        index = partition(arr, l, r) 
  
        if (index - l == k - 1): 
            return arr[index] 
  
        if (index - l > k - 1): 
            return kthSmallest(arr, l, index - 1, k) 
        else:
            return kthSmallest(arr, index + 1, r, k - index + l - 1) 
    return INT_MAX 
```
- Worst - O(n<sup>2</sup>)
- Best - O(n)
- Decent - O(n)
- Average - O(n)

can you take worst case down from n<sup>2</sup>? Median of Medians. 

## <mark>Median of Medians</mark>
### Code for Median of Medians | [Source](https://brilliant.org/wiki/median-finding-algorithm/) 
```python
def median_of_medians(A, i):
    #divide A into sublists of len 5
    sublists = [A[j:j+5] for j in range(0, len(A), 5)]
    medians = [sorted(sublist)[len(sublist)/2] for sublist in sublists]
    if len(medians) <= 5:
        pivot = sorted(medians)[len(medians)/2]
    else:
        #the pivot is the median of the medians
        pivot = median_of_medians(medians, len(medians)/2)
    #partitioning step
    low = [j for j in A if j < pivot]
    high = [j for j in A if j > pivot]
    k = len(low)
    if i < k:
        return median_of_medians(low,i)
    elif i > k:
        return median_of_medians(high,i-k-1)
    else: #pivot = k
        return pivot
```

### Median of Five (copied from slides)
>The “Median of Medians” algorithm assumes we can easily compute the median of five elements. (Initially there are n/5 groups of 5 elements each.)<br><br>
There are five possible medians, with C(4, 2) = 4!/(2!2!) = 6 ways to group the other four elements, two on each side of a median, for 30 possibilities.<br><br>
How do we  find the median of five?<br>
Sort the five elements using Insertion Sort in (5 * 4 / 2) = 10 comparisons -<br>
 https://en.wikipedia.org/wiki/Median_of_medians#Algorithm<br><br>
Compute it directly in 7 comparisons: log(5!) ~= 6.91 -
https://cs.stackexchange.com/questions/44981/least-number-of-comparisons-needed-to-sort-order-5-elements/44982#44982 <br><br>
Compute it directly in 6 comparisons -
https://cs.stackexchange.com/questions/45374/prove-that-minimal-number-of-comparisons-to-find-median-among-five-elements-is-5


### Median of Medians Analysis (copied from slides)
Recurrence is T(n) ≤ T(n/5) + T(7n/10) + cn
- T(n/5) is for finding the median of the n/5 medians
    - the middle row of the diagram on the previous page

- cn is for the partitioning in Quick Select
    - this is a multiple of n (probably 1)
    - The reason for c will become apparent on next slide  
- T(7n/10) is for the recursion in Quick Select
    - Will not exceed 70% of the elements
    - For every 10 elements (a pair of columns in the diagram) at least 3 elements will be on one side of the overall median of the medians, leaving at most 7 elements on the bigger side


### Quick Select + Median of Medians: Analysis (copied from slides)
What is the solution to T(n) ≤ T(n/5) + T(7n/10) + cn?
Proof by induction…
Assume "Inductive Hypothesis" T(m) ≤ 10cm for m < n (smaller cases)
Now plug into recurrence:

- Smaller case: T(n/5) ≤  10 c n/5 = 2cn
- Smaller case: T(7n/10) ≤ 10 c 7n/10 = 7cn
- cn is just 1cn

So larger case n: T(n) ≤ 2cn + 7cn + cn = 10cn 
QED

Thus, T(n) ≤ 10cn = O(n), even in the worst case!	
