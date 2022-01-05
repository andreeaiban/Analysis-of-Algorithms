
# What is an Algorithm?

At the very core, a sequence of well-defined steps to solve a problem or do something. 

It should do have an output AKA do something, it might not always need an input or prerequsite.

# Requirements for an Algorithm

There should be an input of sorts, sometimes there is not. 

There should also be an output

Each instruction should be clear, concise and simple.

There should also be an end to the algorithm, it can't go on forever. Very few cases where it goes on forever.

It should output something correct for all test cases. However sometimes, we're willing to accept an approximation or something that is correct most of the time. 

# Classifications of Algorithms

You can classify algorithms by their use cases, some are used in data science and some are used in cyber security

You can also classify them by the type of problem they solve

This course will focus on algorithms by design technique, AKA Divide and Conquer, Greedy and Brute Force/Naive

They also classify problems by how feasible it is to solve said problem. How hard, easy or possible it is to solve a given problem. 

1. super easy and can be seen without much thought.
2. difficult but can be taught to be easy.
3. genuinely hard to solve by experts.
4. actually impossible to solve currently.

Algorithms can also be classified by the how much resources they use AKA time and space. input space, auxillary space. Time does not mean actual time. Time is really computation. 

# Time Complexity

 Best Case - T<sub>b</sub>(n)

Average Case - T<sub>a</sub>(n)

Worst Case - T<sub>w</sub>(n)

We will primarily be focusing on worst case, cause we don't learn much about efficiancy and how it behaves overall.

In cases where we have more than 1 input, you might see somthing like T<sub>w</sub>(n,m)

# Asymptotic Notation

We are mostly concerned with how algo's work with varying sizes of inputs. Particularly what happens with large amounts of input. 

[Aymptotic Cheat Sheet from MIT, there was too much information to put](http://web.mit.edu/broder/Public/asymptotics-cheatsheet.pdf)

# Searches

## Linear Search
### Overview
- Starts the beginning of the array
- Keeps going till the elemennt in question is found
- returns the index, or -1 if not found
- array doesnt need to be sorted
- can be done iterative or recursive

### Code: Iterative

```python
def search(arr,n,x):

    for i in range(0,n):
        if arr[i] == x :
            return i

    return -1
```
### Analysis
- one comparison at the current index
- loops through the array until found
- formulate a recurrence equation
- T(n) = number of comparisons needed to search N elements
- T(1) = 1
- T(n) = 1 - T(n-1), Because we always need to do at least one comparison

### Solving the Recurrence (Telescoping, when terms cancel out each other)

1. T(n) = 1 - T(n-1)
2. T(n) - T(n-1) = 1
3. Decrement n's and they still hold up
4. Example T(n-1) - T(n-2) =1, and T(n-2) - T(n-3) = 1
5. Add all the possible ones up.
6. T(n) - T(1) = n - 1
7.  Analysis states T(1) is 1, T(n) - 1 = n - 1
8. T(n) = n - 1 + 1, T(n) = n

### Best, Worst and Average Case

Best - T(n) = 1

Worst - T(n) = n

Average - T(n) = n/2, assuming its in the array
- if not in the array, goes to n
---

## Binary search

### Overview
- Sorts the array if not sorted
    -  adds time to the algo, may not be worth
- Starts at the middle of the array
- Checks key, returns key if correct
- If not key, determines which half to check
- Repeats previous 3 steps on sub array
- Divides until found or cant be further divided
- Can be recursive or iterative

### Code: Recursive | [Link for source](https://www.geeksforgeeks.org/python-program-for-binary-search/)
```python
def binary_search(arr, low, high, x):

    if high >= low:
 
        mid = (high + low) // 2
 
        if arr[mid] == x:
            return mid

        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)
 
        else:
            return binary_search(arr, mid + 1, high, x)
 
    else:
        return -1
```
### Analysis

- T(0) = 0 as no operations will be done
- T(1) = 1 as we need to check to the key regardless
- T(n) = T(n/2) + 1 because we need one comparison to determine which half to breakdown and search, then search that half
- How many comparisons does this really do? 3. one index comparison and two for data. 
- Aside: Comparing data CAN be expensive and in general will be more than checking indexes.

### Solving for recurrence (Expansion, plugging in terms recursively to look for a pattern)

- T(n) = 1 + T(n/2)
- = 1 + (1 + T(n/4))
- = 2 + (1 + T(n/8))
- = 3 + (1 + T(n/16))
- you eventually end up with k + T(n/2<sup>k</sup>)
- which is 1 + log<sub>2</sub>n

**Next two section are just copied from slides, because i dont have a good understanding of it yet**
### Solve Recurrence by Domain Expansion
- Start with T(n) = T(n/2) + 1
- Substitution: assume n = 2k so k = log2n  
- Note: n/2 = 2k / 2 = 2k-1 
- Then S(k) = T(n) and S(k-1) = T(n/2) and S(0) = T(20) = T(1) = 1 
- We have now transformed the domain (input) from n to k
- We call this a “Domain Transformation”
- So T(n) = 1 + T(n/2) can be rewritten as
- S(k) = 1 + S(k-1) or 
- S(k) - S(k-1) = 1


### Master theorem for Divide and Conquer Algo's

More of a general kind of thing, i can't explain it any better than the slides currently. 

- T(n) = aT(n/b) + f(n)
- There are a sub-problems
- Each sub-problem is of size n/b
- f(n) is the work involved in dividing the problem into subproblems and/or later combining the solutions into a master solution
Assume that f(n) = O(nc)
- Three cases:
    - Case 1: logba < c : T(n) = Θ(nc)
    - Case 2: logba = c : T(n) = Θ(nc logbn)
    - Case 3: logba > c : T(n) = Θ(n ^ logba) 

Recurrence is T(n) = T(n/2) + 1
- a=1, one sub-problem
- b=2, half the size of the original problem
- c=0, f(n) = 1 = n0

log21 = 0

Case 2 givesT(n) = Θ(n0 log2n) = Θ(log n) 

---

## Interpolation

### Overview

- Sort the array if it's not already sorted
- "Guestimate" where in the array the item is located
- Repeat that process as needed
- Analogy: classic phonebook ("white pages")
- Works well with evenly distributed data, not as good with lopsided data AKA data that varies greatly in a short span. 

### Code - Recursive | [Source](https://www.geeksforgeeks.org/interpolation-search/ )
```python 
def interpolationSearch(arr, lo, hi, x):
    if (lo <= hi and x >= arr[lo] and x <= arr[hi]):
     	pos = lo + ((hi - lo) // (arr[hi] - arr[lo]) * (x - arr[lo]))
		if arr[pos] == x:
			return pos
		if arr[pos] < x:
			return interpolationSearch(arr, pos + 1, hi, x)
		if arr[pos] > x:
			return interpolationSearch(arr, lo, pos - 1, x)
	return -1
```

### Analysis 
- Space: O(1), no added data needed just that array for each sub problem
- Time:
    - Worst: O(n), it goes through all the entries
    - Average: O(log(log n)), elements are distributed nicely
---
## Jump Search 
### Overview
- Sort the array if not already sorted
- Rather than jump one (Linear Search), or half the remaining array (Binary Search), we jump at other increments
- Best increment determined to be m = √n
- Time complexity between that of linear and binary searches
- Good for finding elements in extreme positions

### Code - Recursive | [Source](https://www.geeksforgeeks.org/jump-search/ )
```python
def jumpSearch( arr , x , n ):
	step = math.sqrt(n)
	prev = 0
	while arr[int(min(step, n)-1)] < x:
		prev = step
		step += math.sqrt(n)
		if prev >= n:
			return -1
	while arr[int(prev)] < x:
		prev += 1
		if prev == min(step, n):
			return -1
	if arr[int(prev)] == x:
		return prev
	return -1
```



