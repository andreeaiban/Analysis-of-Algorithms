# Design & Analysis of Algorithms (CSCI 323)
# Winter Session 2022
# Assignment 3: This program Native Search(python),Brute-Force,Knuth-Morris-Pratt,Rabin-Karp, Rabin-Karp
# to analyze string-search (aka pattern-matching) algorithms from random text & patters
# @Andreea Ibanescu


import string
import random
import time
import pandas as pd
import matplotlib.pyplot as plt


# random text https://www.educative.io/edpresso/how-to-generate-a-random-string-in-python


def random_text(n):
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(n))


def random_pattern(size, text):
    index = random.randint(0, len(text) - size)
    return text[index:index+size]

# native search


def native_sort(pat, txt):
    txt.find(pat)


# Brute-force  (https://www.geeksforgeeks.org/naive-algorithm-for-pattern-searching/ )


def brute_search(pat, txt):
    m = len(pat)
    n = len(txt)

    # A loop to slide pat[] one by one */
    for i in range(n - m + 1):
        j = 0

        # For current index i, check
        # for pattern match */
        while j < m:
            if txt[i + j] != pat[j]:
                break
            j += 1

        if j == m:
          print("Pattern found at index ", i)

# Knuth- Morris-Pratt https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/ )


def kmp_search(pat, txt):
    m = len(pat)
    n = len(txt)

    # create lps[] that will hold the longest prefix suffix
    # values for pattern
    lps = [0] * m
    j = 0  # index for pat[]

    # Preprocess the pattern (calculate lps[] array)
    compute_lps_array(pat, m, lps)

    i = 0  # index for txt[]
    while i < n:
        if pat[j] == txt[i]:
            i += 1
            j += 1

        if j == m:
            print("Found pattern at index " + str(i - j))
            j = lps[j - 1]

        # mismatch after j matches
        elif i < n and pat[j] != txt[i]:
            # Do not match lps[0..lps[j-1]] characters,
            # they will match anyway
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1


def compute_lps_array(pat, m, lps):
    len = 0
    # length of the previous longest prefix suffix the loop calculates lps[i] for i = 1 to M-1
    lps[0]
    # lps[0] is always 0
    i = 1
    while i < m:
        if pat[i] == pat[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            # This is tricky. Consider the example. AAACAAAA and i = 7. The idea is similar to search step.
            if len != 0:
                len = lps[len - 1]
                # Also, note that we do not increment i here
            else:
                lps[i] = 0
                i += 1

# Rabin Karp Algorithm given in CLRS book (https://www.geeksforgeeks.org/rabin-karp-algorithm-for-pattern-searching/ )


def karp_search(pat, txt):
    d = 256
    q = 101

    def rabin_karp_search(pat, txt):
        M = len(pat)
        N = len(txt)
        i = 0
        j = 0
        p = 0  # hash value for pattern
        t = 0  # hash value for txt
        h = 1
        # The value of h would be "pow(d, M-1)%q"
        for i in range(M - 1):
            h = (h*d) % q
        # Calculate the hash value of pattern and first window of text
        for i in range(M):
            p = (d*p + ord(pat[i])) % q
            t = (d*t + ord(txt[i])) % q
        # Slide the pattern over text one by one
        for i in range(N - M + 1):
            # Check the hash values of current window of text and pattern if the hash values match then only check for characters on by one
            if p == t:
                # Check for characters one by one
                for j in range(M):
                    if txt[i + j] != pat[j]:
                        break
                    else:
                        j += 1
                # if p == t and pat[0...M-1] = txt[i, i+1, ...i+M-1]
                if j == M:
                    print("Pattern found at index " + str(i))
            # Calculate hash value for next window of text: Remove leading digit, add trailing digit
            if i < N - M:
                t = (d(t - ord(txt[i]) * h) + ord(txt[i + M])) % q
                # We might get negative values of t, converting it to positive
                if t < 0:
                    t = t + q


# of Boyer Moore String Matching Algorithm (https://www.geeksforgeeks.org/boyer-moore-algorithm-for-pattern-searching/ )


def bad_char_heuristic(string, size):
    no_of_chars = 256

    # Initialize all occurrence as -1
    badChar = [-1] * no_of_chars

    # Fill the actual value of last occurrence
    for i in range(size):
        badChar[ord(string[i])] = i;

    # return initialized list
    return badChar


def boyer_search(txt, pat):

    m = len(pat)
    n = len(txt)

    # create the bad character list by calling
    # the preprocessing function badCharHeuristic()
    # for given pattern
    badChar = bad_char_heuristic(pat, m)

    # s is shift of the pattern with respect to text
    s = 0
    while s <= n - m:
        j = m - 1

        # Keep reducing index j of pattern while
        # characters of pattern and text are matching
        # at this shift s
        while j >= 0 and pat[j] == txt[s + j]:
            j -= 1

        # If the pattern is present at current shift,
        # then index j will become -1 after the above loop
        if j < 0:
            print("Pattern occur at shift = {}".format(s))
            s += (m - badChar[ord(txt[s + m])] if s + m < n else 1)
        else:
            s += max(1, j - badChar[ord(txt[s + j])])

#Plot function


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


def plot_times_bargraph(dict_searches, sizes, searches):
    sort_num = 0
    plt.xticks([j for j in range(len(sizes))], [str(size) for size in sizes])
    for sort in searches:
        sort_num +=1
        d = dict_searches[sort.__name__]
        x_axis = [j + 0.05 * sort_num for j in range(len(sizes))]
        y_axis = [d[i] for i in sizes]
        plt.bar(x_axis, y_axis, width=.05, alpha=.5, label=sort.__name__)
    plt.legend()
    plt.title("Run time of search algorithms")
    plt.xlabel("Number of elements")
    plt.ylabel("Time for 100 trials in ms")
    plt.savefig("String_search_graph_bar.png")
    plt.show()


# main function


def main():
    # Smaller numbers because my laptop will crash
    max_int = 100
    trials = 5
    searches = [native_sort, brute_search, kmp_search, karp_search, boyer_search ]
    sizes = [100, 200, 300,400]
    dict_searches = {}

    # for loop
    for search in searches:
        dict_searches[search.__name__] = {}
    for size in sizes:
        for search in searches:
            dict_searches[search.__name__][size] = 0
        # create the random text & it's pattern for this trial
        txt = random_text(size)
        pat_size = int(size / 10)
        pat = random_pattern(pat_size, txt)

        for search in searches:
            start_time = time.time()
            # search through text and patterns for the text
            search(pat, txt)

            # calculate net time for each searching algo
            end_time = time.time()
            net_time = (end_time - start_time) * 1000
            dict_searches[search.__name__][size] += net_time

    df = pd.DataFrame.from_dict(dict_searches).T
    print(df)
    plot_times_bargraph(dict_searches, sizes, searches)


if __name__ == '__main__':
    main()

