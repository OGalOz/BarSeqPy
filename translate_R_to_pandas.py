import pandas as pd
import numpy as np
"""

Functions to translate:
    merge: Can be substituted with pandas merge OR pandas join. Simple union join how="outer",
            intersection join -> how="inner". 
    cbind: Stands for column-bind. Combines columns, vectors, matrices, or dataframes by column names
    names:
    apply(value_to_be_iterated_over, 1/2 (rows or cols), function to apply): 
        e.g.
            apply(abs(lrn), 1, quantile, percentile)
            where lrn is a dataframe, 1 means over the rows,
            quantile is (?) and percentile is a float
    mapply(): Applies function to multiple lists or vectors - Example:
                    values1 <- list(a = c(1, 2, 3), b = c(4, 5, 6), c = c(7, 8, 9))
                    values2 <- list(a = c(10, 11, 12), b = c(13, 14, 15), c = c(16, 17, 18)
                    Then mapply(function(num1, num2) max(c(num1, num2)), values1, values2)
                    returns 
                        a   b   c
                        12  15  18
                    Essentially iterating over the columns of values1 and values2 in parallel
    lapply(list, function(x)) (apply function to every element x from list and return list of equal length) 
        (like list iteration)
    mclapply is a parallelized version of lapply - 'multi core list' apply
    tapply:  Apply a function to each cell of a ragged array, that is to each (non-empty) group of values given by a unique combination of the levels of certain factors.
    writeDelim: Custom function, writes table out to file without the row names
    rowMeans: takes the means of the rows in a dataframe
    rowSums:
    colSums: # colSums returns a vector with the sum for each column
            in pandas: df.sum(axis=0)
    split: Returns grouping by value in the list - translated by 'py_split'
    unique: returns a list of values which don't repeat
    which: returns a list of indexes at which some boolean expresses to be true
    aggregate: Combines dataframes with a function?
    match: returns a vector of the positions of (first) matches of its first argument in its second
    order: Returns indeces of locations of sorted values - Can also be substituted by
            py_order, OR df.sort_values(by=['col1', 'col2'])
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
            type returned is an integer vector
    runmed: equivalent to pandas.Series.rolling(window (int)).median() reference:
             https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.median.html 
    density: Assuming it's "Gaussian", From scipy import stats.gaussian_kdeA
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    qnorm: scipy.stats.norm.ppf()
           https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    cor:  Cor in R just returns a single number which is the correlation between two vectors.
          To get this in pandas, use Series correlation: 
            https://pandas.pydata.org/docs/reference/api/pandas.Series.corr.html
            series1.corr(series2, method="spearman") etc.
    rep: rep(x,) replicates the values in X
    approx: Use pandas series "interpolate". approx essentially approximates intermediate values,
            up to a given 'n', say n=50. So you have (1,2,3) -> ([1] 1 , ...., [25] 2, ...., [50] 3)
    pmax: parallel maximum - given two vectors get the max from each index
    unsplit: Given a dict with label -> list of values, AND a list of labels that corresponds to 
            those labels from that dict, we can create a dataframe that matches a row for every time
            that same label comes up in the list of labels.
    ifelse: (test -> bool, yes, no)
    quantile: quantile(df$col, probs = c(0.05, 0.95))
                    5%  95%
                    11.995 31.300,
            Where 11.995 and 31.300 are the values at those percents
            For a vector, in pandas, use:
                Series.quantile(q=0.5, interpolation='linear')
                https://pandas.pydata.org/docs/reference/api/pandas.Series.quantile.html
            For a dataframe use:
                DataFrame.quantile(q=0.5, axis=0, numeric_only=True, interpolation='linear')
                https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.quantile.html

    which:
        Given a boolean vector, returns the indices where True
        
   
"""

def py_unsplit(inp_d, list_o_vals, columns=None):
    """
    Args:
        inp_d (dict): val (str) -> list<elem>
        list_o_vals (list<val (str)>):
        columns list<str>: If columns is None, we assume this is a Series
    Description:
        We go through each value in list_o_vals, which should correspond to a key
        in the inp_d. And the number of elements in inp_d in the list corresponding
        to that value should be the same as the total number of times you find that
        value in the list_o_vals (i.e. at the beginning:
            list_o_vals.count(val) = len(inp_d[val]))

    Returns:
        DataFrame with one row per value in list_o_vals
    """
    return_vals = []
    for val in list_o_vals:
        return_vals.append(inp_d[val].pop(0))
    if columns is not None:
        main_result = pd.DataFrame(return_vals, columns=columns)
    else:
        main_result = pd.Series(return_vals)

    return main_result

def py_order(srs_a, reverse=False, tie_breaker=None):
    """
    Args:
        srs_a (pandas Series): A list of sortable values
        tie_breaker (list): An optional list of values, same length as list_a,
                            that is a tiebreaker for list_a
    Returns:
        orders (list<int>): A list of length list_a with the indeces for the sorted values
                            (0-indexed) default is ascending
    """


    if tie_breaker is None:
        z = srs_a.sort_values() 
    else:
        if len(srs_a) != len(tie_breaker):
            raise Exception("For sorting, length of tie_breaker list must be the same as original list.")
        sorting_df = pd.DataFrame.from_dict({"A": srs_a,
                                             "B": tie_breaker})
        sorting_df.set_index(srs_a.index)
        sorted_df = sorting_df.sort_values( by=["A","B"] )
    
        z = sorted_df["A"]

    z_list = list(z)
    z_index = z.index
    if None in z:
        return [z_index[z_list.index(val)] for val in srs_a if val is not None] + [None]
    else:
        return [z_index[z_list.index(val)] for val in srs_a]






def py_match(list_a, list_b):
    """
    Args:
        list_a: list of values to check loc in list_b
        list_b: list of values to check list_a against
    Returns:
        A list<int> with length list_a with indeces of first matches
            within list_b

    Desc:
        e.g. 
        list_a = ["a", "b", "c", "d", "e"]
        list_b = ["f", "a", "c", "b", "d", "d"]
        result: [1, 3, 2, 4, None]
    """
    return [list_b.index(val) if val in list_b else np.nan for val in list_a]


def py_split(pd_series, group_by_list, typ="indices"):
    """
    Returns groups defined by the group_by_list
    The group names are the index labels for
    the pd_series (?)
    

    Args:
        typ (str): can be one of ['indices', 'groups']

    Note:
        indices returns the index numbers of the groupings,
        so for example if you group by some value
        then for that value you will get a list of indexes
        within the pd_series in which that value matches
        (Numerical indexes)
        groups returns the index labels of the groupings,
        so not the numerical value but the index label.
    """
    grouped_series = pd_series.groupby(by=group_by_list)

    if typ == "indices":
        return grouped_series.indices
    elif typ == "groups":
        return grouped_series.groups
    else:
        raise Exception("Did not recognize split type")


def py_mult_vect(vector_A, vector_B):
    """
    We multiply each value of the two vectors (which have equal length)
    and return a vector of equal length with the products.
    e.g. [1,2,3] * [4,5,6] = [4,10,18]
    
    Args:
        vector_A: list<Number>
        vector_B: list<Number>
    Returns:
        list<Number>
    """
    if len(vector_A) != len(vector_B):
        raise Exception("When running py_mult_vect you must use two vectors of equal length.")
    new_vect = []
    for i in range(len(vector_A)):
        new_vect.append(vector_A[i] * vector_B[i])
    return new_vect

#from collections import Counter

def py_table(list_of_str, return_unique=False):
    """
    Args:
        list_of_str could also be pd series of str.
        return_unique: if you want a list of the unique values,
            set this to True
    Returns:
        ret_d (dict): of object -> number of times 
        it appears in the list
        {str -> int, for each str in list_of_str}
    """
    if type(list_of_str) == pd.core.series.Series:
        list_of_str = list(list_of_str)

    ret_d = {}
    for x in list_of_str:
        if x in ret_d:
            ret_d[x] += 1
        else:
            ret_d[x] = 1

    if return_unique: 
        return [ret_d, list(ret_d.keys())]
    else:
        return ret_d 


def py_aggregate(dataframe_A, group_by_label, func='sum',
                reset_index_bool=False):
    """
    
    Question: 
        Does this return a DataFrame/ Series with the new unique
        group by labels as the 'index' of the dataframe?


    group dataframe_A according to items
    in list and apply function over them.
    func = ['sum' or 'mean']
    For example, suppose data frame has values:

    x1 x2 x3 group 
     1  2  1  A
     2  3  1  A
     3  4  1  B
     4  5  1  C
     5  6  1  C

    and group_by_label has value 'group' 

    And the function is sum.

    Then what we get as a result,
    is the following:

    A sum of values of rows 1 &2
    B  values of row 3 (untouched) 
    C sum of values of rows 4 & 5

    group x1 x2 x3
    A     3   5   2
    B     3   4   1
    C     9   11  2

    Returns:
        res (pandas DataFrame):
            res is a data frame with the same columns as dataframe_A
            but now the group_by_label column has only unique entries
            and for each unique entry the values of the other columns
            are the sums/means/medians of those values
    """

    gb = dataframe_A.groupby(group_by_label)

    if func == 'sum':
        res = gb.sum()
    elif func == 'mean':
        res = gb.mean()
    elif func == 'median':
        res = gb.median()
    else:
        raise Exception("Func must be str from ['sum', 'mean']")
        
    if reset_index_bool:
        return res.reset_index()
    else:
        return res



def py_aggregate_df_series(dataframe_A, aggregate_series, col_name,
                  func='sum'):
    """
    How it works:
        Add the series over which to aggregate to dataframe_A
        by creating a new dataframe with the column aggregate_series
        under the column name 'col_name'.
        Then sum over the repeated elements within the new column.

    group dataframe_A according to items
    in list group and apply function over them.
    e.g. dataframe_A is

    x1 x2 x3 
     1  2  1 
     2  3  1 
     3  4  1 
     4  5  1 
     5  6  1 

    aggregate_series with col_name 'group' is
        A
        A
        B
        C
        C

    And the function is sum.

    Then we get as a result:

    A sum of values of rows 1 &2
    B  values of row 3 (untouched) 
    C sum of values of rows 4 & 5

    group x1 x2 x3
    A     3   5   2
    B     3   4   1
    C     9   11  2
    """
    # deep copy is the default
    df_copy = dataframe_A.copy(deep=True)
    df_copy[col_name] = aggregate_series
    gb = df_copy.groupby(group_by_label)

    if func == 'sum':
        res = gb.sum().reset_index()
        return res
    elif func == 'mean':
        res = gb.mean().reset_index()
        return res
    elif func == 'median':
        res = gb.median().reset_index()
        return res
    else:
        raise Exception("Func must be str from ['sum', 'mean']")


def py_aggregate_series_to_series(to_func_series, to_func_col_name,
                                  aggregate_series, aggregate_col_name,
                                  func='sum'):
    """
    Args:
        to_func_series: pandas series to sum over e.g.
        to_func_col_name (str): name of series in df
        aggregate_series: pandas series to group by
        aggregate_col_name: name of the aggregate by series in df.
        func: constrained vocab: ['sum', or 'mean']
    How it works:
        Combine the two series into a dataframe, 
        by creating a new dataframe with the column aggregate_series
        Then sum over the repeated elements within the new column.

    """

    pre_aggregate_df = pd.DataFrame.from_dict({
                                    to_func_col_name: to_func_series,
                                    aggregate_col_name: aggregate_series
                                    })

    gb = pre_aggregate_df.groupby(aggregate_col_name)

    if func == 'sum':
        res = gb.sum().reset_index()
        return res
    elif func == 'mean':
        res = gb.mean().reset_index()
        return res
    elif func == 'median':
        res = gb.median().reset_index()
        return res
    else:
        raise Exception("Func must be str from ['sum', 'mean']")



def py_tapply_series(series_a, values_to_sum, func_str='median'):
    """
    Args:
        series_a: pandas series
        values_to_sum: list<str>
        func_str: str
    """
    return None


def match_ix(A, B, dbg_bool=False):
    """
    # A and B are lists, we find the 0-indexed indeces
    # of matches for every item in A within B and return
    # an array with those matches
    # Returns a list of length A
    """
    if dbg_bool:
        print("Matching two lists:")
        print("First list (10)")
        print(A[:10])
        print("Second list (10)")
        print(B[:10])

    match_list = []
    for i in range(len(A)):
        if A[i] in B:
            match_list.append(B.index(A[i]))
        else:
            if dbg_bool:
                print("Could not find value {A[i]} in list B.")
            match_list.append(np.nan)

    return match_list

def stop(inp_int):
    raise Exception(f"Stopping: {inp_int}")
