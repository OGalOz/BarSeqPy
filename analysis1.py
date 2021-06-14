
import os
import logging
import pandas as pd
import numpy as np
from scipy import stats
import json
import statistics
import sys
import math 
import time
from datetime import datetime
from og_util import debug_print 
from translate_R_to_pandas import *





def analysis_1(all_df, exps_df, genes_df, 
               expsT0, t0tot, 
               genesUsed, genesUsed12, strainsUsed,  
               minGenesPerScaffold=10, meta_ix=7,debug=False, nDebug_cols=None,
               starting_debug_col=0):
    """
    Questions: is all_df at all modified? 
    exps_df has modified column names right?
    genes_df modified?

    Args:
        all_df: The all.poolcount complete table (no removals)
        expsT0:
            dict {t0_date -> list experiment_names}
        t0tot:
            dataframe cols (same str as expsT0 keys)
            num rows the same as all_df
            each row is a sum of all_df over the related T0 vals (from expsT0)
        genesUsed:
            list<locusIds (str)> whose length defines the size of the dataframes
                created in the future.
        genesUsed12:
           list<locusIds (str)>  a more stringent list of locusIds- they have to have
                                an abundant enough number of strains in the first
                                AND second half (0.1<f<0.5 & 0.5<f<0.9)
        strainsUsed:
            list<bool> Length of all_df which decides which of the 'strains'
            we actually use. Must pass two qualifications:
            The mean of the t0tot over the strain has to pass a threshold 
                'minT0Strain'
            The insertion location of the strain has to be between 0.1 and 0.9
            of a gene.
            It also only includes strains with locusIds that
            are in genesUsed
        starting_debug_col (int): Should we start running experiments at some point
                                  after the first option? E.g. start at experiment 20?
                                  Default is 0, so start at the beginning.
        nDebug_cols (int or None): How many columns should we run through (int)
                                    All of them (None)

    Returns:
        GeneFitResults: (dict) set_index_names -> gene_strain_fit_result
            gene_strain_fit_result (dict):
                gene_fit: DataFrame, contains cols:
                    fit, fitNaive, fit1, fit2, fitnorm, fitnorm1, fitnorm2, fitRaw
                    locusId, n, nEff, pseudovar, sumsq, sd, sdNaive, se, t, tot1
                    tot0_1, tot2, tot2_0, tot, tot0
                strain_fit: pandas Series (float) with a computation applied to values
                strain_se: pandas Series (float) with a computation applied to values

    Description:
        In this part of the analysis, we compute fitness per strain as well as
        fitness per Gene. The way we do this is that we go through each experiment,
        which are all the columns in all_df (all.poolcount dataframe) after the
        metadata columns.
        The first thing we do is we get a list of all the experiment names, and
        we place these in a list called 'all_index_names'. We also get the subset
        of all the strains that are useful for Gene Fitness computations, the
        number of strains in this subset is the number of 'True's in the list
        'strainsUsed', so we compute this number again (it is named 
        'nAllStrainsCentralGoodGenes'). After that we get the subsets of the 
        dataframes all_df and t0tot (the Time0 totals summed over all_df) to
        be used in Gene Fitness computations later.
        Next, if one wanted to run the computations on a subset of the experiments, 
        they could give nDebug_cols=N where N is the number of experiments
        they would like to run analysis on.
        Finally, before running the analysis on each selected experiment,
        we find the subset of the good strains which were inserted in 
        the first half of the genes (in other words if the insertion occured
        before the halfway point of the gene). We call this array of booleans
        use1.
        Now we run the analysis on each experiment 
        (number of experiments = nDebug_cols)
        and each analysis returns a dictionary with 3 values: gene_fit (a pd dataframe
        with gene fitness values), strain_fit (a pd series with strain fitness values),
        and strain_se (a pd series with strain standard error values). Note that the
        number of rows in the dataframe gene_fit is equivalent to the total number
        of usable genes, whereas the two series are the same length, and are equivalent
        to the total number of strains in all.poolcount (much longer than the total
        number of usable genes).
        In a larger dictionary called, GeneAndStrainFitResults,
        we create a key with the current experiment, and the value is the dictionary
        mentioned above. In other words, we get a dictionary with keys 
        ['gene_fit', 'strain_fit', 'strain_se']
        within a dictionary called GeneAndStrainFit with keys being the experiment
        names, and each experiment is associated with one of the smaller dictionaries.
        We return this dictionary GeneAndStrainFitResults.
        The internal explanation of what happens during the analysis is described 
        in the function 'gene_strain_fit_func'.

        Function descriptions:
    """

    # The bulk of the program occurs here: We start computing values
    all_index_names = list(all_df.columns)[meta_ix:]
    nAllStrainsCentralGoodGenes = list(strainsUsed).count(True)
    if nAllStrainsCentralGoodGenes == 0:
        raise Exception("After data preparing, no usable strains are left.")
    print(f"nAllStrainsCentralGoodGenes: {nAllStrainsCentralGoodGenes}")

    # length of these two dataframe is nAllStrainsCentralGoodGenes
    # all_df_used are the original strains that we are using
    # t0tot_used are the t0total sums over those same strains
    all_df_used = all_df[strainsUsed]
    t0tot_used = t0tot[strainsUsed]


    # We take all the index names without the meta indeces (0-meta_ix (int))
    nSetIndexToRun = len(all_index_names) if nDebug_cols == None else nDebug_cols
    num_ix_remaining = nSetIndexToRun
    # use1 refers to the strains inserted in the first half of the gene
    use1 = [bool(x < 0.5) for x in all_df_used['f']]

    GeneAndStrainFitResults = {}
    print(f"Running through {num_ix_remaining}/{len(all_index_names)} possible experiments"
          f" starting at experiment number {starting_debug_col} from all")
    for set_index_name in all_index_names[starting_debug_col: starting_debug_col + nSetIndexToRun]:
        print(f"Currently working on index {set_index_name}")
        
        start_time = time.time()
        if set_index_name is not None:
            # We choose the right column
            exp_used_strains = all_df_used[set_index_name]
            # below is a dict with 3 keys: gene_fit (df), strain_fit (srs), strain_se (srs)
            gene_strain_fit_result_d = gene_strain_fit_func(set_index_name, 
                                                          exps_df, exp_used_strains, 
                                                          genes_df, expsT0,
                                                          t0tot_used, 
                                                          genesUsed, genesUsed12, minGenesPerScaffold,
                                                          all_df_used,
                                                          use1,
                                                          all_df)

            if gene_strain_fit_result_d is not None:
                GeneAndStrainFitResults[set_index_name] = gene_strain_fit_result_d
            else:
                print(f"For index {set_index_name} result was None")

        end_time = time.time()
        num_ix_remaining -= 1
        print(f"{num_ix_remaining}/{len(all_index_names)} left to run through")
        print(f"Estimated time remaining: {((end_time-start_time)*num_ix_remaining)/60} minutes.")
        print(f"Current time: {datetime.now().strftime('%H:%M:%S')} PST.")

    # If there are no 
    if len(GeneAndStrainFitResults.keys()) == 0:
        raise Exception("All comparisons failed.")

    if debug:
        print("passed GeneFitness section")

    return GeneAndStrainFitResults



def gene_strain_fit_func(set_index_name, exps_df, exp_used_strains, 
                         genes_df, expsT0,
                         t0tot_used, 
                         genesUsed, genesUsed12, minGenesPerScaffold,
                         all_df_used, use1,
                         all_df
                         ):
    """
    Description:
        This function is run for every single set_index_name in all_df, and that set_index_name
        is passed into this function as the first argument, 'set_index_name'. All other arguments
        are not changed at all when this function is called and are documented elsewhere. 
        We get results for GeneFitness (fitness per Gene), and Strain Fitness (fitness
        per strain). GeneFitness gives a dataframe whose number of rows is the number of genes
        in genesUsed. StrainFitness returns two series whose length is the same as the total
        number of strains in all.poolcount. StrainFitness is a very simple function that gives
        log2 ratios and standard error, each of which take only a line to compute. On the other
        hand, GeneFitness is a very complicated function in which multiple normalizations and
        statistical computations are done in order to make the GeneFitness a more reasonable
        value.
        Within this function, we first find the associated t0set name related to our current 
        experiment, and get the reads and t0 read sums. If our current experiment
        (denoted by the set_index_name) is a Time0 experiment, then we subtract the current
        experiment values from the sum for this Time0, since the sum for this Time0 is
        an aggregation over the several experiments associated with this Time0.
        If this is the only Time0 experiment associated with this Time0, then we
        skip computation for it. At this point, we get the total StrainFitness
        values, which, as mentioned above, returns two series whose length is the 
        same as the total number of strains in all.poolcount. These two series are
        StrainFitness and Standard Error per strain. We don't use these two series
        further within this function, now we move on to computing per Gene Fitness.
        When computing Gene Fitness, we use a subset of all.poolcount in which 
        the strains were inserted into genes and in good locations and in abundant 
        enough amounts over genes and over scaffolds. GeneFitness returns a Dataframe,
        and is a very complicated function which is described in its own Description
        section; it returns a dataframe with per Gene values.

        
        
    Args:
        set_index_name: (str) Name of experiment (set and index) from all_df (all.poolcount file)

        exps_df: Data frame holding exps file (FEBABarSeq.tsv)

        exp_used_strains: pandas Series of this set_index_name from all.poolcount
                                        with only values related to useful reads.
                                        Length is nAllStrainsCentralGoodGenes 

        [all_df_used]: Subset of the Data frame holding all.poolcount file with only the reads
                        that passed multiple threshold tests (Length is nAllStrainsCentralGoodGenes)
        genes_df: Data frame holding genes.GC table
        expsT0: (dict) mapping (date setname) -> list<experiment_name (str)>
        t0tot_used: data frame where column names are 'date setname'
                and linked to a list of sums over the indexes that relate 
                to that setname, (Length is nAllStrainsCentralGoodGenes) 

        genesUsed: list<locusId> where each locusId is a string
        genesUsed12 (list<str>): list of locusIds that have both high f (>0.5) and low f (<0.5)
                    insertions with enough abundance of insertions on both sides
        minGenesPerScaffold: int
        all_df_central_inserts (Dataframe): The parts of all_df that corresponds to True in strainsUsed
                                            Num rows is nAllStrainsCentral 
        use1: boolean list for the all_df_used with 0.1 < f <0.5 is True, otherwise false,
                Length is nAllStrainsCentralGoodGenes
        all_df: Just for Strain Fitness, the entire all dataframe not excluding any strains
        tot0: Just for Strain Fitness, the entire tot0 dataframe not excluding any strains

    Created vars:
        to_subtract: a boolean which says whether the 'short' name
                    is Time0
        t0set: Setname of related t0 set to current index name
        all_cix: The all_df column which is related to the current set_index_name
            (Should be a panda series)
        t0_series = the series from t0tot_used that is the current Time0 sums for each
                    strain

    Returns:
        returns None if there are no t0 values for it. Otherwise returns ret_d
        ret_d: (dict)
            gene_fit: DataFrame, contains cols:
                fit, fitNaive, fit1, fit2, fitnorm, fitnorm1, fitnorm2, fitRaw
                locusId, n, nEff, pseudovar, sumsq, sd, sdNaive, se, t, tot1
                tot0_1, tot2, tot2_0, tot, tot0
            strain_fit: pandas Series (float) with a computation applied to values
            strain_se: pandas Series (float) with a computation applied to values

    """
    # t0set is a string, to_subtract is a bool depending on if this set has short Time0
    t0set, to_subtract = get_t0set_and_to_subtract(set_index_name, exps_df)

    # t0_used is the related time 0 total series.
    t0_used = t0tot_used[t0set]

    # to_subtract is true if this is a time zero itself, so we remove
    # its values from the other time0 values.
    if to_subtract:
        # We subtract the poolcount values from the t0 totals 
        t0_used = t0_used - exp_used_strains 

    # We check if any value is under 0
    for ix, value in t0_used.iteritems():
        if value < 0:
            raise Exception(f"Illegal counts under 0 for {set_index_name}: {value}")
        if pd.isnull(value):
            logging.warning("Empty value in t0_used")

    # Checking if there are no control counts
    # If all are 0
    if t0_used.sum() == 0:
        logging.info("Skipping log ratios for " + set_index_name + ", which has no"
                     " control counts\n.")
        return None

    # Getting the cntrl values (besides this one if it is a Time0)
    cntrl = list(expsT0[t0set])
    if set_index_name in cntrl:
        cntrl.remove(set_index_name)
    if len(cntrl) < 1:
        raise Exception(f"No Time0 experiments for {set_index_name}, should not be reachable")

    strain_fit_ret_d = StrainFitness(all_df[set_index_name], 
                      all_df[cntrl].sum(axis=1),
                      debug_print=False
                      )

    all_used_locId = all_df_used['locusId'] 
    all_used_f = all_df_used['f']
    # We need to update the boolean indexing lists- program bound to fail.
    gene_fit = GeneFitness(genes_df, all_used_locId, 
                           exp_used_strains, all_used_f, 
                           t0_used,
    		           genesUsed, sorted(genesUsed12), 
    		           minGenesPerScaffold=minGenesPerScaffold,
                           set_index_name=set_index_name,
                           cdebug=False,
                           use1 = use1)
    
    # gene_fit, strain_fit, and strain_se
    ret_d = {"gene_fit": gene_fit, 
            "strain_fit": strain_fit_ret_d['fit'], 
            "strain_se": strain_fit_ret_d['se']
            }

    return ret_d


def get_t0set_and_to_subtract(set_index_name, exps_df):
    """ We use exps_df and set_index_name to find if this
        relates or belongs to a t0set, and if yes, which is the related
        t0set.
    Args:
        set_index_name: (str)
        exps_df: Dataframe of FEBABarSeq.tsv file
    Returns:
       t0set: (str) Related t0set to set_index_name
       to_subtract: (bool) Whether or not we need to subtract
            values (if this is a Time0)
    """

    # to_subtract is a boolean which says whether the short is a Time0 
    # t0set holds related t0set for the current index name
    t0set = None
    to_subtract = False
    for i in range(len(exps_df['name'])):
        if exps_df['name'].iloc[i] == set_index_name:
            if exps_df['short'].iloc[i].upper() == "TIME0":
                to_subtract = True 
            t0set = exps_df['t0set'].iloc[i]
            break

    return t0set, to_subtract



def GeneFitness(genes_df, all_used_locId, exp_used_strains,
                all_used_f, t0_used, genesUsed,
                genesUsed12, minGenesPerScaffold=None,
                set_index_name=None,
                base_se = 0.1,
                cdebug=False,
                use1=None):
    """
    Args:
        genes_df: Data frame holding genes.GC table
                    must include cols locusId, scaffoldId, and begin (genes)

        Length of below 4 objects is nAllStrainsCentral
        all_used_locId (pandas Series): all the locusIds from all_df_used
        all_used_f (pandas Series): all the f values from all_df_used (float)
                                    fractional insertion values.

        exp_used_strains (pandas Series): with counts for the current set.indexname 
                                 with strainsUsed value true (0.1<f<0.9) [countCond]
        t0_used (pandas Series): with t0 counts for each strain [countT0]
        strainsUsed_central_insert pandas Series(list<bool>): whose length is Trues in strainsUsed
                        equivalent index to strainsUsed True values


        Length of this object is nGenesUsed 
        genesUsed: list<locusId> where each locusId is a string 

        genesUsed12 (list<str>): list of locusIds that have both high f (>0.5) and low f (<0.5)
                    insertions with enough abundance of insertions on both sides
        minGenesPerScaffold: int
        set_index_name: name of current set and index name from all.pool
        
        use1: boolean list for the all_df_used with 0.1 < f <0.5 is True, otherwise false,
                Length is nAllStrainsCentralGoodGenes


        # other arguments are passed on to AvgStrainFitness()
        # base_se -- likely amount of error in excess of that given by variation within fitness values
        # 	for strains in a gene, due to erorrs in normalization or bias in the estimator
        #
        # Returns a data frame with a row for each gene in genesUsed. It includes
        # locusId,
        # fit (unnormalized), fitnorm (normalized),
        # fit1 or fit2 for 1st- or 2nd-half (unnormalized, may be NA),
        # fitnorm1 or fitnorm2 for normalized versions,
        # se (estimated standard error of measurement), and t (the test statistic),
        # as well as some other values from AvgStrainFitness(), notably sdNaive,
        # which is a different (best-case) estimate of the standard error.


    Returns:
        main_df (pandas DataFrame): Contains cols:
            locusId <str>: The locusId to which this row is associated.
            fit: fitRaw column normalized by Median 
            fitNaive (float): Median normalized log2 difference between tot0 and tot 
            fitnorm (float): Scaffold normalized fit scores (median and mode)
            fitRaw (float): Sum of weighted adjusted fitness scores divided by total weight. 
            n (int): Total number of strains in this locusId
            nEff (float ): The sum of the strain weights in these indeces/ max weight
            pseudovar (float): [ (median(abs(fit1 - fit2))^2)/Constant ] * (sdNaive/(median(sdNaive[genesUsed12])^2))
            sd (float): Standard Deviation computed fancy way
            sumsq (float): [Sum of the weighted square of the difference between adjusted fitness 
                            and fitRaw] divided by total weight.
            sdNaive (float): Standard Deviation computed in Naive way 
            tot (int ): The sum of the experiment reads over the locusID
            tot0 (int): The sum of the Time0s over the locusId
            se (float) Standard Error
            t: (float) t-statistic
            fit1 (float): For every locusId found in genesUsed12, we give the fit value of first_half_df
            fit2 (float): For every locusId found in genesUsed12, we give the fit value of second_half_df
            fitnorm1 (float): fit1 + fitnorm - fit
            fitnorm2 (float): fit2 + fitnorm - fit
            tot1 (int or nan): For every locusId found in genesUsed12, we give the tot value of first_half_df
            tot0_1 (int or nan): For every locusId found in genesUsed12, we give the tot0 value of first_half_df
            tot2 (int or nan): For every locusId found in genesUsed12, we give the tot value of second_half_df
            tot0_2 (int or nan): For every locusId found in genesUsed12, we give the tot0 value of second_half_df


    Description:
        This runs on every experiment (set + Index name).
        Our four primary inputs are the exps_used_strains (current experiment
        strains which are used), all_used_locId (the locusIds for all the used
        strains), all_used_f (the insertion fraction for all the strains),
        and the t0_used, which is the related t0 sums for the current experiment.
        First, we call Average Strain Fitness on all the strains given as an input;
        this gives us a dataframe (main_df) with values per Gene. The values it returns
        are 
            fit: fitRaw column normalized by Median 
            fitNaive (float): Median normalized log2 difference between tot0 and tot 
            locusId <str>: The locusId to which this row is associated.
            fitRaw (float): Sum of weighted adjusted fitness scores divided by total weight. 
            sd (float): Standard Deviation computed fancy way
            sumsq (float): [Sum of the weighted square of the difference between adjusted fitness 
                            and fitRaw] divided by total weight.
            sdNaive (float): Standard Deviation computed in Naive way 
            n (int): Total number of strains in this locusId
            nEff (float ): The sum of the strain weights in these indeces/ max weight
            tot (int ): The sum of the experiment reads over the locusID
            tot0 (int): The sum of the Time0s over the locusId
        Then we normalize the 'fit' values using the function NormalizeByScaffold,
        which simply adds a column to our dataframe, the column is called
        'fitnorm', for 'Normalized Fitness'.
        Then we call Average Strain Fitness twice again. Once for the whole set of 
        gene insertions, once for the insertions within .1<f<.5, and once for .5<f<.9. 
        The num rows of first_half_df and second_half_df (called for .1<f<.5 and .5<f<.9) is nGenesUsed12, 
        which is the total number of genes that have enough insertions on both sides of f.
        We use these two dataframes (first_half_df and second_half_df) to get statistical information
        on how balanced the insertions are regarding gene Fitness (pseudovar, se, t).
        The new dataframe columns we compute are 
        fit1, fit2, fitnorm1, fitnorm2, tot1, tot0_1, tot2, tot0_2, pseudovar, se, t
        fit1 through tot0_2 are just the computed fitness scores for the 1st and 
        second halves respectively, except for fitnorm1 and fitnorm2 are computed
        like this:
            fit1 + fitnorm - fit
        Where fitnorm and fit come from the main_df dataframe, and fit1 are the fit
        scores for first_half_df. fitnorm2 is computed similarly just with fit2 instead of
        fit1.
        When it comes to the reasoning behind computing 'pseudovar', 'se' and 't', 
        things get complicated. Explained separately. 
        We return main_df, which has the same number of rows as nGenesUsed, with 
        the values associated with first_half or second_half insertions
        included, but the ones with no locusId associated are nan. In other words,
        for the columns with '1' or '2' at the end of their name, they are
        only present for locusIds that have abundant enough insertions on
        both halves of the genes, and when there is a gene in main_df
        that doesn't pass that threshold, they don't have any values. Thus,
        they only have real values where the genes have good two sided
        abundance, and otherwise their values are nan.

    """

    
    main_df = AvgStrainFitness(exp_used_strains, 
                               t0_used, 
                               all_used_locId,
                               mini_debug=1,
                               current_experiment_name=set_index_name,
                               run_typ="main_df",
                               debug=False)
    
    main_df['fitnorm'] = NormalizeByScaffold(main_df['fit'], main_df['locusId'],
                                             genes_df, minToUse=minGenesPerScaffold,
                                             cdebug=False)

    strainsUsed_first_half = [bool(all_used_f.iat[i] < 0.5 and all_used_locId.iat[i] in genesUsed12) \
                    for i in range(len(all_used_locId))]

    # num rows should be len(genesUsed12)
    first_half_df = AvgStrainFitness(exp_used_strains[strainsUsed_first_half], 
                               t0_used[strainsUsed_first_half], 
                               all_used_locId[strainsUsed_first_half],
                               mini_debug=1,
                               current_experiment_name=set_index_name,
                               run_typ="first_half_df")

    
    strainsUsed_second_half = [bool(all_used_f.iat[i] >= 0.5 and all_used_locId.iat[i] in genesUsed12) \
                    for i in range(len(all_used_locId))]
    # num rows is equal to num rows of first_half_df = len(genesUsed12)
    second_half_df = AvgStrainFitness(exp_used_strains[strainsUsed_second_half], 
                               t0_used[strainsUsed_second_half], 
                               all_used_locId[strainsUsed_second_half],
                               mini_debug=1,
                               current_experiment_name=set_index_name,
                               run_typ="second_half_df")

    del strainsUsed_first_half, strainsUsed_second_half
    
    if cdebug:
        #DEBUG
        main_df.to_csv("tmp/Fpy_main_df.tsv", sep="\t")
        first_half_df.to_csv("tmp/Fpy_first_half_df.tsv", sep="\t")
        second_half_df.to_csv("tmp/Fpy_second_half_df.tsv", sep="\t")

    # why do we need the indexes to match?
    for i in range(first_half_df.shape[0]):
        if first_half_df['locusId'].iat[i] != second_half_df['locusId'].iat[i]:
            raise Exception(f"Non-matching locusId: {first_half_df['locusId'].iat[i]}"
                            f" != {second_half_df['locusId'].iat[i]}, at index {i}")

    matched_ixs = py_match(list(main_df['locusId']), list(first_half_df['locusId'])) 


    main_df['fit1'] = pd.Series([first_half_df['fit'].iloc[x] if x is not np.nan else np.nan for x in matched_ixs ])
    main_df['fit2'] = pd.Series([second_half_df['fit'].iloc[x] if x is not np.nan else np.nan for x in matched_ixs])
    # np.nan + integer = np.nan
    main_df['fitnorm1'] = main_df['fit1'] + main_df['fitnorm'] - main_df['fit']
    main_df['fitnorm2'] = main_df['fit2'] + main_df['fitnorm'] - main_df['fit']
    main_df['tot1'] = pd.Series(
                [first_half_df['tot'].iloc[x] if x is not np.nan else np.nan for x in matched_ixs])
    main_df['tot0_1'] = pd.Series(
                [first_half_df['tot0'].iloc[x] if x is not np.nan else np.nan for x in matched_ixs])
    main_df['tot2'] = pd.Series(
                [second_half_df['tot'].iloc[x] if x is not np.nan else np.nan for x in matched_ixs])
    main_df['tot0_2'] = pd.Series(
                [second_half_df['tot0'].iloc[x] if x is not np.nan else np.nan for x in matched_ixs])

    """
    for low n, the estimated variance is driven by the overall variance, which can be estimated
    from the median difference between 1st and 2nd halves via the assumptions
    Var(fit) = Var((fit1+fit2)/2) ~= Var(fit1-fit2)/4
    median abs(normal variable) = qnorm(0.75) * sigma = 0.67 * sigma
    which leads to Var(fit) = Var(fit1-fit2)/4
    = sigma12**2/4 = median abs diff**2 / (qnorm(0.75)*2)**2
    The median difference is used because a few genes may have genuine biological differences
    between the fitness of the two halves.
    Furthermore, assuming that genes with more reads are less noisy, this
    pseudovariance should be rescaled based on sdNaive**2
    
    """

    if cdebug:
        print("Length of main_df's columns: " + str(len(main_df['fitRaw'])))

    # Statistics comparing first_half_df and second_half_df (first half and second half insertions)
    # pseudovar_std is a constant (scalar)
    pseudovar_std = (((main_df['fit1'] - main_df['fit2']).abs()).median()**2) / ((2*stats.norm.ppf(0.75))**2)
    main_df['pseudovar'] = pseudovar_std * (main_df['sdNaive'] / ((main_df['sdNaive'][main_df['fit1'].notnull()]).median()**2) )
    # given the variable weighting in sumsq, it is not intuitive that the degrees of freedom is still n-1
    # however, this is the result given the assumption that the weighting is the inverse of the variance
    est_var = (main_df['pseudovar'] + main_df['sumsq'])/main_df['n']
    main_df['se'] = est_var.apply(np.sqrt)
    # paralmax_series
    paralmax_series = pd.Series([max(main_df['sdNaive'].iat[i]**2, est_var.iat[i]) for i in range(len(main_df['sdNaive']))])
    # base_se is an input to this function, normally 0.1
    main_df['t'] = main_df['fitnorm']/(base_se**2 + paralmax_series).apply(np.sqrt)

    return main_df





def AvgStrainFitness(exp_used_strains, 
                    t0_used, 
                    all_used_locId,
                    current_experiment_name=None,
		    minStrainT0 = 4, minGeneT0 = 40,
                    minGeneFactorNStrains=3,
                    strainFitAdjust=0,
		    maxWeight = 20,
		    debug=False,
                    mini_debug=0,
                    run_typ=None):

    """

    Args:
        exp_used_strains (Pandas Series <int>): counts at the 
                    end of the experiment condition.
                    Comes from all_df, only counts that have genes. Same length as 
                    t0_used (Reads for this experiment name)
                    Total length is nAllStrainsCentralGoodGenes* 
        t0_used (Pandas Series <int>): counts for Time0 for each used strain
        all_used_locId (Pandas Series <locusId (str)>): total locusIds of 
                                        all_df - the same for every time 
                                        this function is run. Same length as above two 
                                        variables (exp_used_strains, t0_used)
                                        What if no locusId exists for strain?
        minStrainT0: int
        minGeneT0: int
        maxWeight: int 
		 # maxWeight of N corresponds to having N reads on each side
                 #     (if perfectly balanced); use 0 for even weighting
		 # 20 on each side corresponds to a standard error of ~0.5; keep maxWeight low because outlier strains
		 # often have higher weights otherwise.

        current_experiment_name (str): Name of experiment (set-index), that
                                        we are currently analyzing
        run_typ (str): Debugging which part of GeneFitness are we running?
                        Fixed options: 'main_df', 'df_1', 'df_2'

    Returns:
        fitness_df (pandas DataFrame): with cols
            fit: fitRaw column normalized by Median 
            fitNaive (float): Median normalized log2 difference between tot0 and tot 
            locusId <str>: The locusId to which this row is associated.
            fitRaw (float): Sum of weighted adjusted fitness scores divided by total weight. 
            sd (float): Standard Deviation computed fancy way
            sumsq (float): [Sum of the weighted square of the difference between adjusted fitness 
                            and fitRaw] divided by total weight.
            sdNaive (float): Standard Deviation computed in Naive way 
            n (int): Total number of strains in this locusId
            nEff (float ): The sum of the strain weights in these indeces/ max weight
            tot (int ): The sum of the experiment reads over the locusID
            tot0 (int): The sum of the Time0s over the locusId
        
        * The length of the columns should be equal to the number of unique values
        in all_used_locId[strainsUsed_short] = nGenesUsed

    
    # If genesUsed (as a list of locusId) and strainsUsed_short (as boolean vector) are provided,
    # then considers only those strains & genes; minimum requirements.

    Description:
        We get strain fitness values for an experiment
    """

    if mini_debug > 0:
        print(f"Running AverageStrainFitness on {current_experiment_name} ({run_typ})")

    # crt_all... and crt_t0... contain integers, all_used_locId is str (locusId)
    if (len(exp_used_strains) < 1 or 
            len(exp_used_strains) != len(t0_used) or 
            len(exp_used_strains) != len(all_used_locId)):
        raise Exception("None or misaligned input data:\n"
                f"exp_used_strains len: {len(exp_used_strains)}\n"
                f"t0_used len: {len(t0_used)}\n"
                f"all_used_locId len: {len(all_used_locId)}.\n"
                "All lengths must be equal and above 1."
                )

    # Check if accurate?
    crt_t0_name = t0_used.name

    if debug:
        logging.info("Number of unique values: " + str(len(all_used_locId.unique())))
        logging.info("Above number is equivalent to number of rows in final DFs")
        t0_used.to_csv("tmp/py_t0_used_A1.tsv", sep="\t")
        exp_used_strains.to_csv("tmp/py_exp_used_strains_A1.tsv", sep="\t")
        all_used_locId.to_csv("tmp/py_all_used_locId.tsv", sep="\t")


    # this won't happen because the sum of t0's is always above 0 (in func  
    # gene_strain_fit_func. Just a double check
    if sum(t0_used) != 0:
        # readratio < 1 -> loss of fitness, readratio > 1 -> gain fitness.
        readratio = exp_used_strains.sum()/t0_used.sum()
        print(f'readratio: {readratio}')
    else:
        raise Exception(f"No positive t0 values for this set/index value: {current_experiment_name}\n"
                         " Cannot get readratio (Division by 0).")

    
    # This is where we get strain Fitness (pandas Series) - median normalized log2 ratios between
    # strain and T0 sums. pandas Series whose length is nAllStrainsCentralGoodGenes(*)
    strainFit = getStrainFit(exp_used_strains, t0_used, readratio, debug=True)

    # Per-strain "smart" pseudocount to give a less biased per-strain fitness estimate.
    # This is the expected reads ratio, given data for the gene as a whole
    # Arguably, this should be weighted by T0 reads, but right now it isn't.
    # Also, do not do if we have just 1 or 2 strains, as it would just amplify noise


    # strainPseudoCount is a pandas Series, length is nAllStrainsCentralGoodGenes*
    # Uses medians of strainFit over locusIds and computes a measure on that using
    # the read ratio. These counts are used to stabilize the strainFit values
    strainPseudoCount = getStrainPseudoCount(all_used_locId, 
                            strainFit, readratio, 
                            current_experiment_name=current_experiment_name,
                            minGeneFactorNStrains=minGeneFactorNStrains, 
                            debug_print_bool=True)
   
    # We create strainFit_adjusted ------->

    # length of the following pandas Series is nAllStrainsCentralGoodGenes*
    # Remember no values in strainPseudoCount can be <= 0, so 
    # 1/strainPseudocount.sqrt is fine. ( "PC" -> "PseudoCount" )
    expPC = strainPseudoCount.apply(np.sqrt)
    t0PC = 1/expPC # (This applies element-wise in the series)
    strainFit_adjusted = (expPC + exp_used_strains).apply(np.log2) \
                        - (t0PC + t0_used).apply(np.log2) \
                        - strainFitAdjust
    del expPC, t0PC, strainPseudoCount 


    # strain Standard Deviation (list of floats) (We add 1 to avoid division by zero error)
    strainSD = ( (1/(1 + t0_used) + 1/(1 + exp_used_strains)).apply(np.sqrt) )/np.log(2)
    
    # Getting strainWeight
    # "use harmonic mean for weighting; add as small number to allow maxWeight = 0."
    strainWeightUnbounded = 2/( (1/(1 + t0_used)) + (1/(1 + exp_used_strains)) )
    # We make sure all the weights are lower than the max weight (bounding)
    # here we take any value in strainWeightUnbounded that's above maxWeight
    # and turn it into maxWeight. e.g. maxWeight = 20, [13, 34, 45, 10] -> [13, 20, 20, 10]
    strainWeight = 0.5 + strainWeightUnbounded.combine(maxWeight, min) 

    # We debug how many values were taken from above maxWeight to maxWeight
    num_max_weight = list(strainWeight).count(maxWeight)
    print(f"{num_max_weight} of the {len(strainWeight)} strainWeights surpassed" \
          f" the max weight of {maxWeight} and were bounded.")

    # Just debugging
    if mini_debug > 1:
        # Vars to output: strainSD, strainWeight, strainFit_adjusted, strainFit,
        # abs(strainFit_adjusted - strainFit), t0PseudoCount, condPseudoCount,
        # strainPseudoCount, geneFit1
        for x in [["strainSD.tsvsrs", strainSD],
                  ["strainWeight.tsvsrs", strainWeight],
                  ["strainFit_adjusted.tsvsrs", strainFit_adjusted],
                  ["strainFit.tsvsrs", strainFit],
                  ["strainFitDifference.tsvsrs", strainFit_adjusted - strainFit]
                  ]:
            x[1].to_csv("tmp/" + x[0], sep="\t")
        raise Exception("mini_debug>1 so stopping after printing vars")



    # We create a list of values for each of the following derived floats/ints (except locusId, which is str)
    fitness_d = {
             "fitRaw": [],
             "sd": [],
             "sumsq": [],
             "sdNaive": [],
             "n": [],
             "nEff": [],
             "tot": [],
             "tot0": [],
             "locusId": []
            }

    # This groups our row number from all_df by the same locusIds
    t0_index_groups = t0_used.groupby(by=all_used_locId).groups 
    for locId, ix_lbls in t0_index_groups.items():
        # Each key is a different locusId
        # n will be the length of 'ix_lbls' - 
        # which is the number of times the locusId appears in all_used_locId
        # the values in ix_lbls are the indexes of where this locusId appears
        crt_result_d = sub_avg_fitness_func(list(ix_lbls), strainWeight, strainFit_adjusted,
                               exp_used_strains, t0_used,
                               strainSD, locId, cdebug=False)
        for keyy, valu in crt_result_d.items():
            fitness_d[keyy].append(valu)


    # fitness_l is a list that is populated with elements that are Series of 
    # dicts with values as numbers. We create a dataframe with all of them.
    fitness_df = pd.DataFrame.from_dict(fitness_d)
    fitness_df.sort_values(by=['locusId'], inplace=True)
    fitness_df['fit'] = mednorm(fitness_df['fitRaw'])
    fitness_df['fitNaive'] = mednorm(np.log2(1+fitness_df['tot']) - np.log2(1 + fitness_df['tot0']))

    return fitness_df



def getStrainFit(exp_used_strains, t0_used, readratio,
                 debug=False):
    """
    Args:
        crt... : pandas series with integers. Length is nAllStrainsCentralGoodGenes
        readratio: float
    returns:
        strainFit (pandas series): of floats length is the same as len(exp_used_strains) =
                                                                   len(t0_used)
                                    Normalized log 2 difference between values and time0s
                                    Length is nAllStrainsCentralGoodGenes(*)
                                    (Could be shortened due to having only strains in genesUsed12)
                                    Index are original row number from all_df 
    
    Description:
        We take the current values, add the readratio (to eliminate possible log2(0)) then take the log2 values
            then normalize by the median, returning a pandas series (vector) of normalized log2 ratios.
        Why do we do median normalization while taking strain fitness?

    use sqrt(readratio), or its inverse, instead of 1, so that the expectation
    is about the same regardless of how well sampled the strain or gene is
    """
    # use sqrt(readratio), or its inverse, instead of 1, so that the expectation
    # is about the same regardless of how well sampled the strain or gene is
    exp_1 = exp_used_strains + np.sqrt(readratio)
    t0_1 = t0_used + 1/np.sqrt(readratio)
    exp_2 = exp_1.apply(np.log2)
    t0_2 = t0_1.apply(np.log2)
    strainFit = mednorm(exp_2 - t0_2)

    if debug:
        fn = os.path.join('tmp','strainfit.tsv')
        strainFit.to_csv(fn, sep='\t')
        print(f"strainFit original to file at {fn}")
    return strainFit


def getGeneFitMedians(strainFit, good_strainLocusIds, current_experiment_name=None, print_op=None):
    """
    
    Both of the following inputs have the same length
    strainFit (pandas Series <float>): floats with all_df row nums as index labels . It's the 
                                normalized log2 difference between actual counts and t0 counts.
                                Length is nAllStrainsCentralGoodGenes(*)
                                (* Could be shortened due to having only strains in genesUsed12)
    good_strainLocusIds (pandas Series <locusId (str)>): related locusIds to above floats
                    Length is nAllStrainsCentralGoodGenes(*)
                    (* Could be shortened due to having only strains in genesUsed12)
    current_experiment_name (str): Experiment name

    Returns: 
        geneFitMedians (pandas Series <float>): 
                                    Its length will be the number of unique
                                    locus Ids in good_strainLocusIds,
                                    which could be the number of genes
                                    in genesUsed or genesUsed12 depending
                                    on if the run is main_df, or df_1/df_2
                                    Index is the locusId, so it's locusId -> number.
                                    Thus we can access its values using locusIds 


    Description:
        We group the values of strainFit by their locusIds
            in good_strainLocusIds, and calculate the median of each group
            Then we normalize by the median, which means we subtract
            the total median from each value.
            We return this pandas Series.
            Its length will be the number of unique
            locus Ids in good_strainLocusIds,
            which could be the number of genes
            in genesUsed or genesUsed12 depending
            on if the run is main_df, or df_1/df_2
            Index is the locusId 
    """

    #logging.info(f"Getting geneFitMedians for {strainFit.name}")
    if current_experiment_name is None:
        current_experiment_name = "placeholder" 
    new_df = pd.DataFrame.from_dict({
            current_experiment_name : strainFit,
            'locusId': good_strainLocusIds
    })
    
    # We get the medians over all the strains with the same locusId
    # The index will be the locusIds
    medians_df = new_df.groupby(by='locusId').median()

    geneFitMedians = mednorm(medians_df[current_experiment_name])


    if print_op is not None:
        geneFitMedians.to_csv(print_op, sep='\t') 

    return geneFitMedians


def getStrainPseudoCount(all_used_locusId, strainFit, readratio, 
                        current_experiment_name="placeholder",
                         minGeneFactorNStrains=3, 
                         debug_print_bool=False):
    """
    Args:

        all_used_locusId (Pandas Series <locusId (str)>): which locus the strain is associated with 
                                                     from all_df_subset['locusId'], and applied
                                                     boolean list 'strainsUsed' to it.
                    Length is nAllStrainsCentralGoodGenes(*)
                    (* Could be shortened due to having only strains in genesUsed12)
        minGeneFactorNStrains: int - Threshold test for number of times genes Inserted in.
        strainFit (pandas Series <float>): length is same as all_used_locusId
        readratio (float): (sum of counts/ sum of t0 for this sample index)

    Returns:
        strainPseudoCount (pandas Series <float>): We get the medians over locusIds
                                            for the original strainFitness values,
                                            then we get the number of strains
                                            per locusId (i.e. how many strains
                                            were inserted into a given locusId).
                                            Then for each locusId, we check if
                                            it passes the threshold 'minGeneFactorNStrains',
                                            if it does, we add a certain value,
                                            otherwise, we add the readratio (exp/cntrl)


    Created vars:
        geneFitMedians (pandas Series): median-normalized medians of locusIds over values from
                                  StrainFit.
                                    Its length will be the number of unique
                                    locus Ids in all_used_locId,
                                    which could be the number of genes
                                    in genesUsed or genesUsed12 depending
                                    on if the run is main_df, or df_1/df_2
                                    Index is the locusId, so it's locusId -> number.
                                    Thus we can access its values using locusIds 
    """

    # pd.Series length of nGenesUsed* - normalized medians of 
    # strainFit over locusIds. 
    # index is locusIds. Essentially a dict from locusId to medians over strain Fitness values.
    geneFitMedians = getGeneFitMedians(strainFit, all_used_locusId, current_experiment_name) 

    # This 'table; is unique locus Ids pointing to the number of times they occur
    locusId2TimesSeen_d = py_table(all_used_locusId) 
    

    strainPseudoCount = []
    for locId in all_used_locusId.values:
        if locusId2TimesSeen_d[locId] >= minGeneFactorNStrains:
            # remember readratio is sum(experiment)/sum(time0s)
            strainPseudoCount.append((2**geneFitMedians[locId])*readratio)
        else:
            # Why this?
            strainPseudoCount.append(readratio)

    strainPseudoCountSeries = pd.Series(strainPseudoCount, index=all_used_locusId.index)

    if debug_print_bool:
        strainPseudoCountSeries.to_csv('tmp/py_strainPseudoCount.tsvsrs')
        print("Wrote strainPseudoCount Series to tmp/py_strainPseudoCount.tsvsrs")


    return strainPseudoCountSeries


def get_strainFitWeight(condPseudoCount, crt_all_series_hg_su,
                        t0PseudoCount, crt_t0_series_hg_su,
                        strainFitAdjust = 0
                        ):
    """
condPseudoCount, exp_used_strains,
                                            t0PseudoCount, t0_used)


    Args:


        # length of the following series is nAllStrainsCentralGoodGenes*


        condPseudoCount:
        t0PseudoCount: 
        strainFitAdjust: (int)

    Returns:
        strainFit_weight (pandas Series) with index labels fitting crt_all_series...
    """
    '''
    strainFit_weight = []
    for i in range(len(condPseudoCount)):
        strainFit_weight.append(math.log2(condPseudoCount[i] + crt_all_series_hg_su.iat[i]) \
                                - math.log2(t0PseudoCount[i] + crt_t0_series_hg_su.iat[i]) \
                                - strainFitAdjust)

    return pd.Series(data=strainFit_weight, index=crt_all_series_hg_su.index)
    '''
    return None


def sub_avg_fitness_func(ix_l, strainWeight, strainFit_adjusted,
                               exp_used_strains, t0_used,
                               strainSD, locusIdstr, cdebug=False):
    """
    Args:
        ix_l (int): list<int> of index labels (from grouped locusIds 
                    in all_used_locId) from all_df where this locusId 
                    (locusIdstr) is found. So if locusId 'mygene' was
                    found at locations 134, 249, 321, etc. in the 
                    indexes of all_used_locId, then these are the
                    same as in the other series, and can be used on
                    them.

        strainWeight (pandas Series <float>): each element has a maximum value of 'maxWeight', 
                                    which normally equals 20,
                                    other elements have values which are computed 
                                    in AvgStrainFitness func. All positive values.
                                    Length of this is  nAllStrainsCentralGoodGenes*
        strainFit_adjusted pandas Series <float>:  Same index as strainWeight
                                    Length of this is  nAllStrainsCentralGoodGenes*
        exp_used_strains (pandas series <int>): The used strain-> reads from all_df
                                    Length of this is  nAllStrainsCentralGoodGenes*
        t0_used (pandas series <int>): The strain -> t0sum from t0tot 
                                    Length of this is  nAllStrainsCentralGoodGenes*
        strainSD (pandas Series <float>): 
                                    Length of this is  nAllStrainsCentralGoodGenes*
        locusIdstr: (str)
    Returns:
           ret_d: dict with the following keys: (single values, not series)
                fitRaw (float): Sum of weighted adjusted fitness scores divided by total weight. 
                sd (float): Standard Deviation computed fancy way
                sumsq (float): [Sum of the weighted square of the difference between adjusted fitness 
                                and fitRaw] divided by total weight.
                sdNaive (float): Standard Deviation computed in Naive way 
                n (int): Total number of strains in this locusId
                nEff (float ): The sum of the strain weights in these indeces/ max weight
                tot (int ): The sum of the experiment reads over the locusID
                tot0 (int): The sum of the Time0s over the locusId
    Description:
        What are the strainWeights? 
        We get the sum of the weights of all the strains
        
    """

    total_weight = strainWeight[ix_l].sum()
    fitRaw = (strainWeight[ix_l] * strainFit_adjusted[ix_l]).sum()/total_weight
    tot = exp_used_strains[ix_l].sum()
    tot0 = t0_used[ix_l].sum()
    sd = math.sqrt( ( (strainWeight[ix_l]**2) * (strainSD[ix_l]) ).sum()/total_weight)
    sumsq = ( strainWeight[ix_l] * ((strainFit_adjusted[ix_l] - fitRaw)**2) ).sum()/total_weight
    
    # 'high-N estimate of the noise in the log2 ratio of fitNaive'
    # 'But sdNaive is actually pretty accurate for small n -- e.g.'
    # 'simulations with E=10 on each side gave slightly light tails'
    # '(r.m.s.(z) = 0.94).'

    sdNaive = np.sqrt(  (1/(1+tot)) + (1/(1+tot0)) )/np.log(2)
    
    nEff = total_weight/(strainWeight[ix_l].max())
    ret_d = {
             "fitRaw": fitRaw,
             "sd": sd,
             "sumsq": sumsq,
             "sdNaive": sdNaive,
             "n":len(ix_l),
             "nEff": nEff,
             "tot": tot,
             "tot0": tot0,
             "locusId": locusIdstr 
            }

    return ret_d


def StrainFitness(all_cix_series,
                all_cntrl_sum,
                debug_print=False):
    """
    Args:
        all_cix_series (pandas Series): The current experiment name column of values from all_df_used 
                                        length = nAllStrainsCentralGoodGenes
        all_cntrl_sum (pandas Series): The sum of the current control values without the current index; 
                                        Is a pandas series the same length as all_cix series,
                                        but with the sum of the other control values
                                        length = nAllStrainsCentralGoodGenes
        debug_print (bool): Decides whether to print out this function's results and stop
                            the program

    Returns:
        fit: pandas Series (float) with a computation applied to values
            Same length as inputs: nAllStrainsCentralGoodGenes
        se: pandas Series (float) with computations applied to values
            Same length as inputs: nAllStrainsCentralGoodGenes


    Description:
        fit: Median-Normalized log2 difference between Current experiment and the time0s
        se: Standard Error of the values 
        "
        # simple log-ratio with pseudocount (of 1) and normalized so each scaffold has a median of 0
        # note is *not* normalized except to set the total median to 0
        "
    """

    sf_fit = mednorm( (1+all_cix_series).apply(np.log2) - (1 + all_cntrl_sum).apply(np.log2) )
    sf_se = (1/(1 + all_cix_series) + 1/(1 + all_cntrl_sum)).apply(math.sqrt)/ np.log(2)



    return {
            "fit": sf_fit,
            "se": sf_se
            }



def NormalizeByScaffold(fitValues, locusIds, genes_df, window=251, minToUse=10,
                        subtract_mode=True, cdebug=False):
    """
    Args:
        fitValues (pandas Series): main_df['fit'] from AvgStrainFitness,
                                length is  nGenesUsed
        locusIds (pandas Series): main_df['locusIds'] from AvgStrainFitness
                                length the same as above (nGenesUsed)
        genes_df: Data Frame created from genes.GC
                                length is more than nGenesUsed, it is nTotalGenes
        window (int): window size for smoothing by medians. Must be odd, default 251. For scaffolds
                      with fewer genes than this, just uses the median.
        minToUse (int): If a scaffold has too few genes, cannot correct for possible DNA extraction
                        bias so we need to remove data for that gene (i.e., returns NA for them)
        subtract_mode (bool): Should we also subtract the mode 

    Returns:
        fitValues (pandas Series of floats): Rolling median and mode normalized over
        scaffoldIds. Length is the same as fitValues (=nGenesUsed)

    Description:
        We create the column 'fitnorm' for main_df in GeneFitness.
        How:
        We take the series of locusIds from main_df (which are unique),
        and find their locations within the series genes_df['locusId'] which are also
        unique. Then we take the subset of the dataframe genes_df which match
        to those locations and take its scaffoldId, begin values and locusIds and
        combine them with the fitness values from AvgStrainFitness, and create
        a dataframe called tmp_df.
        Then we group values by scaffoldIds (which are not unique), and get
        the rows of tmp_df which are associated with that scaffoldId.
        We check that the total number of rows associated with that scaffoldId
        pass the threshold 'minToUse'. If it doesn't, we remove the fitness
        values for those rows. If it does we continue to the next part.
        Now we take the median of the rows associated with that scaffoldId
        and subtract it from them (median normalization on a subset of the
        rows).
        Then, if the number of rows associated with this scaffoldId pass
        the window threshold, we take the running median with window size
        251 of the sorted genes (A running median looks backwards and 
        forwards the same amount and computes the median using half 
        the window after and half before, so in this case 125 after and 
        125 before), and then we subtract that median from those values.
        Then we also normalize by the mode, and a simple way to 
        estimate the mode if there aren't repeating values is through
        the gaussian kernel density estimation function:
        //rmflight.github.io/post/finding-modes-using-kernel-density-estimates/

    """
    
    # Initialize the new set of values
    NormalizedFitValues = fitValues.copy(deep=True)

    if cdebug:
        print(f"locusIds from dataframe: {len(list(locusIds))}",
              f"locusIds from genes_df: {len(list(genes_df['locusId']))}")

    # We find indexes of locusIds within the genes' dataframe, locusId, column
    cmatch = py_match(list(locusIds), list(genes_df['locusId']))
    if None in cmatch:
        raise Exception("Fitness data for loci not in genes_df")

    matched_genes_df = genes_df.iloc[cmatch]
    matched_genes_df.reset_index(inplace=True)

    tmp_df = pd.DataFrame.from_dict({
                        "fit": NormalizedFitValues,
                        "scaffoldId": matched_genes_df['scaffoldId'],
                        "begin": matched_genes_df['begin'],
                        "locusId": matched_genes_df['locusId']
                        })

    perScaffoldRows = tmp_df.groupby(by='scaffoldId').indices

    '''
    # py_split returns groupings of numerical iloc NormalizedFitValues grouped by the scaffoldIds
    perScaffoldRows = py_split(pd.Series(list(range(0, len(NormalizedFitValues)))), 
                               list(matched_genes_df['scaffoldId']), 
                               typ='indices')
    '''
    
    # scaffoldId is str, rows is a list of ints (indeces for iloc) (iterable(?))
    for scaffoldId, rows in perScaffoldRows.items():
        if cdebug:
            print(f"Working on scaffoldId {scaffoldId} within NormalizeByScaffold")
        if len(rows) < minToUse:
            if cdebug:
                print("Removing " + str(len(rows)) + " NormalizedFitValues for " + scaffoldId)
            NormalizedFitValues[rows] = np.nan 
        else:
            med = NormalizedFitValues[rows].median()
            if cdebug:
                print("Subtracting median for " + scaffoldId + " " + str(med))
            NormalizedFitValues[rows] = NormalizedFitValues[rows] - med

            if len(rows) >= window:
                if cdebug:
                    print("Num rows: {len(rows)} passed window {window}")
                # srtd_begs is a list of indexes for the sorted begin NormalizedFitValues for this scaffold
                srtd_begs = py_order(tmp_df.iloc[rows]['begin'])
                 
                # center=True converts rolling to running (centered median)
                runmds = NormalizedFitValues[rows[srtd_begs]].rolling(window, center=True).median()
                # Here we set the nan values to the first and last valid indexes
                first_non_na_index = runmds.first_valid_index()
                last_non_na_index = runmds.last_valid_index()
                first_non_na_value = runmds[first_non_na_index]
                last_non_na_value = runmds[last_non_na_index]
                runmds.update(pd.Series([first_non_na_value]*first_non_na_index, 
                              index=list(range(first_non_na_index))))
                runmds.update(pd.Series([last_non_na_value]*(len(runmds) - last_non_na_index), 
                              index=list(range(last_non_na_index, len(runmds)))))

                if cdebug:
                    print("Subtract smoothed median for " + scaffoldId + ". max effect is " + \
                         f"{max(runmds) - min(runmds)}")
                # Changing NormalizedFitValues of the pandas series by the running median
                NormalizedFitValues[rows[srtd_begs]] = NormalizedFitValues[rows[srtd_begs]] - runmds[srtd_begs]
    
                if subtract_mode:
                    # We use density function to estimate mode 
                    dns = stats.gaussian_kde(NormalizedFitValues[rows].dropna())
                    cmax, cmin = NormalizedFitValues[rows].max(), NormalizedFitValues[rows].min()
                    estimate_x = [cmin + (((cmax - cmin)/512)*i) for i in range(512)]
                    estimate_y = dns.evaluate(estimate_x)
                    mode = estimate_x[list(estimate_y).index(max(estimate_y))]
                    if cdebug:
                        print("Subtract mode for " + scaffoldId + " which is at " + str(mode))
                    NormalizedFitValues[rows] = NormalizedFitValues[rows] - mode

    return NormalizedFitValues


def mednorm(pd_series):
    # Desc: Takes pandas series and returns pandas series with median subtracted
    # Note if you run this once, that's as much as you'd need to run it,
    # because the median becomes 0, and running it again won't change
    # the values at all.
    return pd_series - pd_series.median() 
