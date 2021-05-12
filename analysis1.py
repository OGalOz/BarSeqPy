
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
               genesUsed, genesUsed12, strainsUsed, central_insert_bool_list, 
               minGenesPerScaffold=10, meta_ix=7,debug=False, nDebug_cols=None):
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
        central_insert_bool_list (list<bool>): 

    Returns:
        GeneFitResults: (dict) set_index_names -> gene_strain_fit_result
            gene_strain_fit_result (dict):
                gene_fit: DataFrame, contains cols:
                    fit, fitNaive, fit1, fit2, fitnorm, fitnorm1, fitnorm2, fitRaw
                    locusId, n, nEff, pseudovar, sumsq, sd, sdNaive, se, t, tot1
                    tot1_0, tot2, tot2_0, tot, tot0
                strain_fit: pandas Series (float) with a computation applied to values
                strain_se: pandas Series (float) with a computation applied to values
    """

    # The bulk of the program occurs here: We start computing values
    GeneFitResults = {}
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

    print(f"Running through {num_ix_remaining}/{len(all_index_names)} indices")
    for set_index_name in all_index_names[:nSetIndexToRun]:
        print(f"Currently working on index {set_index_name}")
        
        start_time = time.time()
        if set_index_name is not None:
            # We choose the right column
            exp_used_strains = all_df_used[set_index_name]
            gene_strain_fit_result = gene_strain_fit_func(set_index_name, 
                                                          exps_df, exp_used_strains, 
                                                          genes_df, expsT0,
                                                          t0tot_used, 
                                                          genesUsed, genesUsed12, minGenesPerScaffold,
                                                          all_df_used,
                                                          use1)
            if gene_strain_fit_result is not None:
                GeneFitResults[set_index_name] = gene_strain_fit_result
            else:
                print(f"For index {set_index_name} result was None")

        end_time = time.time()
        num_ix_remaining -= 1
        print(f"{num_ix_remaining}/{len(all_index_names)} left to run through")
        print(f"Estimated time remaining: {((end_time-start_time)*num_ix_remaining)/60} minutes.")
        print(f"Current time: {datetime.now().strftime('%H:%M:%S')} PST.")

    # If there are no 
    if len(GeneFitResults.keys()) == 0:
        raise Exception("All comparisons failed.")

    if debug:
        print("passed GeneFitness section")

    return GeneFitResults



def gene_strain_fit_func(set_index_name, exps_df, exp_used_strains, 
                         genes_df, expsT0,
                         t0tot_used, 
                         genesUsed, genesUsed12, minGenesPerScaffold,
                         all_df_used, use1
                         ):
    """
    Description:
        This function is run for every single set_index_name in all_df, and that set_index_name
        is passed into this function as the first argument, 'set_index_name'. All other arguments
        are not changed at all when this function is called and are documented elsewhere. 
        Note that all_df_used is a subset
        of all_df (all.poolcount) in which the barcode was inserted within a gene and within the
        central 80% of the gene and the locusId is found in genesUsed. There may be other thresholds.
        t0tot_used has the exact same number of rows as all_df_used.

        Then the majority of the work of the function is done within
        creating the variable 'gene_fit' while calling the function 'GeneFitness'.

        What happens in this function?
            First we find if this value is part of a t0set.
            If not, we get the related t0 set.

        
    Args:
        set_index_name: (str) Name of set and index from all_df (all.poolcount file)

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
        all_df_central_inserts (Dataframe): The parts of all_df that corresponds to True in central_insert_bool_list
                                            Num rows is nAllStrainsCentral 
        use1: boolean list for the all_df_used with 0.1 < f <0.5 is True, otherwise false,
                Length is nAllStrainsCentralGoodGenes

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
                tot1_0, tot2, tot2_0, tot, tot0
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

    strain_fit_ret_d = StrainFitness(exp_used_strains, 
                      all_df_used[cntrl].sum(axis=1),
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
                                 with central_insert_bool_list value true (0.1<f<0.9) [countCond]
        t0_used (pandas Series): with t0 counts for each strain [countT0]
        strainsUsed_central_insert pandas Series(list<bool>): whose length is Trues in central_insert_bool_list
                        equivalent index to central_insert_bool_list True values


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

    Description:
        We call Average Strain Fitness 3 times. Once for the whole set of gene insertions,
            once for the insertions within .1<f<.5, and once for .5<f<.9. The num rows
            of df1 and df2 (called for .1<f<.5 and .5<f<.9) is nGenesUsed12, which is
            the total number of genes that have enough insertions on both sides of f.


    Returns:
        main_df (pandas DataFrame): Contains cols:
            locusId (str),
            fit (float): (unnormalized
            fitNaive (float):
            fit1 (float):
            fit2 (float):
            fitnorm (float):
            fitnorm1 (float)
            fitnorm2 (float)
            fitRaw (float)
            locusId (str)
            n (int)
            nEff (float)
            pseudovar (float)
            sumsq (float):
            sd (float)
            sdNaive (float)
            se (float) Standard Error
            t: (float) t-statistic
            tot1 (int or nan)
            tot1_0 (int or nan)
            tot2 (int or nan)
            tot2_0 (int or nan)
            tot (int or nan)
            tot0 (int or nan)
    """

    
    # Python code:
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

    strainsUsed_now = [bool(all_used_f.iat[i] < 0.5 and all_used_locId.iat[i] in genesUsed12) \
                    for i in range(len(all_used_locId))]
    # num rows should be len(genesUsed12)
    df_1 = AvgStrainFitness(exp_used_strains[strainsUsed_now], 
                               t0_used[strainsUsed_now], 
                               all_used_locId[strainsUsed_now],
                               mini_debug=1,
                               current_experiment_name=set_index_name,
                               run_typ="df_1")

    
    strainsUsed_now = [bool(all_used_f.iat[i] >= 0.5 and all_used_locId.iat[i] in genesUsed12) \
                    for i in range(len(all_used_locId))]
    # num rows is equal to df_1, should be len(genesUsed12)
    df_2 = AvgStrainFitness(exp_used_strains[strainsUsed_now], 
                               t0_used[strainsUsed_now], 
                               all_used_locId[strainsUsed_now],
                               mini_debug=1,
                               current_experiment_name=set_index_name,
                               run_typ="df_2")

    del strainsUsed_now
    
    if cdebug:
        #DEBUG
        main_df.to_csv("tmp/Fpy_main_df.tsv", sep="\t")
        df_1.to_csv("tmp/Fpy_df_1.tsv", sep="\t")
        df_2.to_csv("tmp/Fpy_df_2.tsv", sep="\t")
        #genesUsed12.to_csv("tmp/Fpy_genesUsed12.tsv", sep="\t")

    # why do we need the indexes to match?
    for i in range(len(df_1['locusId'])):
        if df_1['locusId'].iat[i] != df_2['locusId'].iat[i]:
            raise Exception(f"Non-matching locusId: {df_1['locusId'].iat[i]}"
                            f" != {df_2['locusId'].iat[i]}, at index {i}")

    # do we need one of these for df_2 as well? How are the locusIds listed?
    matched_ixs = py_match(list(main_df['locusId']), list(df_1['locusId'])) 
    if cdebug:
        debug_print(matched_ixs, 'matched_ixs')
        with open("tmp/py_matches.json", "w") as g:
            g.write(json.dumps(matched_ixs, indent=2))

    main_df['fit1'] = pd.Series([df_1['fit'].iloc[x] if x is not np.nan else np.nan for x in matched_ixs ])
    #main_df['fit1'].to_csv("tmp/COMPARE/py_fit1.tsv")
    main_df['fit2'] = pd.Series(
                [df_2['fit'].iloc[x] if x is not np.nan else np.nan for x in matched_ixs])
    #main_df['fit2'].to_csv("tmp/COMPARE/py_fit2.tsv")
    main_df['fitnorm1'] = main_df['fit1'] + (main_df['fitnorm'] - main_df['fit'])
    main_df['fitnorm2'] = main_df['fit2'] + (main_df['fitnorm'] - main_df['fit'])
    main_df['tot1'] = pd.Series(
                [df_1['tot'].iloc[x] if x is not np.nan else np.nan for x in matched_ixs])
    main_df['tot1_0'] = pd.Series(
                [df_1['tot0'].iloc[x] if x is not np.nan else np.nan for x in matched_ixs])
    main_df['tot2'] = pd.Series(
                [df_2['tot'].iloc[x] if x is not np.nan else np.nan for x in matched_ixs])
    main_df['tot2_0'] = pd.Series(
                [df_2['tot0'].iloc[x] if x is not np.nan else np.nan for x in matched_ixs])

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
    pseudovar_std = (((main_df['fit1'] - main_df['fit2']).abs()).median()**2) / ((2*stats.norm.ppf(0.75))**2)
    main_df['pseudovar'] = pseudovar_std * (main_df['sdNaive'] / ((main_df['sdNaive'][main_df['fit1'].notnull()]).median()**2) )
    # given the variable weighting in sumsq, it is not intuitive that the degrees of freedom is still n-1
    # however, this is the result given the assumption that the weighting is the inverse of the variance
    est_var = (main_df['pseudovar'] + main_df['sumsq'])/main_df['n']
    main_df['se'] = est_var.apply(math.sqrt)
    # paralmax_series
    paralmax_series = pd.Series([max(main_df['sdNaive'].iat[i]**2, est_var.iat[i]) for i in range(len(main_df['sdNaive']))])
    main_df['t'] = main_df['fitnorm']/(base_se**2 + paralmax_series).apply(math.sqrt)
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
            fitNaive (float): 
            fitRaw: list<float>
            locusId: list<str>
            n: list<int>
            nEff: list<float>
            sd: list<float>
            sumsq: list<float>
            sdNaive: list<float>
            tot: list<int>
            tot0: list<int>
        
        * The length of the columns should be equal to the number of unique values
        in all_used_locId[strainsUsed_short]

    
    # If genesUsed (as a list of locusId) and strainsUsed_short (as boolean vector) are provided,
    # then considers only those strains & genes; minimum requirements.

    Description:
        We take the subsets of the pandas Series that align with hasGene from all_df, 
            exp_used_strains_has_gene is the column of the index
            t0_used_has_gene is the sum of the related t0s
            all_used_locId is the column of locusId that's related.
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
    # 
    strainPseudoCount = getStrainPseudoCount(all_used_locId, 
                            strainFit, readratio, 
                            minGeneFactorNStrains=minGeneFactorNStrains, 
                            debug_print_bool=True)
   
    # We create strainFit_adjusted
    # length of the following pandas Series is nAllStrainsCentralGoodGenes*
    # Remember no values in strainPseudoCount can be 0, so 1/strainPseudocount.sqrt is fine 
    # PC -> PseudoCount
    expPC = strainPseudoCount.apply(np.sqrt)
    t0PC = 1/expPC # (This applies to every element in the series)

    # place holder for 'strain fit adjusted' values
    strainFit_adjusted = (expPC + exp_used_strains).apply(np.log2) \
                        - (t0PC + t0_used).apply(np.log2) \
                        - strainFitAdjust
    del expPC, t0PseudoCount, strainPseudoCount


    # strain Standard Deviation (list of floats) (We add 1 to avoid division by zero error)
    strainSD = ( (1/(1 + t0_used) + 1/(1 + exp_used_strains)).apply(np.sqrt) )/np.log(2)
    
    # Getting strainWeight
    # "use harmonic mean for weighting; add as small number to allow maxWeight = 0."
    s1 = 2/( 1/(1+t0_used) + 1/(1 + exp_used_strains) )
    strainWeight = s1.combine(maxWeight, min, 0)
    del s1
    num_max_weight = list(strainWeight).count(maxWeight)
    print(f"{num_max_weight} of the {len(strainWeight)} strainWeights surpassed" \
          f" the max weight of {maxWeight}")

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
    for k, v in t0_index_groups.items():
        # crt_result is a dict that matches with fitness_d above
        # n will be the length of 'v' - 
        # which is the number of times a locusId repeats in all_used_locId
        crt_result_d = sub_avg_fitness_func(list(v), strainWeight, strainFit_adjusted,
                               exp_used_strains, t0_used,
                               strainSD, k, cdebug=False)
        for keyy, valu in crt_result_d.items():
            fitness_d[keyy].append(valu)

    # fitness_l is a list that is populated with elements that are Series of 
    # dicts with values as numbers. We create a dataframe with all of them.
    fitness_df = pd.DataFrame.from_dict(fitness_d)
    fitness_df.sort_values(by=['locusId'], inplace=True)
    fitness_df['fit'] = mednorm(fitness_df['fitRaw'])
    fitness_df['fitNaive'] = mednorm(np.log2(1+fitness_df['tot']) - np.log2(1 + fitness_df['tot0']))
    #DEBUG fitness_df.to_csv("tmp/PY_fitness_df.tsv", sep="\t") 
    if debug:
        print("Actual number of groups: " + str(count_vals))

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


def getGeneFit1(strainFit, good_strainLocusIds, current_experiment_name=None, print_op=None):
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
        geneFit1 (pandas Series <float>): 
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

    #logging.info(f"Getting geneFit1 for {strainFit.name}")

    new_df = pd.DataFrame.from_dict({
            current_experiment_name : strainFit,
            'locusId': good_strainLocusIds
    })
    
    # We get the medians over all the strains with the same locusId
    # The index will be the locusIds
    medians_df = new_df.groupby(by='locusId').median()

    geneFit1 = mednorm(medians_df[current_experiment_name])


    if print_op is not None:
        geneFit1.to_csv(print_op, sep='\t') 

    return geneFit1


def getStrainPseudoCount(all_used_locusId, strainFit, readratio, minGeneFactorNStrains=3, 
                         debug_print_bool=False):
    """
    Args:

        all_used_locusId (Pandas Series <locusId (str)>): which locus the strain is associated with 
                                                     from all_df_subset['locusId'], and applied
                                                     boolean list 'strainsUsed' to it.
                    Length is nAllStrainsCentralGoodGenes(*)
                    (* Could be shortened due to having only strains in genesUsed12)
        minGeneFactorNStrains: int
        strainFit (pandas Series <float>): length is same as all_used_locusId
        readratio (float): (sum of counts/ sum of t0 for this sample index)

    Returns:
        strainPseudoCount (pandas Series): list of floats, same length as geneFit1,
                                            and we keep the index, being the row 
                                            number from all_df. No values can be 0

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

    # pd.Series length of nGenesUsed* - medians over gene locusIds with insertions
    # index is locusIds. Essentially a dict from locusId to medians over insertions.
    geneFitMedians = getGeneFit1(strainFit, all_used_locId, current_experiment_name) 

    # This 'table; is unique locus Ids pointing to the number of times they occur
    locusId2TimesSeen_d = py_table(all_used_locusId) 
    

    strainPseudoCount = []
    for locId in all_used_locusId.values:
        if locusId2TimesSeen_d[locId] >= minGeneFactorNStrains:
            # remember readratio is sum(experiment)/sum(time0s)
            strainPseudoCount.append(2**geneFitMedians[locId]*readratio)
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
        ix_l (int): list<int> of indexes (from grouped locusIds in t0_used)
                    (grouped by locusId)

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
           ret_d: dict with the following keys:
                fitRaw (float): 
                sd (float):
                sumsq (float):
                sdNaive (float):
                n (int):
                nEff (float ):
                tot (int ):
                tot0 (int):
    Description:
        What are the strainWeights? 
        We get the sum of the weights of all the strains
        
    """

    total_weight = strainWeight[ix_l].sum()
    fitRaw = (strainWeight[ix_l] * strainFit_adjusted[ix_l]).sum()/total_weight
    tot = exp_used_strains[ix_l].sum()
    tot0 = t0_used[ix_l].sum()
    sd = math.sqrt( ( (strainWeight[ix_l]**2) * (strainSD[ix_l]) ).sum()/total_weight)
    pre_sumsq1 = (strainFit_adjusted[ix_l] - fitRaw)**2
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
    simple log-ratio with pseudocount (of 1) and normalized so each scaffold has a median of 0
    note is *not* normalized except to set the total median to 0
    
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
    """

    sf_fit = mednorm( (1+all_cix_series).apply(np.log2) - (1 + all_cntrl_sum).apply(np.log2) )
    sf_se = (1/(1 + all_cix_series) + 1/(1 + all_cntrl_sum)).apply(math.sqrt)/ np.log(2)


    if debug_print:
        print("Input Series to check:")
        print(all_cix_series)
        print("Input Control Sum Series to check:")
        print(all_cntrl_sum)
        print("Computed strain fitness")
        print(sf_fit)
        print("Computed strain standard error")
        print(sf_se)
        raise Exception("Stopping for debugging.")

    return {
            "fit": sf_fit,
            "se": sf_se
            }



def NormalizeByScaffold(values, locusIds, genes_df, window=251, minToUse=10, cdebug=False):
    """
    Args:
        values (pandas Series): main_df['fit'] from AvgStrainFitness
        locusIds (pandas Series): main_df['locusIds'] from AvgStrainFitness
        genes_df: Data Frame created from genes.GC
        window (int): window size for smoothing by medians. Must be odd, default 251. For scaffolds
                      with fewer genes than this, just uses the median.
        minToUse (int): If a scaffold has too few genes, cannot correct for possible DNA extraction
                        bias so we need to remove data for that gene (i.e., returns NA for them)

    Returns:
        values (pandas Series of floats)

    Description:
    """


    if cdebug:
        print(f"locusIds from dataframe: {len(list(locusIds))}",
              f"locusIds from genes_df: {len(list(genes_df['locusId']))}")

    # We find indexes of locusIds within the genes' dataframe, locusId, column
    cmatch = py_match(list(locusIds), list(genes_df['locusId']))
    if None in cmatch:
        raise Exception("Fitness data for loci not in genes_df")

    # We get the begins of those genes in genes_df
    gn_begin = genes_df['begin'][cmatch]
    if cdebug:
        print(f"Length of genes beginning matched: {len(list(gn_begin))}")

    # py_split returns groupings of numerical iloc values grouped by the scaffoldIds
    perScaffoldRows = py_split(pd.Series(list(range(0, len(values)))), 
                               list(genes_df['scaffoldId'][cmatch]), 
                               typ='indices')

    # scaffoldId is str, rows is a list of ints (indeces for iloc) (iterable(?))
    for scaffoldId, rows in perScaffoldRows.items():
        if cdebug:
            print(f"Working on scaffoldId {scaffoldId} within NormalizeByScaffold")
            print("Rows associated with this scaffoldId: ")
            print(rows)
        if len(rows) < minToUse:
            if cdebug:
                print("Removing " + str(len(rows)) + " values for " + scaffoldId)
            values[rows] = np.nan 
        else:
            med = values[rows].median()
            if cdebug:
                print("Subtraxting median for " + scaffoldId + " " + str(med))
            values[rows] = values[rows] - med

            if len(rows) >= window:
                if cdebug:
                    print("Num rows: {len(rows)}")
                # We reset the index of the values to go from 0 to n instead of being the locusIds 
                new_gn_begin = gn_begin.iloc[rows].reset_index(drop=True)
                # srtd_begs is a list of indexes for the sorted begin values for this scaffold
                srtd_begs = py_order(new_gn_begin)
                rollmds = values[rows[srtd_begs]].rolling(window).median()
                if cdebug:
                    print("Subtract smoothed median for " + scaffoldId + ". max effect is " + \
                         f"{max(rollmds) - min(rollmds)}")
                # Changing values of the pandas series by the rolling median
                values[rows[srtd_begs]] = values[rows[srtd_begs]] - rollmds[srtd_begs]
                # density: kernel density estimates - default gaussian
                dns = stats.gaussian_kde(values[rows].dropna())
                cmax, cmin = values[rows].min(), values[rows].max();
                estimate_x = [cmin + (((cmax - cmin)/512)*i) for i in range(512)]
                estimate_y = dns.evaluate(estimate_x)
                mode = estimate_x[list(estimate_y).index(max(estimate_y))]
                if cdebug:
                    print("Subtract mode for " + scaffoldId + " which is at " + str(mode))
                values[rows] = values[rows] - mode

    return values


def mednorm(pd_series):
    # takes pandas series and returns pandas series with median subtracted
    crt_median = pd_series.median()
    new_series = pd_series - crt_median
    return new_series
