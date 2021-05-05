
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
    all_index_names = list(all_df.head())[meta_ix:]

    strainsUsed_hg2 = pd.Series(data=[bool(strainsUsed[i]) for i in range(len(strainsUsed)) if central_insert_bool_list[i]],
                                index=[i for i in range(len(strainsUsed)) if central_insert_bool_list[i]])
    all_df_central_inserts = all_df[central_insert_bool_list]
    num_ix_remaining = len(all_index_names)
    print(f"{num_ix_remaining}/{len(all_index_names)} total indeces to run through")

    # We take all the index names without the meta indeces (0-meta_ix (int))
    nSetIndexToRun = len(all_index_names) if nDebug_cols == None else nDebug_cols

    for set_index_name in all_index_names[:nSetIndexToRun]:
        print(f"Currently working on index {set_index_name}")
        
        start_time = time.time()
        if set_index_name is not None:
            exp_reads_w_central_insertions = all_df_central_inserts[set_index_name]
            gene_strain_fit_result = gene_strain_fit_func(set_index_name, 
                                                          exps_df, exp_reads_w_central_insertions, 
                                                          genes_df, expsT0,
                                                          t0tot, strainsUsed_hg2, central_insert_bool_list,
                                                          genesUsed, genesUsed12, minGenesPerScaffold,
                                                          all_df_central_inserts,
                                                          all_df)
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



def gene_strain_fit_func(set_index_name, exps_df, exp_reads_w_central_insertions, 
                         genes_df, expsT0,
                         t0tot, strainsUsed_hg2, central_insert_bool_list,
                         genesUsed, genesUsed12, minGenesPerScaffold,
                         all_df_has_gene,
                         all_df
                         ):
    """
    Description:
        This function is run for every single set_index_name in all_df, and that set_index_name
        is passed into this function as the first argument, 'set_index_name'. All other arguments
        are not changed at all when this function is called and are documented elsewhere. 
        Note that all_df_has_gene is a subset
        of all_df (all.poolcount) in which the barcode was inserted within a gene and within the
        central 80% of the gene. Then the majority of the work of the function is done within
        creating the variable 'gene_fit' while calling the function 'GeneFitness'.

    What happens in this function?
        First we find if this value is part of a t0set.
        If not, we get the related t0 set.
        
    Args:
        set_index_name: (str) Name of set and index from all_df (all.poolcount file)

        exps_df: Data frame holding exps file (FEBABarSeq.tsv)

        exp_reads_w_central_insertions: pandas Series of this set_index_name from all.poolcount
                                        with only values that have central insertions.

        [all_df]: Data frame holding all.poolcount file
        genes_df: Data frame holding genes.GC table
        expsT0: (dict) mapping (date setname) -> list<set.Index>
        t0tot: data frame where column names are 'date setname'
                and linked to a list of sums over the indexes that relate
                to that setname, with the list length being equal to the
                total number of strains (barcodes) in all.poolcount
                all columns are t0's?
        strainsUsed_hg2 pandas Series(list<bool>): whose length is same as num of Trues in central_insert_bool_list
                        equivalent index to central_insert_bool_list True values
        central_insert_bool_list: list<bool> whose length is total number of strains.
                    row with strains that have gene insertions between
                    0.1 < f < 0.9 hold value True
        genesUsed: list<locusId> where each locusId is a string
        genesUsed12 (list<str>): list of locusIds that have both high f (>0.5) and low f (<0.5)
                    insertions with enough abundance of insertions on both sides
        minGenesPerScaffold: int
        all_df_has_gene (Dataframe): The parts of all_df that corresponds to True in central_insert_bool_list

    Created vars:
        to_subtract: a boolean which says whether the 'short' name
                    is Time0
        t0set: Setname of related t0 set to current index name
        all_cix: The all_df column which is related to the current set_index_name
            (Should be a panda series)
        t0_series = 

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
    t0set, to_subtract = get_t0set_and_to_subtract(set_index_name, exps_df)

    # t0_series is the related time 0 total series.
    t0_series = t0tot[t0set]

    # to_subtract is true if this is a time zero itself, so we remove
    # its values from the other time0 values.
    if to_subtract:
        # We subtract the poolcount values from the t0 totals 
        t0_series = t0_series - exp_reads_w_central_insertions 

    # We check if any value is under 0
    for value in t0_series:
        if value < 0:
            raise Exception(f"Illegal counts under 0 for {set_index_name}: {value}")

    # Checking if there are no control counts
    # If all are 0
    if t0_series.sum() == 0:
        logging.info("Skipping log ratios for " + set_index_name + ", which has no"
                     " control counts\n.")
        return None
  
    use1 = [bool(x < 0.5) for x in all_df_has_gene['f']]

    # Note that central_insert_bool_list has to be the same length as all_cix, 
    # and t0_series, and strainsUsed
    gene_fit = GeneFitness(genes_df, all_df_has_gene, 
                           exp_reads_w_central_insertions, t0_series[central_insert_bool_list],
    		           strainsUsed_hg2, genesUsed, sorted(genesUsed12), 
    		           minGenesPerScaffold=minGenesPerScaffold,
                           set_index_name=set_index_name,
                           cdebug=False,
                           use1 = use1)

    
    cntrl = list(expsT0[t0set])
    if set_index_name in cntrl:
        cntrl.remove(set_index_name)
    if len(cntrl) < 1:
        raise Exception(f"No Time0 experiments for {set_index_name}, should not be reachable")

    strain_fit_ret_d = StrainFitness(exp_reads_w_central_insertions, 
                      all_df[cntrl].sum(axis=1)
                      )
    
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
            values (if this is a t0set)
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



def GeneFitness(genes_df, all_df_has_gene, crt_all_series_has_gene,
                crt_t0_series_has_gene, strainsUsed_has_gene, genesUsed,
                genesUsed12, minGenesPerScaffold=None,
                set_index_name=None,
                base_se = 0.1,
                cdebug=False,
                use1=None):
    """
    Args:
        genes_df: Data frame holding genes.GC table
                    must include cols locusId, scaffoldId, and begin (genes)
        all_df_has_gene: 
            subset of all_df (with good genes) which at the least contains headers:
                locusId, f (strainInfo)

        crt_all_series_has_gene (pandas Series): with counts for the current set.indexname 
                                 with central_insert_bool_list value true (0.1<f<0.9) [countCond]
        crt_t0_series_has_gene (pandas Series): with t0 counts for each strain [countT0]

        # Convert below into pandas series

        strainsUsed_has_gene pandas Series(list<bool>): whose length is Trues in central_insert_bool_list
                        equivalent index to central_insert_bool_list True values

        genesUsed: list<locusId> where each locusId is a string 
        genesUsed12 (list<str>): list of locusIds that have both high f (>0.5) and low f (<0.5)
                    insertions with enough abundance of insertions on both sides
        minGenesPerScaffold: int
        set_index_name: name of current set and index name from all.pool
        use1: list<bool> length of True values in has_gene, has True where strain insertion
                is .1<f<.5


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
            once for the insertions within .1<f<.5, and once for .5<f<.9


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
    if cdebug:
        with open("tmp/py_use1.txt", "w") as g:
            g.write(json.dumps(use1, indent=2))

    
    # Python code:
    main_df = AvgStrainFitness(crt_all_series_has_gene, 
                               crt_t0_series_has_gene, 
                               all_df_has_gene['locusId'],
      		               strainsUsed=strainsUsed_has_gene, genesUsed=genesUsed,
                               debug=False, mini_debug=1,
                               current_set_index_name=set_index_name,
                               run_typ="main_df")
    
    main_df['fitnorm'] = NormalizeByScaffold(main_df['fit'], main_df['locusId'],
                                             genes_df, minToUse=minGenesPerScaffold,
                                             cdebug=True)

    # Same as R:
    stn_used_hg1 = pd.Series(
                    data=[bool(strainsUsed_has_gene.iloc[i] and use1[i]) for i in range(len(strainsUsed_has_gene))],
                    index = strainsUsed_has_gene.index
                    )
    if cdebug:
        with open("tmp/py_sud1.txt", "w") as g:
            g.write(json.dumps(stn_used_hg1, indent=2))

    df_1 = AvgStrainFitness(crt_all_series_has_gene, 
                               crt_t0_series_has_gene, 
                               all_df_has_gene['locusId'],
      		               strainsUsed=stn_used_hg1, genesUsed=genesUsed12,
                               mini_debug=1,
                               current_set_index_name=set_index_name,
                               run_typ="df_1")
    
    # Same as R
    stn_used_hg2 = pd.Series(
                        data = [bool(strainsUsed_has_gene.iloc[i] and not use1[i]) for i in range(len(strainsUsed_has_gene))],
                        index = strainsUsed_has_gene.index
                        )
    if cdebug:
        with open("tmp/py_sud2.txt", "w") as g:
            g.write(json.dumps(stn_used_hg2, indent=2))
    if cdebug:
        debug_print(stn_used_hg2, 'stnhg2')


    df_2 = AvgStrainFitness(crt_all_series_has_gene, 
                            crt_t0_series_has_gene, 
                            all_df_has_gene['locusId'],
      		            strainsUsed=stn_used_hg2, genesUsed=genesUsed12,
                            mini_debug=1,
                            current_set_index_name=set_index_name,
                            run_typ="df_2")
    
    if cdebug:
        #DEBUG
        main_df.to_csv("tmp/py_main_df.tsv", sep="\t")
        df_1.to_csv("tmp/py_df_1.tsv", sep="\t")
        df_2.to_csv("tmp/py_df_2.tsv", sep="\t")
        with open("tmp/py_genesUsed12.json", "w") as g:
            g.write(json.dumps(genesUsed12, indent=2))


    for i in range(len(df_1['locusId'])):
        if df_1['locusId'].iat[i] != df_2['locusId'].iat[i]:
            raise Exception(f"Non-matching locusId: {df_1['locusId'].iat[i]}, at index {i}")

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




def AvgStrainFitness(crt_all_series_has_gene, 
                    crt_t0_series_has_gene, 
                    strainLocus,
		    minStrainT0 = 4, minGeneT0 = 40,
		    genesUsed=None, strainsUsed=None,
		    maxWeight = 20,
		    minGeneFactorNStrains=3,
		    debug=False,
                    mini_debug=0,
                    current_set_index_name=None,
                    run_typ=None):

    """
    Description:
        We take the subsets of the pandas Series that align with hasGene from all_df, 
            crt_all_series_has_gene is the column of the index
            crt_t0_series_has_gene is the sum of the related t0s
            strainLocus is the column of locusId that's related.

    Args:
        crt_all_series_has_gene (Pandas Series <int>): counts at the 
                    end of the experiment condition.
                    Comes from all_df, only counts that have genes.
        crt_t0_series_has_gene (Pandas Series <int>): counts for Time0 for each strain
        strainLocus (Pandas Series <locusId (str)>): total locusIds of 
                                        all_df - the same for every time 
                                        this function is run. These should correspond to 
                                        the rows in all_series and t0 series
        minStrainT0: int
        minGeneT0: int
        genesUsed: list<locusId> where each locusId is a string 
        maxWeight: int 
		 # maxWeight of N corresponds to having N reads on each side
                 #     (if perfectly balanced); use 0 for even weighting
		 # 20 on each side corresponds to a standard error of ~0.5; keep maxWeight low because outlier strains
		 # often have higher weights otherwise.

        strainsUsed: pandas Series: Subset of strainsUsed (list bool) which is True in
                              central_insert_bool_list and might also have other conditions such as f >/< 0.5
        current_set_index_name (str): Name of set index in all.poolcount that
                                    we are currently analyzing
        run_typ (str): Debugging which part of GeneFitness are we running?

    Returns:
        DataFrame: with cols:
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
        in strainLocus[strainsUsed]

    
    # If genesUsed (as a list of locusId) and strainsUsed (as boolean vector) are provided,
    # then considers only those strains & genes; minimum requirements.
    """

    if mini_debug > 0:
        print(f"Running AverageStrainFitness on {current_set_index_name} ({run_typ})")

    if (len(crt_all_series_has_gene) < 1 or len(crt_t0_series_has_gene) < 1 
            or len(strainLocus) < 1
            or len(crt_all_series_has_gene) != len(crt_t0_series_has_gene) or 
            len(crt_all_series_has_gene) != len(strainLocus)):
        raise Exception("None or misaligned input data:\n"
                f"crt_all_series len: {len(crt_all_series_has_gene)}\n"
                f"crt_t0_series len: {len(crt_t0_series_has_gene)}\n"
                f"strainLocus len: {len(strainLocus)}.\n"
                "All lengths must be equal and above 1."
                )
   
    # Check if accurate?
    crt_t0_name = crt_t0_series_has_gene.name


    # Up to here it's exactly the same as the R file, Note that the indexes of strainsUsed
    #       map to index integer locations in strainLocus
    strainsUsed = [bool(strainsUsed.iloc[ix] and (strainLocus.iloc[ix] in genesUsed)) for ix in \
                    range(len(strainsUsed))]

    if strainsUsed.count(True) == 0:
        raise Exception("After data preparing, no usable strains are left.")

    # All 3 series below have the same length
    # Note, already a difference of 2 values between current values and R input
    crt_t0_series_hg_su = crt_t0_series_has_gene[strainsUsed]
    crt_all_series_hg_su = crt_all_series_has_gene[strainsUsed]
    strainLocus_su = strainLocus[strainsUsed]

    if debug:
        logging.info("Number of unique values: " + str(len(strainLocus_su.unique())))
        logging.info("Above number is equivalent to number of rows in final DFs")
        crt_t0_series_hg_su.to_csv("tmp/py_crt_t0_series_A1.tsv", sep="\t")
        crt_all_series_hg_su.to_csv("tmp/py_crt_all_series_A1.tsv", sep="\t")
        strainLocus_su.to_csv("tmp/py_strainLocus_su.tsv", sep="\t")


    
    if sum(crt_t0_series_hg_su) != 0:
        readratio = sum(crt_all_series_hg_su) / sum(crt_t0_series_hg_su)
    else:
        raise Exception(f"No t0 values for this set/index value: {current_set_index_name}\n"
                         " Cannot get readratio (Division by 0).")

    if debug:
        print('readratio:')
        print(readratio)
    
    # This is where we get strain Fitness
    strainFit = getStrainFit(crt_all_series_hg_su, crt_t0_series_hg_su, readratio)

    if debug:
        with open('tmp/py_StrainFit.tsv', 'w') as g:
            g.write(json.dumps(list(strainFit), indent = 2))

    #print(strainFit)

    strainFitAdjust = 0

    # Per-strain "smart" pseudocount to give a less biased per-strain fitness estimate.
    # This is the expected reads ratio, given data for the gene as a whole
    # Arguably, this should be weighted by T0 reads, but right now it isn't.
    # Also, do not do if we have just 1 or 2 strains, as it would just amplify noise
    # note use of as.vector() to remove names -- necessary for speed

    # nStrains_d is a dict which takes list strainLocus_su of object -> number of times 
    #   it appears in the list. Ordered_strains is a unique list of strains.
    nStrains_d, ordered_strains = py_table(list(strainLocus_su), return_unique=True)

    # Almost the same as R version - what's the difference?
    nStrains = [nStrains_d[ordered_strains[i]] for i in range(len(ordered_strains))]

    if debug:
        with open('tmp/py_NStrains.tsv', 'w') as g:
            g.write(json.dumps(list(nStrains), indent = 2))


    geneFit1 = getGeneFit1(strainFit, strainLocus_su, current_set_index_name) 

    strainPseudoCount = getStrainPseudoCount(nStrains, minGeneFactorNStrains,
                                             geneFit1, readratio, strainLocus_su,
                                            debug_print_bool=False)


    condPseudoCount = [math.sqrt(x) for x in strainPseudoCount]
    t0PseudoCount = [1/math.sqrt(x) if x != 0 else np.nan for x in strainPseudoCount]


    strainFit_weight = get_strainFitWeight(condPseudoCount, crt_all_series_hg_su,
                        t0PseudoCount, crt_t0_series_hg_su,
                        strainFitAdjust)
    
    # strain Standard Deviation (list of floats) (We add 1 to avoid division by zero error)
    strainSD_pre = [math.sqrt(1/(1 + crt_t0_series_hg_su.iat[i]) + 1/(1+crt_all_series_hg_su.iat[i]))/np.log(2) \
                    for i in range(len(crt_t0_series_hg_su))]

    strainSD = pd.Series(data=strainSD_pre,
                             index=crt_t0_series_hg_su.index)


    # "use harmonic mean for weighting; add as small number to allow maxWeight = 0."
    strainWeight = []
    # We use ix_vals to maintain the indices from the original series
    ix_vals = []
    
    for i in range(len(crt_t0_series_hg_su)):
        # we get the minimum from 'maxWeight (=20)' and a safe harmonic mean 
        cmin = min(maxWeight, 2/( 1/(1+crt_t0_series_hg_su.iat[i]) + 1/(1 + crt_all_series_hg_su.iat[i]) ) )
        strainWeight.append(cmin)
    strainWeight = pd.Series(data=strainWeight, index=crt_t0_series_hg_su.index)


    # Number of groups should be equal to the number of unique values in strainLocus_su
    if debug:
        num_unique = len(strainLocus_su.unique())
        print(f"Number of unique strains in strainLocus_su: {num_unique}")

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

    # Note: the number of rows in the resultant dataframes is equal to the
    # number of unique values in strainLocus_su
    t0_index_groups = py_split(crt_t0_series_hg_su, strainLocus_su, typ="groups")
    count_vals = 0
    for k, v in t0_index_groups.items():
        count_vals += 1
        if debug:
            print(f"t0_index_groups key: {k}")
            print("t0_index_groups value:")
            print(v)
        # group the values by locusId = strainLocus

        # crt_result is a dict that matches with fitness_d above
        crt_result_d = sub_avg_fitness_func(list(v), strainWeight, strainFit_weight,
                               crt_all_series_hg_su, crt_t0_series_hg_su,
                               strainSD, k)
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



def getStrainFit(crt_all_series_hg_su, crt_t0_series_hg_su, readratio):
    """
    Description:
        We take the current values, add the readratio (why?) then take the log2 values
            then normalize by the median
    Args:
        crt... : pandas series with integers
        readratio: float
    returns:
        strainFit (pandas series): of floats length is the same as len(crt_all_series_hg_su) =
                                                                   len(crt_t0_series_hg_su)

    use sqrt(readratio), or its inverse, instead of 1, so that the expectation
    is about the same regardless of how well sampled the strain or gene is
    """
    # use sqrt(readratio), or its inverse, instead of 1, so that the expectation
    # is about the same regardless of how well sampled the strain or gene is
    all_1 = crt_all_series_hg_su + math.sqrt(readratio)
    t0_1 = crt_t0_series_hg_su + 1/math.sqrt(readratio)
    all_2 = all_1.apply(math.log2)
    t0_2 = t0_1.apply(math.log2)
    strainFit = mednorm(all_2 - t0_2)
    return strainFit


def getGeneFit1(strainFit, strainLocus_su, current_set_index_name, print_op=None):
    """
    strainFit: pandas Series of locusIds as index labels for floats. It's the 
                normalized difference between actual counts and t0 counts.
    strainLocus_su: list<locusId (str)>
        Both inputs have the same length
    We group the values of strainFit by their locusIds
        in strainLocus_su, and calculate the median of each group
        Then we take the overall mednorm, which means subtracting
        the total median from each value.

    Returns: 
        geneFit1 (pandas Series (?)):
    """

    #logging.info(f"Getting geneFit1 for {strainFit.name}")

    new_df = pd.DataFrame.from_dict({
            current_set_index_name : strainFit,
            'locusId': strainLocus_su
    })
   
    medians_df = py_aggregate(new_df, 'locusId', func='median')

    geneFit1 = mednorm(medians_df[current_set_index_name])

    if print_op is not None:
        geneFit1.to_csv(print_op, sep='\t') 

    return geneFit1


def getStrainPseudoCount(nStrains, minGeneFactorNStrains, geneFit1, readratio, strainLocus_su,
                         debug_print_bool=False):
    """
    Args:
        nStrains list: ( used to be pandas Series) list of number of times locusId appeared ordered
                        the same way as
        minGeneFactorNStrains: int
        geneFit1 (pandas Series): median-normalized medians of locusIds over strains
                
        readratio (float): (sum of counts/ sum of t0 for this sample index)
        strainLocus_su (Pandas Series <locusId (str)>): which locus the strain is associated with 
                                                     from all_df_subset['locusId'], and applied
                                                     boolean list 'strainsUsed' to it.

    Returns:
        strainPseudoCount (pandas Series): list of floats, same length as geneFit1 
    """
    
    
    # unique_nums is numbering all unique values from strainLocus_su with numbers 0 and up 
    # e.g., ["a","a","a","b","b",...] -> [0, 0, 0, 1, 1, ...]
    unique_nums = []
    unique_vals = {}
    unique_strain_loci = pd.unique(strainLocus_su)
    crt_unique = -1
    if debug_print_bool:
        print("length of strainLocus_su:")
        print(len(strainLocus_su))
        print(".size ?")
        print(strainLocus_su.size)
        print("Number of unique values:")
        print(len(unique_strain_loci))


    for i in range(strainLocus_su.size):
        locusId = strainLocus_su.iat[i]
        if locusId in unique_vals:
            unique_nums.append(unique_vals[locusId])
        else:
            crt_unique += 1
            unique_vals[locusId] = crt_unique
            unique_nums.append(crt_unique)

    if debug_print_bool:
        #debug_print(unique_nums, 'unique_nums')
        with open("pUniqueNums.tsv", "w") as g:
            g.write(json.dumps(unique_nums, indent=2))


    strainPseudoCount = []
    if debug_print_bool:
        print("length of nStrains")
        print(len(nStrains))
        print("length of geneFit1:")
        print(len(geneFit1))
        print('max val from unique_nums:')
        print(max(unique_nums))
    for i in range(len(unique_nums)):
        if nStrains[unique_nums[i]] >= minGeneFactorNStrains:
            strainPseudoCount.append(2**geneFit1[unique_nums[i]]*readratio)
        else:
            strainPseudoCount.append(readratio)

    if debug_print_bool:
        with open('tmp/py_StrainPseudoCount.json', 'w') as g:
            g.write(json.dumps(strainPseudoCount, indent=2))

        print("length of strainPseudoCount:")
        print(len(strainPseudoCount))

    return pd.Series(data=strainPseudoCount)


def get_strainFitWeight(condPseudoCount, crt_all_series_hg_su,
                        t0PseudoCount, crt_t0_series_hg_su,
                        strainFitAdjust
                        ):
    """
    Args:
        condPseudoCount:
        t0PseudoCount: 
        strainFitAdjust: (int)

    Returns:
        strainFit_weight (pandas Series) with index labels fitting crt_all_series...
    """
    strainFit_weight = []
    for i in range(len(condPseudoCount)):
        strainFit_weight.append(math.log2(condPseudoCount[i] + crt_all_series_hg_su.iat[i]) \
                                - math.log2(t0PseudoCount[i] + crt_t0_series_hg_su.iat[i]) \
                                - strainFitAdjust)

    return pd.Series(data=strainFit_weight, index=crt_all_series_hg_su.index)


def sub_avg_fitness_func(ix_l, strainWeight, strainFit_weight,
                               crt_all_series_hg_su, crt_t0_series_hg_su,
                               strainSD, series_name, cdebug=False):
    """
    Args:
        ix_l (int): list<int> of indexes (from grouped locusIds in crt_t0_series_hg_su)
                    (grouped by locusId)

        strainWeight (pandas Series list<float>): each element has a minimum value of 'maxWeight', 
                                    which normally equals 20,
                                    other elements have values which are computed 
                                    in AvgStrainFitness func
        strainFit_weight pandas Series:  Same index as strainWeight
        crt_all_series_hg_su (pandas series list<int>): 
        crt_t0_series_hg_su (pandas series list<int>): 
        strainSD (list<float>): 
        series_name: (str)
    Returns:
           ret_d: dict with the following keys:
                fitRaw: float
                sd: float
                sumsq: float
                sdNaive: float
                n: int
                nEff: float 
                tot: int 
                tot0: int
    """
    totw = sum(strainWeight[ix_l]) 
    sfw_tmp = list(strainFit_weight[ix_l])
    fitRaw = sum(py_mult_vect(list(strainWeight[ix_l]), sfw_tmp))/totw
    tot = sum(crt_all_series_hg_su[ix_l])
    tot0 = sum(crt_t0_series_hg_su[ix_l])
    pre_sd_list1 = [strainWeight[j]**2 * strainSD[j] for j in ix_l]
    sd = math.sqrt(sum(pre_sd_list1))/totw
    pre_sumsq1 = [(strainFit_weight[j] - fitRaw)**2 for j in ix_l]
    sumsq = sum(py_mult_vect(list(strainWeight[ix_l]), pre_sumsq1))/totw
    
    # 'high-N estimate of the noise in the log2 ratio of fitNaive'
    # 'But sdNaive is actually pretty accurate for small n -- e.g.'
    # 'simulations with E=10 on each side gave slightly light tails'
    # '(r.m.s.(z) = 0.94).'

    sdNaive = math.sqrt( (1/(1+tot)) + (1/(1+tot0)) )/np.log(2)
    
    nEff = totw/(strainWeight[ix_l].max())
    ret_d = {
             "fitRaw": fitRaw,
             "sd": sd,
             "sumsq": sumsq,
             "sdNaive": sdNaive,
             "n":len(ix_l),
             "nEff": nEff,
             "tot": tot,
             "tot0": tot0,
             "locusId": series_name
            }

    return ret_d


def StrainFitness(all_cix_series,
                all_cntrl_sum):
    """
    simple log-ratio with pseudocount (of 1) and normalized so each scaffold has a median of 0
    note is *not* normalized except to set the total median to 0
    
    Args:
        all_cix_series (pandas Series): The current set+index column of values from all.poolcount
        all_cntrl_sum (pandas Dataframe): The sum of the current control values without the current index; should
                       be a data frame with set+index names for controls -> sum of values over all rows.
    Returns:
        fit: pandas Series (float) with a computation applied to values
        se: pandas Series (float) with computations applied to values
    """
    sf_fit = mednorm( (1+all_cix_series).apply(np.log2) - (1 + all_cntrl_sum).apply(np.log2) )
    sf_se = (1/(1 + all_cix_series) + 1/(1 + all_cntrl_sum)).apply(math.sqrt)/ np.log(2)
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
