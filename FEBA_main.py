#!python3

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

"""
Main functions:
    FEBA_Fit



TD:
    compute_cofit
        SpecificPhenotypes

    Rewrite definitions of all variables so you can comprehend the entire
        program from top to bottom.

    Convert all lists into pandas Series, so you can retain indeces from
    the original dataframes.
    
    Whenever you want to change a value within a dataframe, use
        df.loc[row_index, column_name]

    All debugging printed files should start with py_

    Breakpoints occur after there is a lot of computation:
        e.g. after StrainClosestGenes

Function Descriptions:
    AvgStrainFitness:


Explanations:
    lr: "logratios"
    lrn: "logratios normalized"
        
"""




#from collections import Counter

#TD: AvgStrainFitness, NormalizeByScaffold
# TD: Consider very slight differences between python program and R
#       differences of as little as 1-3 lines

def FEBA_Fit(exps_df, all_df, genes_df, 
             dir=".",
	     genesUsed=None, strainsUsed=None, genesUsed12=None,
	     minT0Strain=3, minT0Gene=30,
	     minGenesPerScaffold=10,
	     #pred=CrudeOp(genes),
	     okDay=True, # OK to use Time0 from another day on the same lane, if necessary?
	     okLane=True, # OK to compare to Time0 from another lane, if necessary?
	     metacol=list(range(0,7)), 
             meta_ix=7,
	     # names of experiments to ignore; experiments with Drop=True are also ignored
	     ignore=None,
	     # ignore those below this threshold, unless ignore is set
	     minSampleReads = 2*10e4,
	     debug=False, computeCofit=True,
             dbg_lvl=0,
             debug_cols=10^3):
    """
    This is the central function in the analysis.
    It contains many subfunctions with detailed explanations

    Args:
        exps_df (pandas DataFrame):
            Must contain cols:
                Index
                Date_pool_expt_started
                Description
                SetName
                [Drop]
                short
                name
                [num]
                Group
                [Condition_1]
                [Condition_2]
        all_df (pandas DataFrame): all.poolcount dataframe
            must start the set+index columns at the index described by function parameter 'meta_ix'
            must contain cols:
                locusId
                f

        genes_df (pandas DataFrame): genes.GC dataframe
            Must contain cols:
                scaffoldId
                locusId
                desc
                begin
                end
        genesUsed: optional list of locusIds (list<str>)
        genesUsed12: optional list of locusIds (list<str>)
        strainsUsed: optional list of str, one per gene?
        pred: dataframe  (???)
        KB : setting minSampleReads to ?!
        debug_cols: (int) Number of columns to run through from all_df to compute results 
        ignore (None or ):
    Returns:
        gene_fit_d (python dict): Contains keys:
            g (pandas Series (str)): pandas Series of locusIds
            lr (float): dataframe with one column per setindexname
            lrNaive (float): dataframe with one column per setindexname
            lr1 (float): dataframe with one column per setindexname
            lr2 (float): dataframe with one column per setindexname
            lrn (float): dataframe with one column per setindexname
            lrn1 (float): dataframe with one column per setindexname
            lrn2 (float): dataframe with one column per setindexname
            fitRaw (float): dataframe with one column per setindexname
            n (int): dataframe with one column per setindexname
            nEff (float): dataframe with one column per setindexname
            pseudovar (float): dataframe with one column per setindexname
            q (pandas DataFrame): contains columns:
                name (str), 
                short (str), 
                t0set (str), 
                num (int), 
                nMapped (int), 
                nPastEnd (int), 
                nGenic (int), 
                nUsed (int), 
                gMed (int), 
                gMedt0 (int), 
                gMean (float), 
                cor12 (float), 
                mad12 (float), 
                mad12c (float), 
                mad12c_t0 (float), 
                opcor (float), 
                adjcor (float), 
                gccor (float), 
                maxFit (float) 
                u (bool)
            sumsq (float): dataframe with one column per setindexname
            sd (float): dataframe with one column per setindexname
            sdNaive (float): dataframe with one column per setindexname
            se (float) Standard Error dataframe with one column per setindexname
            t: (float) t-statistic dataframe with one column per setindexname
            tot1 (int or nan) dataframe with one column per setindexname
            tot1_0 (int or nan) dataframe with one column per setindexname
            tot2 (int or nan) dataframe with one column per setindexname
            tot2_0 (int or nan) dataframe with one column per setindexname
            tot (int or nan) dataframe with one column per setindexname
            tot0 (int or nan) dataframe with one column per setindexname
            version (str)
            'genesUsed',
            'strainsUsed',
            'genesUsed12',
            'gN',
            't0_gN',
            'strains',
                used,
                enoughT0
                & multiple others (all_df meta_ix columns)
            'strain_lr',
            'strain_se',
            'high' 
            [pairs]:
                adjDiff:
                    Gene1, Gene2, sysName1, type1, scaffoldId, begin1, end1, strand1, name1, desc1, GC1, 
                    nTA1, locusId, sysName2, type2, begin2, end2, strand2, name2, desc2, GC2, nTA2
                    rfit (float)
                random:
                    Gene1
                    Gene2
                    rfit
                pred:
                    Gene2, Gene1, sysName1, type1, scaffoldId1, begin1, end1, strand1, name1, desc1, GC1, nTA1, 
                    sysName2, type2, scaffoldId2, begin2, end2, strand2, name2, desc2, GC2, nTA2, Sep, bOp
                    rfit
            [cofit] (pandas DataFrame):  
                locusId (str), 
                hitId (str) 
                cofit (float)
                rank (int)
            [specphe]: (Not done)

    """

    # We find the indeces to ignore (info inside func) (ignore is list<str>)
    all_df, exps_df = set_up_ignore(ignore, all_df, 
                                            exps_df, minSampleReads,
                                            meta_ix, dbg_prnt=True)
    

    exps_df = prepare_time0s(exps_df)

    # this is a list of booleans over all rows of all_df if their f is 0.1<f<0.9
    has_gene2 = [True if (0.1<=x<=0.9) else False for x in all_df['f']]

    num_has_gene2 = has_gene2.count(True)

    if dbg_lvl > 0:
        logging.info(f"{num_has_gene2} is the number of strains with central "
                      "insertions in the genes,\n"
                      "which is equivalent to the number of 'Trues' in has_gene2.")

    tmp_all_df = all_df.iloc[:,meta_ix:][has_gene2]
    tmp_all_df['locusId'] = all_df['locusId'][has_gene2]
    # all_gN is a dataframe with unique locusId values with sums
    all_gN = py_aggregate(tmp_all_df, "locusId", func="sum")

    exps_df['t0set'] = [exps_df['Date_pool_expt_started'].iat[ix] + " " + \
                        val for ix, val in exps_df['SetName'].items()]

    if dbg_lvl>2:
        debug_print(exps_df['t0set'], 'exps_df_column t0set')

    expsT0 = createExpsT0(exps_df)
    if dbg_lvl>2:
        with open("tmp/py_expsT0.json", "w") as g:
            g.write(json.dumps(expsT0, indent=2))

    


    expsT0, exps_df = update_expsT0_and_exps_df_with_nont0sets(expsT0, 
                                    exps_df, okLane, okDay,
                                    print_bool=True,
                                    dbgp=True)

    # Here we combine the date set names that are t0 experiments into a single
    # dataframe called t0tot, which has the same number of rows as all.poolcount
    t0tot = create_t0tot(expsT0, all_df, dbg_prnt=True)

    
    if dbg_lvl > 0:
        if len(expsT0.keys()) == 0:
            print("No remaining time0 keys")

    indexBy = createIndexBy(all_df, has_gene2)


    t0_gN = createt0gN(t0tot, has_gene2, indexBy, debug_print_bool=True) 

    print_log_info1(t0tot, t0_gN)

    # strainsUsed will be a list of booleans with length being
    # total number of strains.
    strainsUsed = createStrainsUsed(t0tot, minT0Strain, has_gene2, strainsUsed)

    if dbg_lvl>4:
        with open("tmp/py_strainsUsedA1.tsv", "w") as g:
            g.write("\n".join([str(x) for x in strainsUsed]))


    # We get the unique locus Ids where we can use the strain
    unique_usable_locusIds = all_df['locusId'][strainsUsed].unique()
    if len(unique_usable_locusIds) < 10:
        raise Exception("Less than ten usable locusIds, program designed to stop.")
    else:
        logging.info(f"Unique number of usable locusIds: {len(unique_usable_locusIds)}")

    genesUsed = getGenesUsed(t0tot, strainsUsed, all_df, minT0Gene, genesUsed)

    genesPerScaffold = getGenesPerScaffold(genes_df, genesUsed)

    smallScaffold, smallLocusIds = get_smallScaffold(genesPerScaffold, minGenesPerScaffold,
                                                     genes_df)

    # refining genesUsed (but locusIds aren't scaffolds...)
    genesUsed = [x for x in genesUsed if x not in smallLocusIds]
    genesUsed =  remove_genes_if_not_in_genes_df(genesUsed, genes_df)

    print_info2(has_gene2, all_df, strainsUsed, genesUsed)

    genesUsed12 = get_GenesUsed12(genesUsed12, minT0Gene, strainsUsed, all_df,
                                  t0tot)

    logging.info(f"For cor12, using {len(genesUsed12)} genes. ");

    check_if_every_t0set_is_in_t0tot(exps_df, t0tot)


    export_special_vars("tmp/special_vars", genesUsed, pd.Series(strainsUsed), genesUsed12, 
                                           all_gN, t0_gN, t0tot)



    GeneFitResults = compute_GeneFitResults(all_df, exps_df, genes_df,
                                            expsT0, t0tot, 
                                            genesUsed, genesUsed12, strainsUsed, has_gene2, 
                                            minGenesPerScaffold=10, meta_ix=7,
                                            debug=False, debug_cols=debug_cols)
                            

    strainsUsed_hg2 = pd.Series(data=[bool(strainsUsed[i]) for i in range(len(strainsUsed)) if has_gene2[i]],
                                index=[i for i in range(len(strainsUsed)) if has_gene2[i]])

    #exps_df.to_csv("tmp/py_exps_df235.tsv", sep="\t")
    # Store current results for faster testing
    export_or_import_genefitresults(GeneFitResults, "exp", "tmp/ResultStorage", dbg_print=True)
    gene_fit_d, CrudeOp_df = start_gene_fit_d(GeneFitResults, exps_df, all_df, genes_df, 
                     has_gene2, meta_ix=meta_ix, debug=debug)

    gene_fit_d = finish_gene_fit_d(gene_fit_d, GeneFitResults, genes_df, all_df, exps_df,
                      genesUsed, strainsUsed, genesUsed12,
                      all_gN, t0_gN, t0tot, CrudeOp_df, meta_ix=meta_ix, 
                      minT0Strain=minT0Strain)
    
    export_gene_fit_d(gene_fit_d, "tmp/ResultStorage2")

    logging.debug("Keys in gene_fit_d: \n" + ", ".join(gene_fit_d.keys()))

    return gene_fit_d


def export_special_vars(special_vars_dir, genesUsed, strainsUsed, genesUsed12, 
                        all_gN, t0_gN, t0tot):
    """
    """
    pd.Series(genesUsed).to_csv(os.path.join(special_vars_dir, "genesUsed.tsv"), sep="\t")
    strainsUsed.to_csv(os.path.join(special_vars_dir, "strainsUsed.tsv"), sep="\t")
    pd.Series(genesUsed12).to_csv(os.path.join(special_vars_dir, "genesUsed12.tsv"), sep="\t")
    all_gN.to_csv(os.path.join(special_vars_dir, "all_gN.tsv"), sep="\t")
    t0_gN.to_csv(os.path.join(special_vars_dir, "t0_gN.tsv"), sep="\t")
    t0tot.to_csv(os.path.join(special_vars_dir, "t0tot.tsv"), sep="\t")
    logging.info(f"Exported all special_vars_to dir: {special_vars_dir}")

def compute_GeneFitResults(all_df, exps_df, genes_df, 
                         expsT0, t0tot, 
                         genesUsed, genesUsed12, strainsUsed, has_gene2, 
                         minGenesPerScaffold=10, meta_ix=7,debug=False, debug_cols=None):
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

    strainsUsed_hg2 = pd.Series(data=[bool(strainsUsed[i]) for i in range(len(strainsUsed)) if has_gene2[i]],
                                index=[i for i in range(len(strainsUsed)) if has_gene2[i]])
    all_df_has_gene = all_df[has_gene2]
    num_ix_remaining = len(all_index_names)
    print(f"{num_ix_remaining}/{len(all_index_names)} total indeces to run through")

    # We take all the index names without the meta indeces (0-meta_ix (int))
    nSetIndexToRun = len(all_index_names) if debug_cols == None else debug_cols

    for set_index_name in all_index_names[:nSetIndexToRun]:
        print(f"Currently working on index {set_index_name}")
        
        start_time = time.time()
        if set_index_name is not None:
            gene_strain_fit_result = gene_strain_fit_func(set_index_name, 
                                                          exps_df, all_df, 
                                                          genes_df, expsT0,
                                                          t0tot, strainsUsed_hg2, has_gene2,
                                                          genesUsed, genesUsed12, minGenesPerScaffold,
                                                          all_df_has_gene)
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


def export_or_import_genefitresults(genefitresults, typ, dir_path, dbg_print=False):
    """
    This function is mainly for debugging purposes ( results stored at 'tmp/ResultStorage')
    Args:
        typ (str): One of "exp" (export) or "imp" (import)
        genefitresults: 
            setnameIndex -> ret_d
               ret_d:
                   gene_fit: DataFrame, contains cols:
                        locusId (str),
                        fit (float): (unnormalized
                        fitNaive (float):
                        fit1 (float):
                        fit2 (float):
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
                   strain_fit: pandas Series (float) 
                   strain_se: pandas Series (float) 
        dir_path:
            Directory path to export to or to import from

                    
    """
    if typ == "exp":
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        for setindexname, ret_d in genefitresults.items():
            if dbg_print:
                print(f"Exporting results for ret_d {setindexname}")
            ret_d['gene_fit'].to_csv(
                                os.path.join(dir_path, "py_" + setindexname + "_gene_fit.dftsv"), 
                                sep="\t")
            ret_d['strain_fit'].to_csv(
                                os.path.join(dir_path, "py_" + setindexname + "_strain_fit.dftsv"), 
                                sep="\t")
            ret_d['strain_se'].to_csv(
                                os.path.join(dir_path, "py_" + setindexname + "_strain_se.dftsv"), 
                                sep="\t")

    elif typ == "imp":
        if not os.path.isdir(dir_path):
            raise Exception(f"Import directory does not exist {dir_path}")
        
        input_d = {
                "locusId": str
                }
        dir_files = os.listdir(dir_path)
        setindexnames = {}
        for f in dir_files:
            # accounts for "py_" and "_gene_fit_.dftsv"
            new_stixnm = f.split("py_")[1]
            if "_gene_fit" in new_stixnm:
                new_stixnm = new_stixnm.split("_gene_fit")[0]
            elif "_strain_fit" in new_stixnm:
                new_stixnm = new_stixnm.split("_strain_fit")[0]
            elif "_strain_se" in new_stixnm:
                new_stixnm = new_stixnm.split("_strain_se")[0]
            setindexnames[new_stixnm] = 1
        genefitresults = {}
        for setindexname, nan in setindexnames.items():
            if dbg_print:
                print(f"Importing results for {setindexname}")
            ret_d = {}
            ret_d['gene_fit'] = pd.read_table(
                                    os.path.join(dir_path, "py_" + setindexname + "_gene_fit.dftsv"),
                                        dtype=input_d, index_col=0)
            ret_d['strain_fit'] = pd.read_table(
                                os.path.join(dir_path, "py_" + setindexname + "_strain_fit.dftsv"),
                                index_col=0)
            ret_d['strain_se'] = pd.read_table(
                                os.path.join(dir_path, "py_" + setindexname + "_strain_se.dftsv"),
                                index_col=0)
            genefitresults[setindexname] = ret_d

        return genefitresults

    else:
        raise Exception(f"Cannot recognize type {typ}")

    if dbg_print:
        print("Finished Exporting results")

    return None

            




def start_gene_fit_d(GeneFitResults, exps_df, all_df, genes_df, 
                    has_gene2, meta_ix=7, debug=False):
    """
    Args:
        GeneFitResults:
            setnameIndex -> ret_d
               ret_d:
                   gene_fit: DataFrame, contains cols:
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
                   strain_fit: pandas Series (float) 
                   strain_se: pandas Series (float) 
    Returns:
        gene_fit_d: (python dict)
            g (pandas Series (str)): pandas Series of locusIds
            lr (float): dataframe with one column per setindexname
            lrNaive (float): dataframe with one column per setindexname
            lr1 (float): dataframe with one column per setindexname
            lr2 (float): dataframe with one column per setindexname
            lrn (float): dataframe with one column per setindexname
            lrn1 (float): dataframe with one column per setindexname
            lrn2 (float): dataframe with one column per setindexname
            fitRaw (float): dataframe with one column per setindexname
            n (int): dataframe with one column per setindexname
            nEff (float): dataframe with one column per setindexname
            pseudovar (float): dataframe with one column per setindexname
            q (pandas DataFrame): contains columns:
                name (str), 
                short (str), 
                t0set (str), 
                num (int), 
                nMapped (int), 
                nPastEnd (int), 
                nGenic (int), 
                nUsed (int), 
                gMed (int), 
                gMedt0 (int), 
                gMean (float), 
                cor12 (float), 
                mad12 (float), 
                mad12c (float), 
                mad12c_t0 (float), 
                opcor (float), 
                adjcor (float), 
                gccor (float), 
                maxFit (float) 
                u (bool)
            sumsq (float): dataframe with one column per setindexname
            sd (float): dataframe with one column per setindexname
            sdNaive (float): dataframe with one column per setindexname
            se (float) Standard Error dataframe with one column per setindexname
            t: (float) t-statistic dataframe with one column per setindexname
            tot1 (int or nan) dataframe with one column per setindexname
            tot1_0 (int or nan) dataframe with one column per setindexname
            tot2 (int or nan) dataframe with one column per setindexname
            tot2_0 (int or nan) dataframe with one column per setindexname
            tot (int or nan) dataframe with one column per setindexname
            tot0 (int or nan) dataframe with one column per setindexname
            version (str)

        

    """

    gene_fit_d = initialize_gene_fit_d(GeneFitResults, debug=True) 

    # What is q?


    q_col = ["name", "short", "t0set"]
    if "num" in exps_df:
        q_col.append("num")
    # We get the rows which have 'name' in lrn1 columns, and then we 
    #   only get the columns in q_col
    tmp_name_in_lrn = [True if exps_df['name'].iloc[i] in gene_fit_d['lrn1'].head() else False for i \
                        in range(len(exps_df['name']))]
    gene_fit_d['q'] = exps_df[tmp_name_in_lrn][q_col]
    gene_fit_d['q'].index = list(gene_fit_d['q']['name'])
    gene_fit_d['q'].to_csv("tmp/py_gene_fit_q.tsv", sep="\t")
    qnames = gene_fit_d['q']['name']
    for i in range(len(qnames)):
        if not qnames.iat[i] == list(gene_fit_d['lrn'].head())[i]:
            raise Exception(f"Mismatched names in fit: {qnames.iat[i]} != "
                            f"{list(gene_fit_d['lrn'].head())[i]}")



    save_gene_fit_d(gene_fit_d, prnt_dbg=False)

    if debug:
        print("Running FitReadMetrics() and FitQuality()")
    st = time.time()
    fitreadmet = FitReadMetrics(all_df, qnames, has_gene2)
    fitreadmet.to_csv("tmp/py_FitReadMetrics.tsv", sep="\t")
    print(f"Time to run FitReadMetrics: {time.time() - st} seconds")

    st = time.time()
    fq_result, CrudeOp_df = FitQuality(gene_fit_d, genes_df, prnt_dbg=True)
    print(f"Time to run FitQuality: {time.time() - st} seconds")

    gene_fit_d['q'] = pd.concat([gene_fit_d['q'], 
                                 fitreadmet,
                                 fq_result], axis=1)
   
    #DEBUG:
    gene_fit_d['q'].to_csv("tmp/py_gene_fit_q2.tsv", sep="\t")
    # status is a pandas series of str
    status = FEBA_Exp_Status(gene_fit_d['q'], dbg_prnt=True)
    # We get a list of status is ok + False for the rows of q that surpass length of status
    gene_fit_d['q']['u'] = [status.iat[i] == "OK" for i in range(len(status))] + [False]*(gene_fit_d['q'].shape[0] - len(status))


    #DEBUG:
    gene_fit_d['q'].to_csv("tmp/py_gene_fit_q2.tsv", sep="\t")

    for s in ["low_count", "high_mad12", "low_cor12", "high_adj_gc_cor"]:
        if list(status).count(s) > 0:
            logging.info(f"{s}: {gene_fit_d['q']['name'][status == s]}")

    return gene_fit_d, CrudeOp_df


def finish_gene_fit_d(gene_fit_d, GeneFitResults, genes_df, all_df, exps_df,
                      genesUsed, strainsUsed, genesUsed12,
                      all_gN, t0_gN, t0tot, CrudeOp_df, meta_ix=7, minT0Strain=3,
                      dbg_prnt=False):

    """
    Args:
        gene_fit_d (python dict):
            g (pandas Series (str)): pandas Series of locusIds
            lr (float): dataframe with one column per setindexname (Fitness)
            lrNaive (float): dataframe with one column per setindexname
            lr1 (float): dataframe with one column per setindexname
            lr2 (float): dataframe with one column per setindexname
            lrn (float): dataframe with one column per setindexname ( Fitness normalized)
            lrn1 (float): dataframe with one column per setindexname
            lrn2 (float): dataframe with one column per setindexname
            fitRaw (float): dataframe with one column per setindexname
            n (int): dataframe with one column per setindexname
            nEff (float): dataframe with one column per setindexname
            pseudovar (float): dataframe with one column per setindexname
            sumsq (float): dataframe with one column per setindexname
            sd (float): dataframe with one column per setindexname
            sdNaive (float): dataframe with one column per setindexname
            se (float) Standard Error dataframe with one column per setindexname
            t: (float) t-statistic dataframe with one column per setindexname
            tot1 (int or nan) dataframe with one column per setindexname
            tot1_0 (int or nan) dataframe with one column per setindexname
            tot2 (int or nan) dataframe with one column per setindexname
            tot2_0 (int or nan) dataframe with one column per setindexname
            tot (int or nan) dataframe with one column per setindexname
            tot0 (int or nan) dataframe with one column per setindexname
            version (str)
            q (pandas DataFrame): contains columns:
                name, short, t0set, num, nMapped, nPastEnd, nGenic, nUsed, gMed, gMedt0, gMean, 
                cor12, mad12, mad12c, mad12c_t0, opcor, adjcor, gccor, maxFit, u
        GeneFitResults (dict): set_index_names -> gene_strain_fit_result
            gene_strain_fit_result (dict):
                gene_fit: DataFrame, contains cols:
                    fit, fitNaive, fit1, fit2, fitnorm, fitnorm1, fitnorm2, fitRaw
                    locusId, n, nEff, pseudovar, sumsq, sd, sdNaive, se, t, tot1
                    tot1_0, tot2, tot2_0, tot, tot0
                strain_fit: pandas Series (float) with a computation applied to values
                strain_se: pandas Series (float) with a computation applied to values
        strainsUsed pandas Series(list<bool>):  
        CrudeOp_df (pandas DataFrame): Output from function CrudeOp(genes_df)
                Gene2, Gene1, sysName1, type1, scaffoldId1, begin1, end1, strand1, name1, desc1, GC1, nTA1, 
                sysName2, type2, scaffoldId2, begin2, end2, strand2, name2, desc2, GC2, nTA2, Sep, bOp


    Returns:
        Adds these to gene_fit_d:
            genesUsed
            strainsUsed
            genesUsed12
            gN
            t0_gN
            strains:
                used,
                enoughT0
                all_df meta_ix columns
            strain_lr
            strain_se
            [pairs]:
                adjDiff:
                    Gene1, Gene2, sysName1, type1, scaffoldId, begin1, end1, strand1, name1, desc1, GC1, 
                    nTA1, locusId, sysName2, type2, begin2, end2, strand2, name2, desc2, GC2, nTA2
                    rfit (float)
                random:
                    Gene1
                    Gene2
                    rfit
                pred:
                    Gene2, Gene1, sysName1, type1, scaffoldId1, begin1, end1, strand1, name1, desc1, GC1, nTA1, 
                    sysName2, type2, scaffoldId2, begin2, end2, strand2, name2, desc2, GC2, nTA2, Sep, bOp
                    rfit
            [cofit] (pandas DataFrame):  
                locusId (str), 
                hitId (str) 
                cofit (float)
                rank (int)
            [specphe]: (Not done)
            high (pandas DataFrame): dbg@(tmp/py_new_high_df.tsv)
                locusId, expName, fit, t, se, sdNaive, name, Group, Condition_1, Concentration_1, 
                Units_1, Media, short, u, maxFit, gMean, sysName, desc

    """

    gene_fit_d['genesUsed'] = genesUsed
    gene_fit_d['strainsUsed'] = strainsUsed
    gene_fit_d['genesUsed12'] = genesUsed12
    gene_fit_d['gN'] = all_gN
    gene_fit_d['t0_gN'] = t0_gN
   
    # Creating strains dataframe
    strains = all_df.iloc[:,0:meta_ix]
    del all_df
    strains['used'] = gene_fit_d['strainsUsed']
    strains['enoughT0'] = t0tot[t0tot > minT0Strain].mean()
    gene_fit_d['strains'] = strains

    gene_fit_d['strain_lr'] = pd.DataFrame.from_dict(
                            {x: list(GeneFitResults[x]['strain_fit']) for x in GeneFitResults.keys()}
                            )
    gene_fit_d['strain_se'] = pd.DataFrame.from_dict(
                            {x:list(GeneFitResults[x]['strain_se']) for x in GeneFitResults.keys()}
                            )
    
    gene_fit_d['strain_lrn'] = normalize_per_strain_values(strains, genes_df, gene_fit_d)

    # u_true is an int
    u_true = list(gene_fit_d['q']['u']).count(True)
    if dbg_prnt:
        print(f"u_true: {u_true}")
    if u_true > 20:
        logging.info("Computing cofitness with {u_true} experiments")
        gene_fit_d = compute_cofit(gene_fit_d, genes_df, CrudeOp_df)
    else:
        logging.info(f"Only {u_true} experiments of {gene_fit_d['q'].shape[0]} passed quality filters!")

    gene_fit_d['high'] = HighFit(gene_fit_d, genes_df, exps_df, dbg_prnt=True)

    return gene_fit_d




def normalize_per_strain_values(strains, genes_df, gene_fit_d):
    """
    Args:
        strains:
        genes_df: Dataframe of genes.GC file
        gene_fit_d:
            'g': pandas Series of locusIds (str)
            'strain_lr':
            'lrn':
            'lr':
            'strains':
                'scaffold'

    Returns:
        strain_lrn (pandas DataFrame): Normalized FitNorm values (?)
        
    """

    # strainToGene is pandas Series that has same length as num strains, 
    # and in which each index points to closest gene index by location
    strainToGene = StrainClosestGenes(strains, 
                                      genes_df.iloc[py_match(list(gene_fit_d['g']), 
                                                        list(genes_df['locusId']))].reset_index(),
                                      dbg_prnt=True)

    # Subtract every value from log ratio normalized matrix by log ratio values.
    dif_btwn_lrn_and_lr = gene_fit_d['lrn'] - gene_fit_d['lr']
    strain_lrn = create_strain_lrn(gene_fit_d['strain_lr'], 
                                   dif_btwn_lrn_and_lr,
                                   gene_fit_d, strainToGene)


    """
    """

    return strain_lrn


    
def compute_cofit(gene_fit_d, genes_df, CrudeOp_df):
    """
    Args:

        gene_fit_d: Required keys:
            'lrn'
            'q' (pandas DataFrame):
                'u': (bool)
            'g' (pandas Series):
            't' (pandas DataFrame float):
        genes_df: genes.GC pandas DataFrame

        CrudeOp_df (pandas DataFrame):
            Gene2, Gene1, sysName1, type1, scaffoldId1, begin1, end1, strand1, name1, desc1, GC1, nTA1, 
            sysName2, type2, scaffoldId2, begin2, end2, strand2, name2, desc2, GC2, nTA2, Sep, bOp




    Adds keys:
        pairs (python dict):
            adjDiff:
                Gene1, Gene2, sysName1, type1, scaffoldId, begin1, end1, strand1, name1, desc1, GC1, 
                nTA1, locusId, sysName2, type2, begin2, end2, strand2, name2, desc2, GC2, nTA2
                rfit (float)
            random:
                Gene1
                Gene2
                rfit
            pred:
                Gene2, Gene1, sysName1, type1, scaffoldId1, begin1, end1, strand1, name1, desc1, GC1, nTA1, 
                sysName2, type2, scaffoldId2, begin2, end2, strand2, name2, desc2, GC2, nTA2, Sep, bOp
                rfit
        cofit (pandas DataFrame):  
            locusId (str), 
            hitId (str) 
            cofit (float)
            rank (int)
        specphe: (Not done)
                        

    """
    adj = AdjacentPairs(genes_df)
    adjDiff = adj[adj['strand1'] != adj['strand2']]
    adjDiff['rfit'] = cor12(adjDiff, gene_fit_d['g'], gene_fit_d['lrn'][gene_fit_d['q']['u']])
    CrudeOp_df['rfit'] = cor12(CrudeOp_df, gene_fit_d['g'], gene_fit_d['lrn'][gene_fit_d['q']['u']])
    random_df = pd.DataFrame.from_dict({
                    "Gene1": gene_fit_d['g'].sample(n=len(gene_fit_d['g'])*2, replace=True), 
                    "Gene2": gene_fit_d['g'].sample(n=len(gene_fit_d['g'])*2, replace=True)
                })
    random_df = random_df[random_df['Gene1'] != random_df['Gene2']]
    random_df['rfit'] = cor12(random, gene_fit_d['g'], gene_fit_d['lrn'][gene_fit_d['q']['u']])
    gene_fit_d['pairs'] = {"adjDiff": adjDiff, 
                            "pred": CrudeOp_df, 
                            "random": random_df }
    gene_fit_d['cofit'] = TopCofit(gene_fit_d['g'], gene_fit_d['lrn'][gene_fit_d['q']['u']])
    tmp_df = gene_fit_d['q'][gene_fit_d['q']['u']].merge(exps_df, on=["name","short"])
    """
    gene_fit_d['specphe'] = SpecificPhenotypes(gene_fit_d['g'], 
                            tmp_df, gene_fit_d['lrn'][gene_fit_d['q']['u']], 
                            gene_fit_d['t'][gene_fit_d['q']['u']], dbg_prnt=True)
    """

    return gene_fit_d
    


def cor12(pairs, genes, fitnorm_df, use="p", method="pearson", names=["Gene1", "Gene2"]):
    """
    Args:
        pairs (pandas DataFrame) with the following cols:
            Gene1, Gene2, sysName1, type1, scaffoldId, begin1, end1, strand1, name1, desc1, GC1, 
            nTA1, locusId, sysName2, type2, begin2, end2, strand2, name2, desc2, GC2, nTA2
        genes (pandas Series<locusId (str)>) : gene_fit_d['g']
        fitnorm_df (pandas DataFrame all floats): dataframe with one column per setindexname ( Fitness normalized)
    """
    i1 = py_match(list(pairs[names[0]]), list(genes))
    i2 = py_match(list(pairs[names[1]]), list(genes))
    res = []
    for ix in range(pairs.shape[0]):
        if np.isnan(i1[ix]) or np.isnan(i2[ix]):
            res.append(np.nan)
        else:
            res.append(fitnorm_df.iloc[i1[x]].corr(fitnorm_df.iloc[i2[x]], method=method))

    return res


    

def create_strain_lrn(sfit, gdiff, gene_fit_d, strainToGene):
    """ We normalize per strain values?
    Args:
        sfit:  
            (comes from strain_lr) (float): dataframe with one column per setindexname
        gdiff:  dataframe (float) with one column per setindexname (same length as main_df-
                                which is equivalent to the number of unique locusIds that are used)
        strainToGene pandasSeries<index>: For each strain, the index of the closest
                                            gene center
        gene_fit_d: requires keys:
            'strains', and under this, key:
                'scaffold'
    Returns:
        pandas DataFrame
    """

    print(sfit)
    print(gdiff)
   
    results = {}
    # We iterate over every column in both dataframes sfit & gdiff
    for i in range(len(sfit.columns)):
        sfit_set_index_name = list(sfit.columns)[i]
        gdiff_set_index_name = list(gdiff.columns)[i]
        if sfit_set_index_name != gdiff_set_index_name:
            raise Exception("Columns not matching each other.")
        sfit_col = sfit[sfit_set_index_name]
        gdiff_col = gdiff[gdiff_set_index_name]
        
        # What happens here ??
        sdiffGene = gdiff_col[strainToGene]
        grouped_sfit = sfit_col.groupby(by=gene_fit_d['strains']['scaffold']).groups
        sdiffSc = [( -1*sfit_col[grouped_sfit.loc[group_label]].median() ) \
                        for group_label in grouped_sfit]
        sdiff = sdiffSc if sdiffGene is None else sdiffGene
        results[sfit_set_index_name] = sfit_col + sdiff

    return pd.DataFrame.from_dict(results)

    



    """

    # Normalized per-strain values, based on the closest gene
    strainToGene = StrainClosestGenes(fit$strains, genes[match(fit$g, genes$locusId),]);
    fit$strain_lrn = mapply(function(sfit, gdiff) {
    	# Add the relevant gene normalization; or, if NA, normalize the scaffold to a median of 0
    	sdiffGene = gdiff[strainToGene];
    	sdiffSc = -ave(sfit, fit$strains$scaffold, FUN=median);
    	sdiff = ifelse(is.na(sdiffGene), sdiffSc, sdiffGene);
    	return(sfit + sdiff);
    }, fit$strain_lr, fit$lrn-fit$lr);
    fit$strain_lrn = data.frame(fit$strain_lrn);

    """


   

def StrainClosestGenes(strains, genes, dbg_prnt=False):
    """ 
    Args:
        strains (pandas DataFrame):
            has all the meta columns from all_df and all the rows beneath them, (including 'scaffold')
            additionally contains columns:
                used: pandas Series(list<bool>): whose length is same as num of Trues in has_gene2
                enoughT0: Means of a subset of t0tots who pass the minT0 test.
        genes (pandas DataFrame): same as genes_df, but with a switched order of locusIds. 
                                  Contains same columns as genes.GC

    Intermediate Vars:
        indexSplit (python dict): group_label (scaffoldId) -> list of values (int or np.nan)
            
    Returns:
        pandas Series: Length of 'strains' (all_df), for each row of all_df, we return the index of 
                        the closest gene from 'genes', by taking halfway between each gene's beginning
                        and ending position and comparing it to the position of the strain barcode insertion.
    Description:
        For each strain (barcode in all.poolcount), find the closest gene, as a row number from genes.GC 
        returns a list, same length as strain, with corresponding strain rows -> closest row within genes
        If there is no gene on that scaffold, returns NA.

    * Below can be optimized with multithreading
    """

    genes_index = list(range(0, genes.shape[0]))
    # Are these like dicts -> lists (?)
    strainSplit = strains.groupby(by=strains['scaffold']).groups

    if dbg_prnt:
        print("strainSplit")
        print(strainSplit)
    geneSplit = genes.groupby(by=genes['scaffoldId']).groups
    if dbg_prnt:
        print("geneSplit")
        print(geneSplit)

    indexSplit = {}
    for scaffoldId in strainSplit:
        s = strains.loc[strainSplit[scaffoldId]]
        g = genes.loc[geneSplit[scaffoldId]]
        if g.shape[0] == 0:
            indexSplit[scaffoldId] = [np.nan]*len(s)
        elif g.shape[0] == 1:
            # There is a single index, and we use that.
            indexSplit[scaffoldId] = [list(geneSplit[scaffoldId])[0]] * len(s)
        else:
            # We get the centers of all the genes
            g['pos'] = (g['begin'] + g['end']) / 2

            # Now we find the location of the strain and capture the closest gene center
            # This is the part that could be multithreaded/ sorted
            crnt_scaffold_list = []
            if dbg_prnt:
                print(f"Now finding closest gene for {s.shape[0]} values")
            count = 0
            for ix, row in s.iterrows():
                if count % 5000 == 0:
                        print(f"Currently at count {count} in Strain Closest Genes for"
                              f" scaffoldId {scaffoldId}.")
                gene_pos_minus_strain_pos = (g['pos'] - row['pos']).abs()
                # we get the index of the minimum value
                crnt_scaffold_list.append(gene_pos_minus_strain_pos.idxmin())
                count += 1

            with open("tmp/py_crnt_scaffold_list.json", "w") as g:
                g.write(json.dumps([int(x) for x in crnt_scaffold_list], indent=2))

            indexSplit[scaffoldId] = crnt_scaffold_list


    
    recombined_series = py_unsplit(indexSplit, strains['scaffold']) 
    recombined_series.to_csv("tmp/py_recombined_series.tsv", sep="\t")

    return recombined_series
    

"""

# For each strain, find the closest gene, as a row number -- returns a vector
# If there is no gene on that scaffold, returns NA
StrainClosestGenes = function(strains, genes) {
	genes$index = 1:nrow(genes);
	strainSplit = split(strains, strains$scaffold);
	geneSplit = split(genes, genes$scaffold);
	indexSplit = list();
	for (sc in names(strainSplit)) {
		s = strainSplit[[sc]];
		g = geneSplit[[sc]];
		if (is.null(g)) {
		    indexSplit[[sc]] = rep(NA, nrow(s));
		} else if (nrow(g) == 1) {
		    # cannot approx with 1 value so:
		    indexSplit[[sc]] = rep(g$index[1], nrow(s));
		} else {
		    g$pos = (g$begin + g$end) / 2;
		    g = g[order(g$pos),];
		    # rule 2 means use values from extrema
		    i = round(approx(g$pos, 1:nrow(g), xout = s$pos, rule=2)$y);
		    i = pmax(1, pmin(nrow(g), i));
		    indexSplit[[sc]] = g$index[i];
		}
	}
	unsplit(indexSplit, strains$scaffold);
}
"""


def initialize_gene_fit_d(GeneFitResults, debug=False):
    """
    We create the initial version of central variable
        'gene_fit_d'. Where we essentially flip the column
        names and the set names of the dataframes, in the sense that
        we go from having a single setindex name pointing to a
        dataframe with columns indicating certain info, to the names
        of those columns pointing to a dataframe with that column's info 
        over all the different set index names.

    Args:
        GeneFitResults: (dict) setnameIndex -> ret_d
           ret_d:
               gene_fit: DataFrame, contains cols:
                    fit (float): (unnormalized
                    fitNaive (float):
                    fit1 (float):
                    fit2 (float):
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
               strain_fit: pandas Series (float) 
               strain_se: pandas Series (float) 
    Returns:
        gene_fit_d: (python dict)
            g (pandas Series (str)): pandas Series of locusIds
            lr (float): dataframe with one column per setindexname
            lrNaive (float): dataframe with one column per setindexname
            lr1 (float): dataframe with one column per setindexname
            lr2 (float): dataframe with one column per setindexname
            lrn1 (float): dataframe with one column per setindexname
            lrn2 (float): dataframe with one column per setindexname
            lrRaw (float): dataframe with one column per setindexname
            n (int): dataframe with one column per setindexname
            nEff (float): dataframe with one column per setindexname
            pseudovar (float): dataframe with one column per setindexname
            sumsq (float): dataframe with one column per setindexname
            sd (float): dataframe with one column per setindexname
            sdNaive (float): dataframe with one column per setindexname
            se (float) Standard Error dataframe with one column per setindexname
            t: (float) t-statistic dataframe with one column per setindexname
            tot1 (int or nan) dataframe with one column per setindexname
            tot1_0 (int or nan) dataframe with one column per setindexname
            tot2 (int or nan) dataframe with one column per setindexname
            tot2_0 (int or nan) dataframe with one column per setindexname
            tot (int or nan) dataframe with one column per setindexname
            tot0 (int or nan) dataframe with one column per setindexname
            version (str)
        
    """

    all_ix_names = list(GeneFitResults.keys())
    # This dict will just contain dataframes gene_fit
    fit_locusIds = GeneFitResults[all_ix_names[0]]['gene_fit']['locusId']

    # Why do we replace the name locusId with 'g'?
    gene_fit_d = {'g': fit_locusIds} 
    other_col_names = list(GeneFitResults[all_ix_names[0]]['gene_fit'].head())
    # other_col_names should be:
    #     fit, fitNaive, fit1, fit2, fitnorm1, fitnorm2, fitRaw
    #     locusId, n, nEff, pseudovar, sumsq, sd, sdNaive, se, t, tot1
    #     tot1_0, tot2, tot2_0, tot, tot0
    other_col_names.remove('locusId')
    if "Unnamed: 0" in other_col_names:
        other_col_names.remove("Unnamed: 0")
    print(other_col_names)

    st = time.time()
    for col_name in other_col_names:
        all_col_values_d = {ix_name: GeneFitResults[ix_name]['gene_fit'][col_name] for ix_name in GeneFitResults.keys()}
        gene_fit_d[col_name] = pd.DataFrame.from_dict(all_col_values_d)
    print(f"Time to create gene_fit_d: {time.time() - st}")

    new_gene_fit_d = {}
    for k in gene_fit_d.keys():
        new_key = k.replace("fitnorm","lrn")
        new_key = new_key.replace("fit", "lr")
        new_gene_fit_d[new_key] = gene_fit_d[k].copy(deep=True)

    gene_fit_d = new_gene_fit_d

    if debug:
        print("Extracted fitness values")

    gene_fit_d["version"] = "1.1.1"

    return gene_fit_d


def tmp_prep_wrap_up(all_pc_fp, genes_fp):

    dict_dtypes = {'locusId' : str,
                   'scaffoldId' : str,
                   'scaffold': str}
    all_df = pd.read_table(all_pc_fp, dtype=dict_dtypes, index_col=1)
    genes_df = pd.read_table(genes_fp,dtype=dict_dtypes)
    has_gene2 = [True if (0.1<=x<=0.9) else False for x in all_df['f']]
    return all_df, genes_df, has_gene2


def FitReadMetrics(all_df, qnames, has_gene2):
    """
    Args:
        all_df (pandas DataFrame):
        qnames (pandas Series): list<str> (names of set_index_names)
        has_gene2 list<bool>: gene insertion between 0.1 and 0.9 fraction of length 
    
    Returns:
        DataFrame with cols:
            nMapped
            nPastEnd
            nGenic

    Description:
        Compute read metrics -- nMapped, nPastEnd, nGenic, for the given data columns
        The final argument is used to define genic

    """
    print(all_df.head())

    frm_df = pd.DataFrame.from_dict({
        "nMapped": all_df[qnames].sum(axis=0),
        "nPastEnd": all_df[all_df['scaffold']=="pastEnd"][qnames].sum(axis=0),
        "nGenic": all_df[has_gene2][qnames].sum(axis=0)
        })
    frm_df.index = list(qnames)
    return frm_df

def save_gene_fit_d(gene_fit_d, prnt_dbg=False):
    for k in gene_fit_d.keys():
        if k != "version":
            gene_fit_d[k].to_csv("tmp/GENEFITD/pysave_" + k + ".tsv", sep="\t")

def FitQuality(gene_fit_d, genes_df, prnt_dbg=False):
    """
    Args:
        gene_fit_d: (python dict)
            g (pandas Series (str)): pandas Series of locusIds
            lr (float): dataframe with one column per setindexname
            lrNaive (float): dataframe with one column per setindexname
            lr1 (float): dataframe with one column per setindexname
            lr2 (float): dataframe with one column per setindexname
            lrn1 (float): dataframe with one column per setindexname
            lrn2 (float): dataframe with one column per setindexname
            fitRaw (float): dataframe with one column per setindexname
            n (int): dataframe with one column per setindexname
            nEff (float): dataframe with one column per setindexname
            pseudovar (float): dataframe with one column per setindexname
            sumsq (float): dataframe with one column per setindexname
            sd (float): dataframe with one column per setindexname
            sdNaive (float): dataframe with one column per setindexname
            se (float) Standard Error dataframe with one column per setindexname
            t: (float) t-statistic dataframe with one column per setindexname
            tot1 (int or nan) dataframe with one column per setindexname
            tot1_0 (int or nan) dataframe with one column per setindexname
            tot2 (int or nan) dataframe with one column per setindexname
            tot2_0 (int or nan) dataframe with one column per setindexname
            tot (int or nan) dataframe with one column per setindexname
            tot0 (int or nan) dataframe with one column per setindexname
            version (str)
        genes_df: 
            Dataframe of genes.GC file
        prnt_dbg: boolean
    Created:
        crudeOpGenes:
            DataFrame with cols 
                'Sep', 'bOp' - list<bool>,
                'begin1', 'end1', 'begin2', 'end2'
    
    Returns:
        fit_quality_df:
                Dataframe with cols:
                     "nUsed": 
                     "gMed": 
                     "gMedt0": 
                     "gMean": 
                     "cor12": 
                     "mad12": 
                     "mad12c": 
                     "mad12c_t0":
                     "opcor": 
                     "adjcor": 
                     "gccor":  
                     "maxFit": 
        CrudeOpGenes:
                DataFrame with cols:
                    Gene2, Gene1, sysName1, type1, scaffoldId1, begin1, end1, 
                    strand1, name1, desc1, GC1, nTA1, 
                    sysName2, type2, scaffoldId2, begin2, end2, strand2, name2, 
                    desc2, GC2, nTA2, Sep, bOp


    Description:
        Compute the quality metrics from fitness values, fitness values of halves of genes, or
        counts per gene (for genes or for halves of genes)

    """
    # crudeOpGenes is a dataframe
    crudeOpGenes = CrudeOp(genes_df)
    if prnt_dbg:
        crudeOpGenes.to_csv("tmp/py_crudeOpGenes.tsv", sep="\t")

    # adj is a dataframe
    adj = AdjacentPairs(genes_df, dbg_prnt=True)
    adjDiff = adj[adj['strand1'] != adj['strand2']]
    lrn1 = gene_fit_d['lrn1']
    lrn2 = gene_fit_d['lrn2']

    print("-*-*-*" + "Gene fit D of 'g' then genes_df['locusId'] ")
    print(gene_fit_d['g'])
    print(genes_df['locusId'])
    match_list = py_match(list(gene_fit_d['g']), list(genes_df['locusId']))
    print(match_list)

    print(len(match_list))

    #GC Correlation is the correlation between the fitnorm values and the GC values
    GC_Corr = gene_fit_d['lrn'].corrwith(genes_df['GC'].iloc[match_list], method="pearson")

    """
	adjDiff = adj[adj$strand1 != adj$strand2,];

	data.frame(
		nUsed = colSums(fit$tot),
		gMed = apply(fit$tot, 2, median),
		gMedt0 = apply(fit$tot0, 2, median),
		gMean = apply(fit$tot, 2, mean),
		cor12 = mapply(function(x,y) cor(x,y,method="s",use="p"), fit$lrn1, fit$lrn2),
		mad12 = apply(abs(fit$lrn1-fit$lrn2), 2, median, na.rm=T),
		# consistency of log2 counts for 1st and 2nd half, for sample and for time0
		mad12c = apply(abs(log2(1+fit$tot1) - log2(1+fit$tot2)), 2, median, na.rm=T),
		mad12c_t0 = apply(abs(log2(1+fit$tot1_0) - log2(1+fit$tot2_0)), 2, median, na.rm=T),
		opcor = apply(fit$lrn, 2, function(x) paircor(crudeOpGenes[crudeOpGenes$bOp,], fit$g, x, method="s")),
		adjcor = sapply(names(fit$lrn), function(x) paircor(adjDiff, fit$g, fit$lrn[[x]], method="s")),
		gccor = c( cor(fit$lrn, genes_df$GC[ match(fit$g, genes_df$locusId) ], use="p") ),
		maxFit = apply(fit$lrn,2,max,na.rm=T)
	);
        }       
    """
    # Note axis=0 means we take values from each row
    fitQuality_df = pd.DataFrame.from_dict({
        "nUsed": gene_fit_d['tot'].sum(axis=0),
        "gMed": gene_fit_d['tot'].median(axis=0),
        "gMedt0": gene_fit_d['tot0'].median(axis=0),
        "gMean": gene_fit_d['tot'].mean(axis=0),
        "cor12": [lrn1[col_name].corr(lrn2[col_name]) for col_name in lrn1.head()],
        "mad12": (lrn1-lrn2).abs().median(),
        "mad12c": (np.log2(1 + gene_fit_d['tot1']) - np.log2(1 + gene_fit_d['tot2'])).abs().median(),
        "mad12c_t0": (np.log2(1 + gene_fit_d['tot1_0']) - np.log2(1 + gene_fit_d['tot2_0'])).abs().median(),
        # Remember crudeOpGenes['bOp'] is a list of bools
        "opcor": [paircor(crudeOpGenes[crudeOpGenes['bOp']], 
                  gene_fit_d['g'], 
                  gene_fit_d['lrn'][colname], 
                  method="spearman",
                  dbg_prnt=True) for colname in gene_fit_d['lrn']], 
        "adjcor": [paircor(adjDiff, gene_fit_d['g'], gene_fit_d['lrn'][colname], method="spearman", dbg_prnt=True)\
                    for colname in gene_fit_d['lrn']],
        "gccor":  GC_Corr,
        "maxFit": gene_fit_d['lrn'].max()
        })
   

    if prnt_dbg:
        fitQuality_df.to_csv("tmp/py_fitQuality_df.tsv", sep="\t")

    return fitQuality_df, crudeOpGenes

def paircor(pairs, locusIds, values, use="p", method="pearson", names=["Gene1","Gene2"],
            dbg_prnt=False):
    """
    pairs (pandas DataFrame): dataframe with multiple cols (CrudeOp with TRUE cols from bOp)
    locusIds (pandas Series (str)): locusIds 
    values (pandas Series): normalized fitness scores 
    use: 
    method: Correlation method ("pearson", "spearman")
    names (list<str>): "Gene1", "Gene2"
    dbg_prnt (bool)

    """
    if dbg_prnt:
        print(f"Length of locusIds: {len(locusIds)}")
        if len(locusIds) > 10:
            print(f"First ten locusIds: {locusIds[:10]}")
        print(f"Length of values: {len(values)}")
        if len(values) > 10:
            print(f"First ten values: {values[:10]}")

    premrg1 = pd.DataFrame.from_dict({
                "Gene1": list(locusIds),
                "value1": list(values)
                })

    if dbg_prnt:
        print('premrg1')
        print(premrg1)
    mrg1 = pairs[names].merge(premrg1, left_on=names[0], right_on="Gene1")

    if dbg_prnt:
        print('mrg1')
        print(mrg1)

    premrg2 = pd.DataFrame.from_dict({
                "Gene2": list(locusIds),
                "value2": list(values)
                })

    if dbg_prnt:
        print('premrg2')
        print(premrg2)

    mrg2 = mrg1.merge(premrg2, left_on=names[1], right_on="Gene2")

    if dbg_prnt:
        print('mrg2')
        print(mrg2)


    # method can be spearman or pearson
    res = mrg2['value1'].corr(mrg2['value2'], method=method)


    if dbg_prnt:
        print('res')
        print(res)

    return res 
    


def FEBA_Exp_Status(inp_df, min_gMed=50, max_mad12=0.5, min_cor12=0.1,
                    max_gccor=0.2, max_adjcor=0.25, dbg_prnt=False):
    """
    inp_df: A dataframe with cols:
            nMapped (from FitReadMetrics)
            nPastEnd (from FitReadMetrics)
            nGenic (from FitReadMetrics)
            "nUsed": (from FitQuality)
            "gMed": (from FitQuality)
            "gMedt0": (from FitQuality)
            "gMean": (from FitQuality)
            "cor12": (from FitQuality)
            "mad12": (from FitQuality)
            "mad12c": (from FitQuality)
            "mad12c_t0": (from FitQuality)
            "opcor": (from FitQuality)
            "adjcor": (from FitQuality)
            "gccor":  (from FitQuality)
            "maxFit": (from FitQuality)
            "name": (from exps_df)
            "short": (from exps_df)
            "t0set": (from exps_df)
            ["num"]: (from_exps_df)
            indexes are:
       
    Returns:
        status_list (pandas Series(list<str>)): each status is from: {"OK", "Time0", "low_count", "high_mad12", 
                                                    "low_cor12", "high_adj_gc_cor"}
                                And each status corresponds to one experiment in inp_df (each row)
    Description:
        # Returns status of each experiment -- "OK" is a non-Time0 experiment that passes all quality metrics
        # Note -- arguably min_cor12 should be based on linear correlation not Spearman.
        # 0.1 threshold was chosen based on Marinobacter set5, in which defined media experiments with cor12 = 0.1-0.2
        # clearly worked, and Kang Polymyxin B (set1), with cor12 ~= 0.13 and they barely worked.
    """

    if dbg_prnt:
        print(inp_df.columns)
        print(inp_df.shape[0])
        print(inp_df.index)

    status_list = []
    # Each row corresponds to one experiment
    for ix, row in inp_df.iterrows():
        if row["short"] == "Time0":
            status_list.append("Time0")
        elif row["gMed"] < min_gMed:
            status_list.append("low_count")
        elif row["mad12"] > max_mad12:
            status_list.append("high_mad12")
        elif row["cor12"] < min_cor12:
            status_list.append("low_cor12")
        elif abs(row["gccor"]) > max_gccor or abs(row["adjcor"]) > max_adjcor:
            status_list.append("high_adj_gc_cor")
        else:
            status_list.append("OK")

    if dbg_prnt:
        print("FEBA_Exp_Status: status_list:")
        print(status_list)

    return pd.Series(data=status_list, index=inp_df.index) 

def SpecificPhenotypes(locusIds, exps_df, fitnorm_df, t_score_df,
                        minT=5, minFit=1.0, percentile=0.95,
                        percentileFit=1.0, minDelta=0.5,
                        dbg_prnt=False):
    """
    Args:
        locusIds (pandas Series <str>)
        exps_df (pandas DataFrame): Entire edited FEBA.BarSeq dataframe
        fitnorm_df (pandas DataFrame (float)): length is unique applicable locusId 
        t_score_df (pandas DataFrame (float)): Does this and above dataframe have the 
                                                                exact same dimensions?



    Description:
        Identify "specific phenotypes" -- cases where a gene is sick
        in some experiment(s), with |fit| > minFit and |fit| > percentileFit + minDelta and 
                                                                                    |t| > minT
        percentileFit is defined as the 95th percentile (by default) of |fit| for that gene
        
        exps ideally includes name (the column names of lrn and t_score_df) along with
        short, Group, Condition_1, Concentration_1, Units_1, Condition_2, Concentration_2, Units_2
        
        Returns a data frame of locusId, fit, t, name, short, etc.

    Returns:
        Why return (?) - understand usage
    """
    expsFields = set(exps_df.columns).intersection(set(["name", "short", "Group", "Condition_1",
                                                        "Concentration_1", "Units_1", "Condition_2",
                                                        "Concentration_2", "Units_2", "Condition_3",
                                                        "Concentration_3", "Units_3", "Condition_4",
                                                        "Concentration_4", "Units_4"]))

    # getting the 95th percent quantile over the rows of the absolute values of the dataframe
    rowHi = fitnorm_df.abs().quantile(q=percentile)
    # Does this test over every element of the dataframe? Are t_score_df and fitnorm_df the exact
    # same dimensions (?)
    if dbg_prnt:
        print("Dimensions of fitnorm and then t_score_df:")
        print(f"{fitnorm_df.shape[0]}, {fitnorm_df.shape[1]}")
        print(f"{t_score_df.shape[0]}, {t_score_df.shape[1]}")
        print("Dimensions of rowHi:")
        print(f"{rowHi.shape[0]}, {rowHi.shape[1]}")
        print("Type of rowHi:")
        print(type(rowHi))


    fnabs = fitnorm_df.abs()
    rowHi_bool = bool(rowHi < percentileFit)

    which_pass_list = []
    # We find <row, col> locations that pass thresholds
    for row_ix in range(fitnorm_df.shape[0]):
        for col_ix in range(fitnorm_df.shape[1]):
            if (fnabs.iloc[row_ix, col_ix] > minFit and \
                fnabs.iloc[row_ix, col_ix] > rowHi + minDelta and 
                rowHi_bool and \
                t_score_df.abs().iloc[row_ix, col_ix] > minT):
                which_pass_list.append([row_ix, col_ix])

    # sp - specific
    sp_locId = locusIds.iloc[[x[0] for x in which_pass_list]]
    return None
    """
        SpecificPhenotypes = function(locusIds, exps_df, lrn, t_score_df,
        	   	    minT = 5, minFit = 1.0,
        		    percentile = 0.95, percentileFit = 1.0, minDelta = 0.5,
        		    expsFields = intersect(names(exps_df),
        		         words("name short Group Condition_1 Concentration_1 Units_1 Condition_2 Concentration_2 Units_2 Condition_3 Concentration_3 Units_3 Condition_4 Concentration_4 Units_4")))
        {
        	rowHi = apply(abs(lrn), 1, quantile, percentile);
        	bool = abs(lrn) > minFit & abs(lrn) > rowHi+minDelta & rowHi < percentileFit & abs(t_score_df) > minT;
                # arr.in or arr.ind (?)
        	specsick = data.frame(which(bool, arr.in=T));
        	specsick$locusId = locusIds[specsick$row];
        	specsick$name = names(lrn)[specsick$col];
        	specsick$lrn = as.matrix(lrn)[cbind(specsick$row,specsick$col)];
        	specsick$t = as.matrix(t_score_df)[cbind(specsick$row,specsick$col)];
        	specsick$row = NULL;
        	specsick$col = NULL;
        	return(merge(specsick, exps_df[,expsFields]));
        }
    """



def AdjacentPairs(genes_df, dbg_prnt=False):
    """
    Args:
        genes_df pandas DataFrame of genes.GC tsv
    Returns:
        DataFrame with the following cols:
            Gene1, Gene2, sysName1, type1, scaffoldId, begin1, end1, strand1, name1, desc1, GC1, 
            nTA1, locusId, sysName2, type2, begin2, end2, strand2, name2, desc2, GC2, nTA2
    """
    # get genes in order of scaffoldId and then tiebreaking with increasing begin
    c_genes_df = genes_df.copy(deep=True).sort_values(by=['scaffoldId', 'begin'])

    # We offset the genes with a loop starting at the first
    adj = pd.DataFrame.from_dict({
            "Gene1": list(c_genes_df['locusId']),
            "Gene2": list(c_genes_df['locusId'].iloc[1:]) + [c_genes_df['locusId'].iloc[0]]
        })

    adj.to_csv("tmp/py_preAdj1.tsv", sep="\t")

    c_genes_df = c_genes_df.rename(columns={"locusId": "Gene1"})

    mg1 = adj.merge(c_genes_df, left_on="Gene1", right_on="Gene1")
    if dbg_prnt:
        mg1.to_csv("tmp/py_preAdj2.tsv", sep="\t")

    c_genes_df = c_genes_df.rename(columns={"Gene1":"locusId"})
    # add metadata and only keep pairs with same scaffold
    adj = mg1.merge(c_genes_df,
                    left_on=["Gene2", "scaffoldId"],
                    right_on=["locusId", "scaffoldId"],
                    suffixes = ["1","2"]
                    )
    
    if dbg_prnt:
        adj.to_csv("tmp/py_AdjacentPairsOutput.tsv", sep="\t")

    return adj


def TopCofit(locusIds, lrn, dbg=False, fraction=0.02):
    """
    Args:
        g is genes (i.e., locusIds)
        lrn is a matrix of fitness values with columns set name index 

    Returns:
        out_df (pandas DataFrame): has columns:
            locusId (str), 
            hitId (str) 
            cofit (float)
            rank (int)
    """
    n = min( max(1, math.round(len(locusIds) * fraction)) , len(locusIds) - 1)

    if dbg:
        print(f"n: {n}")

    # Number of locusIds must match number of rows in lrn
    if len(locusIds) != lrn.shape[0]:
        raise Exception("Number of genes and number of rows in matrix do not match.")
    
    # We transpose the matrix lrn
    cofits = lrn.transpose().corr(method="pearson")
    if dbg:
        print("type of cofits:")
        print(type(cofits))
        print("shapes of cofits 0, 1")
        print(f"{cofits.shape[0]}, {cofits.shape[1]}")
    
    nOut = len(locusIds)*n

    if dbg:
        print(f"Making output with {nOut} rows")

    out_hitId = [""]*nOut
    out_cofit = [np.nan]*nOut
    
    for i in range(len(locusIds)):
        values = cofits.iloc[i,:]
        j = py_order(list(values*-1))[1:n]
        outi = (i-1)*n + list(range(n)) # where to put inside out
        out_hitId[outi] = locusIds[j];
        out_cofit[outi] = values[j];

    lI_list = []
    rank = []
    for i in range(len(locusIds)):
        lI_list += [locusIds[i]]*n
        rank += list(range(n))
    
    out_df = pd.DataFrame.from_dict({
        "locusId": lI_list,
        "hitId": out_hitId,
        "cofit": out_cofit,
        "rank": rank
        })

    return(out_df)


def HighFit(gene_fit_d, genes_df, exps_df, min_fit=4, min_t=5, max_se=2, 
            min_gMean=10,max_below=8,dbg_prnt=False):
    """
    Args:
       gene_fit_d (python dict):
            lrn: pandas DataFrame (one col per setindexname) floats (fitness?)
            t (t-score): pandas DataFrame (one col per setindexname) floats (t_score?)
            u (used?): pandasDataFrame (one col per setindexname) floats

    Description:
        We find the [row, col] indexes where the 'lrn' and 't' dataframes (fitness and 
        t score dataframes) have values that pass the thresholds of minimum fitness and 
        minimum t score (parameters min_fit and min_t). We create a new dataframe called
        'high_df' which contains the locusId, experiment name, fitness score and t scores
        where these thresholds are passed. The number of rows in these dataframes is equal
        to the number of locations where the thresholds are passed, and there are doubled
        locusIds and expNames.

    Returns:
        new_high (pandas DataFrame):
            locusId, expName, fit, t, se, sdNaive, name, Group, Condition_1, Concentration_1, Units_1, Media, short, u, maxFit, gMean, sysName, desc
            
        
    """
    lrn = gene_fit_d['lrn']
    t = gene_fit_d['t']
    u = gene_fit_d['q']['u']
    
    # This needs to be two columns: 1 with rows and 1 with columns
    num_rows, num_cols = lrn.shape[0], lrn.shape[1]
    # where is high is a list of [row (int), col(int)] (coming from dataframe, so it's a list whose length
    # is the length of (m x j) for rows and columns in the dataframe.
    where_is_high = []
    for i in range(num_rows):
        for j in range(num_cols):
            if lrn.iloc[i,j] >= min_fit and t.iloc[i,j] >= min_t:
                where_is_high.append([i,j])

    high_df = pd.DataFrame.from_dict({
                    # x[0] -> rows from where_is_high
                    "locusId": gene_fit_d['g'].iloc[[x[0] for x in where_is_high]],
                    # x[1] -> columns from where_is_high
                    "expName": (lrn.iloc[:,[x[1] for x in where_is_high]]).columns,
                    "fit": [lrn.iloc[x[0], x[1]] for x in where_is_high],
                    "t": [t.iloc[x[0], x[1]] for x in where_is_high],
                })

    high_df['se'] = high_df['fit']/high_df['t']
    high_df['sdNaive'] = [gene_fit_d['sdNaive'].iloc[x[0], x[1]] for x in where_is_high]
    high_df = high_df[high_df['se'] <= max_se]

    # Which experiments are ok
    fields = "name Group Condition_1 Concentration_1 Units_1 Media short".split(" ")
    fields = [x for x in fields if x in exps_df.columns]
    crnt_exps = exps_df[fields]
    crnt_exps = crnt_exps.merge(gene_fit_d['q'][["name","u","short","maxFit","gMean"]])
    new_high = high_df.merge(crnt_exps, left_on="expName", right_on="name")
    check_bool = [bool(new_high['gMean'].iloc[ix] >= min_gMean and \
                  new_high['fit'].iloc[ix] >= new_high['maxFit'].iloc[ix] - max_below) \
                  for ix, val in new_high['gMean'].items()]
    new_high = new_high[check_bool]
    new_high = new_high.merge(genes_df[["locusId","sysName","desc"]])
    new_high = new_high.iloc[py_order(list(high_df['expName']), tie_breaker=list(-1*high_df['fit']))]
    
    if dbg_prnt:
        new_high.to_csv("tmp/py_new_high_df.tsv", sep="\t", index=False)

    return new_high
    
    """
    # Note thresholds are different than in high_fit.pl
    HighFit = function(fit, genes, expsUsed, min.fit=4, min.t=5, max.se=2, min.gMean=10, max.below=8) {
    
    # wHigh is a dataframe with two columns, one called 'rows', and one called 'columns'
      wHigh = which(fit$lrn >= min.fit & fit$t >= min.t, arr.ind=T);
      high = data.frame(locusId=fit$g[wHigh[,1]], expName=names(fit$lrn)[wHigh[,2]], fit=fit$lrn[wHigh], t=fit$t[wHigh]);
      # t ~= fit/standard_error, so estimate s.e. = fit/t
      high$se = high$fit/high$t;
      high$sdNaive = fit$sdNaive[wHigh];
      high = subset(high, se <= max.se);
    
      # which experiments are ok
      fields = words("name Group Condition_1 Concentration_1 Units_1 Media short");
      fields = fields[fields %in% names(expsUsed)];
      exps = expsUsed[, fields];
      exps = merge(exps, fit$q[,words("name u short maxFit gMean")]);
      high = merge(high, exps, by.x="expName", by.y="name");
      high = subset(high, gMean >= min.gMean & fit >= maxFit - max.below);
      names(high)[names(high)=="u"] = "used";
      high = merge(genes[,c("locusId","sysName","desc")], high);
      high = high[order(high$expName, -high$fit),];
      return(high);
    }
    """


def getGenesPerScaffold(genes_df, genesUsed):
    """
    Args:
        genes_df: Dataframe of genes.GC
        genesUsed: list<locusId (str)>
    Returns:
        genesPerScaffold:
            genesPerScaffold is a dict with scaffold -> number of genes found in that scaffold
            function py_table comes from file 'translate_R_to_pandas'
    """

    #We iterate over every row of genes_df and find locations of genesUsed locusIds
    rows_with_locus_Ids_in_genesUsed_bool = [genes_df['locusId'][i] in genesUsed \
                                    for i in range(len(genes_df['locusId']))]

    genesPerScaffold = py_table(list(genes_df['scaffoldId'][rows_with_locus_Ids_in_genesUsed_bool]
                                    ))

    return genesPerScaffold


def check_if_every_t0set_is_in_t0tot(exps_df, t0tot):
    """
    Args:
        exps_df:
            Dataframe of FEBABarSeq.tsv
        t0tot: data frame where column names are 'date setname'
                and linked to a list of sums over the indexes that relate
                to that setname, with the list length being equal to the
                total number of strains (barcodes) in all.poolcount
                all columns are t0's?
    """

    # We check if every t0set is in t0tot
    #{datesetname:[] for datesetname in expsT0.keys()}
    incorrect_sets = []
    for t0set in exps_df['t0set'].array:
        if t0set not in t0tot.head():
            incorrect_sets.append(t0set)

    if len(incorrect_sets) > 0:
        raise Exception("incorrect t0sets: \n" + ", ".join(incorrect_sets))



def get_GenesUsed12(genesUsed12, minT0Gene, strainsUsed, all_df,
                    t0tot):
    """
    We get the locusIds which have insertions under 0.5 and over
        0.5 within the gene (percentage of length) and with values
        over the minT0Gene
    Args:
        genesUsed12: None or list<locusId (str)>
        minT0Gene: int
        strainsUsed: list<bool>
        all_df: Dataframe needs col (f)
        t0tot: data frame where column names are 'date setname'
                and linked to a list of sums over the indexes that relate
                to that setname, with the list length being equal to the
                total number of strains (barcodes) in all.poolcount
                all columns are t0's?
    Returns:
        genesUsed12: list of locusIds that have both high f (>0.5) and low f (<0.5)
                    insertions with enough abundance of insertions on both sides
    """

    if genesUsed12 is None:
        minT0GeneSide = minT0Gene/2

        # d1t0tot captures t0tot whose strains have f < 0.5 and True in strainsUsed
        stUsed_and_f_low = [strainsUsed[i] and all_df['f'].iloc[i] < 0.5 for i \
                                in range(len(strainsUsed))]

        d1, d1_row_min_bool = get_non_locusIdSumsForGene12(minT0GeneSide, t0tot, all_df, 
                                                           stUsed_and_f_low)

        # d2t0tot captures t0tot whose strains have f >= 0.5 and True in strainsUsed
        stUsed_and_f_high = [strainsUsed[i] and all_df['f'].iloc[i] >= 0.5 for i 
                                in range(len(strainsUsed))]

        d2, d2_row_min_bool = get_non_locusIdSumsForGene12(minT0GeneSide, t0tot, all_df, 
                                                           stUsed_and_f_high)

        genesUsed12 = list(
                          set(d1['locusId'][d1_row_min_bool]).intersection(
                          set(d2['locusId'][d2_row_min_bool]))
                      )

        # Should the counts for each half of the gene (d1,d2) be saved as a diagnostic?
        # t0_gN should be enough for now
        if (len(genesUsed12) < 100):
            raise Exception(
                    f"Length of genesUsed12 is less than 100. Value: {len(genesUsed12)}"
                    )

    return genesUsed12

def get_non_locusIdSumsForGene12(minT0GeneSide, t0tot, all_df, stUsed_and_good_f):
    """

    Args:
        minT0GeneSide (int): int 
        t0tot (pandas DataFrame): DataFrame of t0 aggregates
        all_df (pandas DataFrame):
        stUsed_and_good_f list(bool): A list of length all_df and t0tot (which are equivalent
                                      in the number of rows they have), which indicates
                                      which strains we care about now.

    Returns:
        crt:
        crt_row_min_bool:
    """
    crtt0tot = t0tot[stUsed_and_good_f]
    crtt0tot['locusId'] = all_df['locusId'][stUsed_and_good_f]
    crt = py_aggregate(crtt0tot,
                      'locusId',
                      'sum')

    crt_mins = crt.loc[:, crt.columns != 'locusId'].min(axis=1)
    #print(crt_mins)
    crt_row_min_bool = [x >= minT0GeneSide for x in list(crt_mins)]

    return crt, crt_row_min_bool



def print_info2(has_gene2, all_df, strainsUsed, genesUsed):
    """
    Args:
        has_gene2: list<bool>
        all_df: DataFrame of all.poolcount
        strainsUsed: list<bool>
        genesUsed: list<locusId (str)>
    """
    
    # We count the number of Trues in has_gene2
    num_true_has_gene2 = has_gene2.count(True)

    num_unique_locus_Ids = len(all_df['locusId'][has_gene2].unique())

    logging.info(f"Using {str(len(strainsUsed))} of {num_true_has_gene2} genic strains.")
    logging.info(f"Using {len(genesUsed)} of {num_unique_locus_Ids} genes with data.")

    return None



def remove_genes_if_not_in_genes_df(genesUsed_list, genes_df):
    """
    We currently check if a single gene from genesUsed_list is in genes_df; 
    we also return a list of all genes that Aren't in genes_df
    Args:
        genesUsed_list: list<locusId (str)>
        genes_df: Dataframe of genes.GC file (~12 columns)
    Returns:
        genesUsed_list: list<locusId (str)>
        genes_in_genes_df_bool: boolean which says if there is a gene in genesUsed_list
            which is also in genes_in_genes_df_bool
    """
    genes_in_genes_df_bool = True
    all_genes_locus_id = list(genes_df['locusId'])
    genes_not_in_genes_df = []
    for x in genesUsed_list:
        if x not in all_genes_locus_id:
            genes_not_in_genes_df.append(x)
            genesUsed_list.remove(x)


    if len(genesUsed_list) < 10 or (not genes_in_genes_df_bool):
        logging.info("genesUsed_list")
        logging.info(genesUsed_list)
        raise Exception(f"Less than 10 genes left, exiting program: {len(genesUsed_list)}")
    
    if len(genes_not_in_genes_df) > 0:
        logging.critical("Gene Locus Ids not in the genes.GC file: \n"
                        ", ".join(genes_not_in_genes_df) + "\n")

    return genesUsed_list 




def get_smallScaffold(genesPerScaffold, minGenesPerScaffold, genes_df, 
                      debug_print_bool=False):
    """
    Args:
        genesPerScaffold: dict scaffold -> number of genes in that scaffold
        minGenesPerScaffold: int
        genes_df: dataframe of genes.GC
   
    Returns:
        smallScaffold: list<scaffold_name (str)> whose number of genes
            in the scaffold is less than minGenesPerScaffold (the minimum)
        smallLocusIds: list<locusId str> All LocusIds related to scaffolds in smallScaffold
    """

    # This is a list of scaffold Names (str) whose gene number is too low 
    smallScaffold = []
    for k, v in enumerate(genesPerScaffold):
        if v < minGenesPerScaffold:
            smallScaffold.append(k)

    if debug_print_bool:
        debug_print(smallScaffold, 'smallScaffold')



    if len(smallScaffold) > 0:
        logging.info("Ignoring genes on small scaffolds "
                     ", ".join(smallScaffold) + " " + \
                     "\ngenes left: " + str(len(genesUsed)) + "\n");

    smallLocus_Ids = []
    for index, row in genes_df.iterrows():
        current_scaffold = row['scaffoldId']
        current_locus_id = row['locusId']
        if current_scaffold in smallScaffold:
            smallLocus_Ids.append(current_locus_id)

    return smallScaffold, smallLocus_Ids


def getGenesUsed(t0tot, strainsUsed, all_df, minT0Gene, genesUsed,
                 debug_print_bool=False):
    """ We create the variable genesUsed
    Args:
        t0tot: A Dataframe which contains datesetname: [sum1, sum2, 
                    ...] for datesetname in expsT0.keys(),
                i.e. A dataframe with timezeros datesetnames
                The number of rows in the data frame is equal
                to the number of rows in all_df.
                Does not contain cols besides datesetnames.
                Contains sum over all samples that match into a datesetname
                that is a 'Time0'
        strainsUsed: list<bool> length of which is the same as all_df and t0tot
        all_df: needs col locusId
        minT0Gene: (int) 
        genesUsed: either None or a list of locusIds to be used
    Returns:
        genesUsed: list of unique locusIds such that their mean Time0 values
                    is greater than minT0Gene

    Description:
        We take the t0tot (Time0 totals), we take the strainsUsed from that
            and add a related column with locusIds from all_df.
        Then we sum these up over the locusIds, so the number of rows 
        in t0_gN_used will be the same as the total number of unique
        locusIds in unique_usable_locus_ids

    """

    # genesUsed is either None or a list of locusIds to be used
    pre_t0_gn_used = t0tot[strainsUsed]
    pre_t0_gn_used['locusId'] = list(all_df['locusId'][strainsUsed])

    if genesUsed is None:
        # t0_gN_used is  
        t0_gN_used = py_aggregate(pre_t0_gn_used, 
                                  'locusId',
                                  func='sum'
                                 )
        if debug_print_bool:
            t0_gN_used.to_csv("tmp/py_t0_gN_used.tsv", index=False, sep="\t")
        # n0 is a pandas series (?) with the means of rows in t0_gN_used which are sums over
        # 
        n0 = t0_gN_used.iloc[:,t0_gN_used.columns != 'locusId'].mean(axis=1)
        logging.info(f"Time0 reads per gene: mean {statistics.mean(n0)}"
                     f"median: {statistics.median(n0)} "
                     f" ratio: {statistics.mean(n0)}/{statistics.median(n0)}")

        # Below is boolean list of locations where the row mean passes minT0Gene
        genesUsedpre = [(n0.iloc[i] >= minT0Gene) for i in range(n0.shape[0])]
        #print(genesUsedpre[:100])
        genesUsed = t0_gN_used['locusId'][genesUsedpre]
        if debug_print_bool:
            genesUsed.to_csv("tmp/py_genesUsed.tsv", sep="\t")


    return genesUsed


def createStrainsUsed(t0tot, minT0Strain, has_gene2, strainsUsed):
    """ Create the variable strainsUsed - uses existing var if not None
    We make strainsUsed a list which contains True or False values for 
        each strain in all_df such that both the strain has an insertion
        centrally in a gene (meaning .1<f<.9) AND that the average number 
        of insertions over the t0 totals is greater than the integer minT0Strain.

    Args:
        t0tot: A Dataframe which contains datesetname: [sum1, sum2, 
                    ...] for datesetname in expsT0.keys(),
                e.g. A dataframe with timezeros datesetnames
                The number of rows in the data frame is equal
                to the number of rows in all_df
                Does not contain cols besides datesetnames
        minT0Strain: int, minimum mean value for total number of
                    barcodes read for a sample name.
        has_gene2: A pandas series of booleans the length 
                       of all_df which marks which strains have
                       insertions in the central 80% of a gene
        strainsUsed: either list of booleans or None
    Returns:
        strainsUsed: list of boolean the length of total number of strains in all_df
    """


    # strainsUsed will be a list of booleans with length being
    # total number of strains.
    if strainsUsed is None: 
        strainsUsed = []
        for i in range(len(has_gene2)):
            if has_gene2[i] and t0tot.iloc[i,:].mean() >= minT0Strain:
                strainsUsed.append(True)
            else:
                strainsUsed.append(False)
    else: 
        strainsUsed = [bool(strainsUsed.iloc[i] and has_gene2[i]) for i in range(len(has_gene2))]



    return strainsUsed



def print_log_info1(t0tot, t0_gN):
    """
    Args:
        t0tot:
        t0_gN:
    """

    logging.info("Central Reads per t0set, in millions:\n")
    # We iterate over the set names
    for k in t0tot.keys():
        try:
            logging.info(f"{k}: {t0_gN[k].sum()/e6:.2f}")
        except Exception:
            logging.info(f"Couldn't print value for key {k}")



def createt0gN(t0tot, has_gene2, indexBy, debug_print_bool=False):
    """
    We take the t0tot (time 0 totals) dataframe, and group it
        by the locusIds of genes which have insertions in their
        central 80%.
    Args:
        t0tot: A Dataframe which contains datesetname: [sum1, sum2, 
                    ...] for datesetname in expsT0.keys(),
                Summed over all_df setname.index which relates
                to a datesetname.
                i.e., A dataframe with timezeros datesetnames
                The number of rows in the data frame is equal
                to the number of rows in all_df.
                Does not contain cols besides datesetnames
        has_gene2: A pandas series of booleans the length 
                   of all_df which marks which strains have
                   insertions in the central 80% of a gene
        indexBy: panda Series with all the locusIds which
            have insertions in the important regions
            it's length should be the same length as the
            number of Trues in has_gene2 - locusIds are not unique 
    Returns:
        t0gN:
            A dataframe with the same number of columns
            as t0tot + 1 for locusIds. Row number is variable-
            grouped by the number of unique locusIds in indexBy.
            It's length should be the same length as the number of 
            unique locusIds

    """

    t0_gN = t0tot[has_gene2]
    t0_gN['locusId'] = indexBy
    
    t0_gN = t0_gN.groupby(["locusId"], as_index=False).sum()

    if debug_print_bool: 
        t0_gN.to_csv("tmp/py_t0_gN.tsv", index=False, sep="\t")

    return t0_gN

def createIndexBy(all_df, has_gene2, print_bool=False, 
                  stop_bool=False):
    """
    indexBy is a panda Series of all the locusIds which
        have insertions in the important regions (keeps indexes)
    Args:
        all_df: Dataframe of all.poolcount
        has_gene2: A pandas series of booleans the length 
                   of all_df which marks which strains have
                   insertions in the central 80% of a gene
    Returns:
        indexBy: panda Series with all the locusIds which
            have insertions in the important regions
            it's length should be the same length as the
            number of Trues in has_gene2 - comes from
            all_df. Note- locusIds are NOT unique.
    """

    # All the locusIds which include insertions in the important regions
    indexBy = all_df['locusId'][has_gene2]
    if print_bool:
        debug_print(indexBy, 'indexBy')
    if stop_bool:
        raise Exception("Stopping for debug")
    return indexBy


def create_t0tot(expsT0, all_df, dbg_prnt=False):
    """
    Args:
        expsT0: dict mapping t0set name 'date' - > pandas Series (<set+Index (str) that's related>)
            for every actual Time0 name, where set+Index is a column name in all_df
        all_df:
            Dataframe of all.poolcount with edited setindex names

    Returns:
        t0tot: A Dataframe which contains datesetname mapped to [sum1, sum2, 
                    ... sum-n] for datesetname in expsT0.keys(), where n is the number
                    of strains in all.poolcount
                Summed over all_df setname.index which relates
                to a datesetname.
                i.e., A dataframe with timezeros datesetnames
                The number of rows in the data frame is equal
                to the number of rows in all_df.
                Does not contain cols besides datesetnames

        
    """

    # We prepare to sum the values for all the pertinent setname-indexes for each datesetname
    # in expsT0.keys
    t0tot = {} #{date: pd_series([sum1, sum2, ...]) for date in expsT0.keys()}
    for date, pd_series in expsT0.items():
        t0tot[date] = all_df[list(pd_series)].sum(axis=1)

    # We recreate t0tot as a DataFrame
    t0tot = pd.DataFrame.from_dict(t0tot)

    if dbg_prnt:
        t0tot.to_csv("tmp/py_t0tot.tsv", sep= "\t")

    return t0tot



def update_expsT0_and_exps_df_with_nont0sets(expsT0, exps_df, okLane, okDay,
                              print_bool=False, dbgp=False):
    """
    Args:
        expsT0: dict mapping t0set name 'date setName' - > list<set+Index (str) that's related>
            for every actual Time0 name
        exps_df: dataframe of exps file with additional col headers. Requires:
                    t0set, Date_pool_expt_started, SetName, short for this function
        okLane: bool
        okDay: bool
        print_bool: to print all the vars


        nont0sets: list of exps_df 't0set' values that don't have 'Time0' as their 'short',
                   

    Returns:
        exps_df: (Updated t0set col to just be date instead of date + setname)
        expsT0: (Updated keys to just be date instead of date + setname) 
            updated values to be pandas Series with indeces


    Description:
        Gets a list of t0set values (date setname) which don't have 'Time0' as their short,
            and it iterates through them. For each of those sets, it removes the setname
            from the date
        Updates exps_df['t0set'] column.
        For each nont0set, 
        
    """

    if dbgp:
        print("A1 Original exps_df t0set:")
        print(exps_df['t0set'])
        print("A1 Original expsT0:")
        debug_print(expsT0, 'expsT0')

    # nont0sets is a list of str date + setname
    nont0sets = get_nont0_sets(exps_df, debug_print_bool=True)

    if print_bool:
        with open("tmp/py_nont0sets.json", "w") as g:
            g.write(json.dumps(nont0sets, indent=2))
        debug_print(exps_df['t0set'], 'expsdf_t0set')

    for datesetname in nont0sets:
        # Each datesetname is '{date} {setName}'
        if dbgp:
            print(f"Current datesetname: {datesetname}")

        # u is a list of bools that matches datesetnames to label where t0set is this one.
        u = exps_df['t0set'] == datesetname
        if print_bool:
            debug_print(u, "u") 

        # This should be a list of length 1
        date_list = list(exps_df[u]['Date_pool_expt_started'].unique())
        if len(date_list) > 1:
            raise Exception("Multiple different dates associated with a single datesetname:\n"
                            ",\n".join(date_list))
        elif len(date_list) == 0:
            raise Exception(f"No date associated with nont0set date+setname value '{datesetname}'")
        else:
            associated_date = date_list[0]

        if print_bool:
            debug_print(associated_date, "associated_date")

        # unique set names over current datesetname 
        unique_applicable_set_names = list(exps_df[u]['SetName'].unique())
        if len(unique_applicable_set_names) > 0:
            current_setname = unique_applicable_set_names[0]
        else:
            raise Exception("No SetName associated with date setname value: {datesetname}")


        t0_date_vals = exps_df[exps_df['Date_pool_expt_started'] == associated_date][exps_df['short'].str.upper() == "TIME0"]
        t0_setName_vals = exps_df[exps_df['SetName'] == current_setname][exps_df['short'].str.upper() == "TIME0"]

        if okLane and t0_date_vals.shape[0] > 0:
            del expsT0[datesetname]
            logging.info(f"Using Time0 from other lanes instead for {datesetname}")
            logging.info("Experiments affected:\n" + "\n".join(list(exps_df['name'][u])))
            for ix in range(len(u)):
                if u.iat[ix]:
                    exps_df['t0set'].iat[ix] = associated_date
            expsT0[associated_date] = exps_df['name'][exps_df['Date_pool_expt_started'] == associated_date][exps_df['short'].str.upper() == "TIME0"]
        elif (okDay and t0_setName_vals.shape[0] > 0 ):
            del expsT0[datesetname]
            newt0sets = t0_setName_vals['t0set']
            newt0set = newt0sets.iloc[0]
            logging.info(f"Warning! Using Time0 from other days instead for {datesetname}")
            logging.info("Experiments affected:\n " + "\n".join(list(exps_df['name'][u])))
            for ix in range(len(u)):
                if u.iat[ix]:
                    exps_df['t0set'].iat[ix] = newt0set
        else:
            raise Exception(f"No Time0 for {datesetname}")


    if dbgp:
        print("A1 Final exps_df t0set:")
        print(exps_df['t0set'])
        print("A1 Final expsT0:")
        debug_print(expsT0, 'expsT0')

    return expsT0, exps_df


def get_nont0_sets(exps_df, debug_print_bool=False):
    """
    Returns:
        unique_nont0sets (pandas Series): list of exps_df t0set values that don't have Time0 as their short,
                   could rename exps_df['t0set'] to exps_df['datesetnames']

    """

    nont0sets = []
    nont0_ix = []
    # We look through all elements of t0set and take unique values that don't have their
    # corresponding 'short' be a Time0
    for ix, val in exps_df['t0set'].items():
        if exps_df['short'].loc[ix].upper() != 'TIME0':
                nont0sets.append(val)
                nont0_ix.append(ix)
    
    nont0sets_srs = pd.Series(data = nont0sets, index=nont0_ix) 
    unique_nont0sets = list(nont0sets_srs.unique())
    
    if debug_print_bool:
        debug_print(unique_nont0sets, 'nont0sets')

    return unique_nont0sets 


def createExpsT0(exps_df, debug_print_bool=False):
    """
    Args: exps_df:
        data frame with cols:
            short (str): string explaining if Time0 or not
            t0set (str): is date + space + setName for ALL experiments in exps_df,
                not only just the t0sets

    Returns 
        expsT0: dict mapping t0set name 'date setName' - > list<set+Index (str) that's related>
            for every actual Time0 name
    """

    time0_df = exps_df[[True if val.upper() == "TIME0" else False for ix, val in exps_df['short'].items()]]

    expsT0 = {}
    for ix, val in time0_df['t0set'].items():
        if val in expsT0:
            expsT0[val].append(time0_df['name'].loc[ix])
        else:
            expsT0[val] = [time0_df['name'].loc[ix]]

    if debug_print_bool:
        debug_print(expsT0, 'expsT0')

    return expsT0


def check_starting_values(exps_df, genes_df, all_df):
    """
    1. We check if there are any experiments in exps_df,
    2. We make sure all names in exps_df are also in all_df
    3. We make sure genes_df has 'scaffold' and 'begin' values
    """

    # Note, dataframe.shape[0] is number of rows
    if (exps_df.shape[0]==0):
        raise Exception("No experiments left to analyze!")

    for ix, nm in exps_df['name'].items():
        if nm not in all_df:
    	    raise Exception(f"name {nm} missing from all.poolcount")
    if None in genes_df.scaffoldId.values:
        raise Exception("No scaffold for genes_df")
    if None in genes_df.begin.values:
        raise Exception("No begin for genes_df")



def prepare_time0s(exps_df, dbg_prnt=False):
    """
    We take situations in which Group = Time0 and 
        make the short of that row also Time0
    Args:
        exps_df (DataFrame):
            Should contain cols 'Group', 'short'

    """

    # if Group = Time0, it is a Time0, even if "short" has a different description
    # the 'short' value defines the final 'Time0' 
    num_time_zero = 0
    if 'Group' in exps_df:
        for ix, val in exps_df['Group'].items():
            if val.upper() == "TIME0":
                num_time_zero += 1
                if dbg_prnt:
                    print(f"For {exps_df['name'][ix]}, Group is Time0")
                exps_df.loc[ix, 'short'] = "Time0"
    if dbg_prnt:
        print(f"Total number of time zeros: {num_time_zero}")

    return exps_df



def gene_strain_fit_func(set_index_name, exps_df, all_df, 
                         genes_df, expsT0,
                         t0tot, strainsUsed_hg2, has_gene2,
                         genesUsed, genesUsed12, minGenesPerScaffold,
                         all_df_has_gene
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
        all_df: Data frame holding all.poolcount file
        genes_df: Data frame holding genes.GC table
        expsT0: (dict) mapping (date setname) -> list<set.Index>
        t0tot: data frame where column names are 'date setname'
                and linked to a list of sums over the indexes that relate
                to that setname, with the list length being equal to the
                total number of strains (barcodes) in all.poolcount
                all columns are t0's?
        strainsUsed_hg2 pandas Series(list<bool>): whose length is same as num of Trues in has_gene2
                        equivalent index to has_gene2 True values
        has_gene2: list<bool> whose length is total number of strains.
                    row with strains that have gene insertions between
                    0.1 < f < 0.9 hold value True
        genesUsed: list<locusId> where each locusId is a string
        genesUsed12 (list<str>): list of locusIds that have both high f (>0.5) and low f (<0.5)
                    insertions with enough abundance of insertions on both sides
        minGenesPerScaffold: int
        all_df_has_gene (Dataframe): The parts of all_df that corresponds to True in has_gene2

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

    # all_cix (all current index) - panda series
    #   is a list of integers, one element for each row of all.poolcount
    all_cix = all_df[set_index_name]
    # t0_series is the related time 0 total series.
    t0_series = t0tot[t0set]

    # to_subtract is true if this is a time zero itself, so we remove
    # its values from the other time0 values.
    if to_subtract:
        # We subtract the poolcount values from the t0 totals 
        t0_series = t0_series - all_cix

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
  
    use1 = [bool(all_df_has_gene['f'].iloc[i] < 0.5) for i in range(len(all_df_has_gene['f']))]

    # Note that has_gene2 has to be the same length as all_cix, 
    # and t0_series, and strainsUsed
    gene_fit = GeneFitness(genes_df, all_df_has_gene, 
                           all_cix[has_gene2], t0_series[has_gene2],
    		           strainsUsed_hg2, genesUsed, sorted(genesUsed12), 
    		           minGenesPerScaffold=minGenesPerScaffold,
                           set_index_name=set_index_name,
                           cdebug=False,
                           use1 = use1,
                           all_df=all_df)

    
    cntrl = list(expsT0[t0set])
    if set_index_name in cntrl:
        cntrl.remove(set_index_name)
    if len(cntrl) < 1:
        raise Exception(f"No Time0 experiments for {set_index_name}, should not be reachable")

    strain_fit_ret_d = StrainFitness(all_cix, 
                      all_df[cntrl].sum(axis=1)
                      )
    
    # gene_fit, strain_fit, and strain_se
    ret_d = {"gene_fit": gene_fit, 
            "strain_fit": strain_fit_ret_d['fit'], 
            "strain_se": strain_fit_ret_d['se']
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


def getuse1(all_df, has_gene2, debug_print_loc=None):
    """
    We get a Dataframe called use1
    Args:
        all_df: all.poolcount
        has_gene2: list<bool> for good insertion genes
    Returns:
        use1: list<bool> with has_gene2 and all_df['f'] < 0.5 
    """
    use1 = all_df[has_gene2][all_df['f'] < 0.5]
    if debug_print_loc is not None:
        use1.to_csv(path_or_buf=debug_print_loc, sep='\t', index=False)
    return use1

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
                use1=None,
                all_df=None):
    """
    Args:
        genes_df: Data frame holding genes.GC table
                    must include cols locusId, scaffoldId, and begin (genes)
        all_df_has_gene: 
            subset of all_df (with good genes) which at the least contains headers:
                locusId, f (strainInfo)

        crt_all_series_has_gene (pandas Series): with counts for the current set.indexname 
                                 with has_gene2 value true (0.1<f<0.9) [countCond]
        crt_t0_series_has_gene (pandas Series): with t0 counts for each strain [countT0]

        # Convert below into pandas series

        strainsUsed_has_gene pandas Series(list<bool>): whose length is Trues in has_gene2
                        equivalent index to has_gene2 True values

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
                                             genes_df, minToUse=minGenesPerScaffold)

    
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
                              has_gene2 and might also have other conditions such as f >/< 0.5
        current_set_index_name (str): Name of set index in all.poolcount that
                                    we are currently analyzing
        run_typ (str): Debugging which part of GeneFitness are we running?

    Returns:
        DataFrame: with cols:
            fit: fitRaw column normalized by Median 
            fitNaive: 
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

    """
    if strainsUsed is None:
        strainsUsed = [bool(x >= minStrainT0) for x in crt_t0_series_has_gene]

    if genesUsed is None:
        # geneT0 is a dataframe with 2 col names: crt_t0_name, and locusId
        geneT0 = py_aggregate_series_to_series(crt_t0_series_has_gene[strainsUsed],
                                               crt_t0_name,
                                               strainLocus[strainsUsed],
                                               'locusId',
                                                func='sum')
        # genesUsed are the locusIds whose crt_t0_name value is larger than minGeneT0
        genesUsed = geneT0['locusId'][[geneT0[crt_t0_name].iloc[i] >= minGeneT0 for i in 
                                        range(len(geneT0[crt_t0_name]))]]
    """

    # Up to here it's exactly the same as the R file, Note that the indexes of strainsUsed
    #       map to index integer locations in strainLocus
    strainsUsed = [bool(strainsUsed.iloc[ix] and (strainLocus.iloc[ix] in genesUsed)) for ix in \
                    range(len(strainsUsed))]

    """
    with open("tmp/py_strainsUsed.json", 'w') as g:
        g.write(json.dumps(strainsUsed, indent=2))
    """

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

    # nStrains_d is a dict which takes list strainLocus_si of object -> number of times 
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
    strainSD_pre = [math.sqrt(1/(1 + crt_t0_series_hg_su.iat[i]) + 1/(1+crt_all_series_hg_su.iat[i]))/np.log(2) for i
                in range(len(crt_t0_series_hg_su))]
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


    
def NormalizeByScaffold(values, locusIds, genes_df, window=251, minToUse=10, cdebug=False):
    """
    Args:
        values: pandas Series of main_df['fit'] from AvgStrainFitness
        locusIds: pandas Series of main_df['locusIds'] from AvgStrainFitness
        genes_df: Data Frame from genes.GC
        window (int): window size for smoothing by medians. Must be odd, default 251. For scaffolds
                      with fewer genes than this, just uses the median.
        minToUse (int): If a scaffold has too few genes, cannot correct for possible DNA extraction
                        bias so we need to remove data for that gene (i.e., returns NA for them)

    Returns:
        values (pandas Series of floats)
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
        if len(rows) < minToUse:
            if cdebug:
                print("Removing " + str(len(rows)) + " values for " + scaffoldId)
            values[rows] = None
        else:
            med = values[rows].median()
            if cdebug:
                print("Subtraxting median for " + scaffoldId + " " + str(med))
            values[rows] = values[rows] - med

            if len(rows) >= window:
                # srtd_begs is a list of indexes for the sorted values
                srtd_begs = py_order(gn_begin.iloc[rows])
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

    """

split divides the data in the vector x into the groups defined by f. The replacement forms replace values corresponding to such a division. unsplit reverses the effect of split.
split(x, f, drop = FALSE, …)

    fitness = lapply(split(1:length(crt_t0_series_hg_su), list(locusId=strainLocus)),
     	           function(j) {
		       n = length(ix_l);
                       totw = sum(strainWeight[ix_l]);
		       fitRaw = sum(strainWeight[ix_l] * strainFit[ix_l]) / totw;
		       tot = sum(crt_all_series_hg_su[ix_l]);
		       tot0 = sum(crt_t0_series_hg_su[ix_l]);
		       sd = math.sqrt(sum(strainWeight[ix_l]**2 * strainSd[ix_l]))/totw;
		       sumsq = sum(strainWeight[ix_l] * (strainFit[ix_l]-fitRaw)**2)/totw;
		       # high-N estimate of the noise in the log2 ratio of fitNaive
		       # But sdNaive is actually pretty accurate for small n -- e.g.
		       # simulations with E=10 on each side gave slightly light tails
		       # (r.m.s.(z) = 0.94).
		       sdNaive = math.sqrt( 1/(1+tot) + 1/(1+tot0) ) / log(2);
		       nEff = totw/max(strainWeight[ix_l]);
		       c(fitRaw=fitRaw, sd=sd, sumsq=sumsq, sdNaive=sdNaive, n=n, nEff=nEff,
		         tot=tot, tot0=tot0);
		});
    fitness = data.frame(do.call(rbind, fitness));
    fitness$fit = mednorm(fitness$fit);
    fitness$fitNaive = mednorm(math.log2(1+fitness$tot) - math.log2(1+fitness$tot0));
    fitness$locusId = row.names(fitness);
    if (is.integer(strainLocus)) fitness$locusId = as.integer(as.character(fitness$locusId));

    if(returnStrainInfo) return(list(genes=fitness,
        strains=data.frame(strainLocusF,crt_all_series_hg_su,crt_t0_series_hg_su,strainPseudoCount,strainFit,strainSd,strainWeight)));
    # else
    return(fitness);

    """

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



def mednorm(pd_series):
    # takes pandas series and returns pandas series with median subtracted
    crt_median = pd_series.median()
    new_series = pd_series - crt_median
    return new_series





def get_sample2locusId(all_df, has_gene2, meta_ix, dbg_lvl=0):
    """ 
    Args:
        all_df: all.poolcount dataframe
        has_gene2: list of booleans relating to genes with insertions
                    0.1 < f < 0.9
        meta_ix: integer marking where sets begin and not metadata
        
        (Created):
        good_inserts_all_df: data frame which has all poolcount inserts
            that are within 0.1 and 0.9 frac of the gene
        good_locus_ids: a list of all the locus ids which have good values
    Returns:
        sample2locusId is a dict that goes from sample name (str) to a dict of 
          locusId -> num barcodes counted in that locusId
            with only the good gene insertions
    """

    if dbg_lvl > 1:
        logging.info("Starting to get sample2locusId")

    # We get a matrix of sample (col) to genes (rows) with entry being total number
    # of insertions within that gene if the insertions occured within 0.1 and 0.9 f
    good_inserts_all_df = all_df[has_gene2]
    good_locus_ids = good_inserts_all_df['locusId']
    unq_good_loc_id = list(good_locus_ids.unique()) 

    sample2locusId = {}
    # This variable is just to save computation within the upcoming cycles
    col_len = len(good_locus_ids)
    # We iterate over all the sample names (i.e. setname + index) 
    for colname in good_inserts_all_df.columns[meta_ix:]:
        locusId2count = {x:0 for x in unq_good_loc_id}
        current_col = good_inserts_all_df[colname]
        for i in range(col_len):
            locusId2count[good_locus_ids.iat[i]] += int(current_col.iat[i])
        sample2locusId[colname] = locusId2count

    if dbg_lvl > 1:
        logging.info("Finished getting sample2locusId")

    return sample2locusId

def DataFrame_sample2locusId_TSV(sample2locusId, op_fp="tmp/py_sample2loc_test.TSV", 
                                 print_bool=False):
    """
    We create a data frame out of sample2locusId
    Args:
        sample2locusId is a dict that goes from sample name (str) to a dict of 
          locusId -> num barcodes counted in that locusId 
          but only includes barcodes with good insertions (0.1<f<0.9)
        
        print_bool:  print to debug
    Returns:
        dfObj:
            cols:
                locusId samplename1 samplename2, etc.
            beneath are the locusIds (first column)
                then values which are the number of barcodes counted in that locusId
    """

    sample_names = list(sample2locusId.keys())
    header_row = ["locusId"] + sample_names
    # every sample has the same locus Ids - we get it from the first sample
    locusIds = sorted(list(sample2locusId[sample_names[0]].keys()))
    df_rows = []
    for locusId in locusIds:
        df_row = [str(locusId)]
        for sample_name in sample_names:
            df_row.append(sample2locusId[sample_name][locusId])
        df_rows.append(df_row)
    
    dfObj = pd.DataFrame(df_rows, columns=header_row)
    if print_bool:
        dfObj.to_csv(path_or_buf=op_fp, sep='\t', index=False)
        logging.info(f"Wrote sample2locusId TSV at {op_fp}")
    return dfObj




def set_up_ignore(ignore, all_df, exps_df, minSampleReads, meta_ix=7, dbg_prnt=False):
    """ Setting up the index (columns of all.poolcount) names to avoid analysis for
    Those indexes we ignore are 
        1. If any are already listed in the variable ignore (list or None)
        2. If the sum of barcodes in that index is less than the value
            'minSampleReads'
        3. If in exps_df the column 'Drop' holds the value 'True' or 'TRUE'
    Note we only need a list of these values because it is not tied to the strains
        in all.poolcount.
    We remove the indeces from all_df (where they are column names) & exps_df (where
        they are under the value 'name')

    Args:
        ignore: None or list of str with sample-index name to ignore
        all_df: Data frame of all.poolcount
        exps_df: Data frame of experiments file
            Must contain cols: 'name'
            Could contain cols: 'Drop'
        minSampleReads: int
        meta_ix: Start of where the indeces become sample/index names
    
    Returns:
        all_df, exps_df, ignore (list<str>, where str is name of indeces
                                we are ignoring)
    """
    # Creating a list to ignore out of the all.poolcount indexes
    if ignore is None: 
        # metacol is ignored 
        # We select all the columns
        tot = all_df.iloc[:,meta_ix:].sum(axis=0)
        # We figure out the columns for which the sum of barcodes
        # found is less than minSampleReads
        ignore = []
        for c in tot.keys():
            if tot[c] < minSampleReads:
                ignore.append(c)
                if dbg_prnt:
                    print(f"Ignoring sample index name: {c}")
    # any returns TRUE or FALSE depending on if some value returns true
    if 'Drop' in exps_df: 
        # The 'Drop' column means if Drop=TRUE then ignore sets column
        for ix, val in exps_df['Drop'].items():
            # Either val is true or it's a string that has text 'true' in it.
            x = str(val)
            if x.strip().upper() == "TRUE" or val == True:
                if exps_df['name'][ix] not in ignore:
                    ignore.append(exps_df['name'][ix])
    if(len(ignore) > 0):
        print("Ignoring " + ", ".join(ignore))
        exps_keep =  [(not (val in ignore)) for val, ix in exps_df['name'].items()]
        if dbg_prnt:
            print("Pre removal:")
            print(exps_df['name'])
            print(exps_keep)
        new_exps_df = exps_df[exps_keep]
        if dbg_prnt:
            print("Post removal:")
            print(new_exps_df['name'])

        all_drop = [x for x in ignore if x in all_df]
        if dbg_prnt:
            print("all_drop:")
            print(all_drop)
        all_df = all_df.drop(labels=all_drop, axis=1)

        return [all_df, new_exps_df, ignore]

    return [all_df, exps_df]


def CrudeOp(genes_df, dbg_out_file=None, dbg=False):
    """
    Crude operon predictions -- pairs of genes that are on the same strand and
    separated by less than the median amount are predicted to be in the same operon
    Input genes is a data frame with locusId, strand, begin, end, with genes in sorted order
    Returns a data frame with Gene1, Gene2, Sep for separation, and bOp (TRUE if predicted operon pair)
    Note: dbg_out_file set to tmp/py_CrudeOpout1.tsv

    Args:
        genes_df is a dataframe which must have keys:
            locusId, begin, end
    Returns:
        DataFrame with cols 
            Gene2, Gene1, sysName1, type1, scaffoldId1, begin1, end1, strand1, name1, desc1, GC1, nTA1, 
            sysName2, type2, scaffoldId2, begin2, end2, strand2, name2, desc2, GC2, nTA2, Sep, bOp

    """
    # To assist with first merge we rename the column name locusId to Gene1
    # We offset all the locusIds by 1: First we ignore the last one, then we ignore the first
    # And place them side by side (Gene1, Gene2)
    g1_g2_df =  pd.DataFrame.from_dict({
                            "Gene1": list(genes_df['locusId'].iloc[:-1]),
                            "Gene2": list(genes_df['locusId'].iloc[1:])
                            })

    genes_df = genes_df.rename(columns={"locusId":"Gene1"})

    mrg1 = g1_g2_df.merge(
                          genes_df, sort=True,
                          left_on="Gene1",
                          right_on="Gene1",
                          how="inner")

    # Removing unused variable from memory
    del g1_g2_df

    if dbg_out_file is not None:
        mrg1.to_csv( dbg_out_file, sep="\t")

    # Now for the second merge we rename the column name Gene1 to Gene2
    genes_df = genes_df.rename(columns={"Gene1":"Gene2"})
    new_df = mrg1.merge(
                        genes_df,
                        sort=True,
                        suffixes=["1","2"],
                        left_on="Gene2",
                        right_on="Gene2",
                        how="inner")
    del mrg1
    

    # Now we return the column to its original name in case it's altered in the original form
    genes_df = genes_df.rename(columns={"Gene2":"locusId"})

    if dbg:
        print("CrudeOp new dataframe column names: " + \
                ", ".join(list(new_df.head())))

    if dbg_out_file is not None:
        new_df.to_csv( dbg_out_file + "second", sep="\t")


    st1_eq_st2 = [bool(new_df['strand1'].iloc[i]==new_df['strand2'].iloc[i]) for i in range(len(new_df['strand1']))]
    if dbg:
        print(f"Num trues in bool list: {st1_eq_st2.count(True)}")
    new_df = new_df[st1_eq_st2]

    paralmin = []
    for i in range(len(new_df['begin1'])):
        paralmin.append(min(abs(new_df['begin1'].iat[i] - new_df['end2'].iat[i]), 
                            abs(new_df['end1'].iat[i] - new_df['begin2'].iat[i]) ))

    new_df['Sep'] = paralmin
    # Below series is boolean (True/False)
    new_df['bOp'] = new_df['Sep'] < new_df['Sep'].median()

    if dbg_out_file is not None:
        new_df.to_csv( dbg_out_file + "third", sep="\t")

    return new_df


def stop(line_num):
    raise Exception(f"Stopped, line {line_num}") 


def tmp_prep_wrap_up2(special_vars_dir):
    """
    special_vars_dir contain the files:
        genesUsed.tsv, strainsUsed_hg2.tsv, genesUsed12.tsv, all_gN.tsv, t0_gN.tsv, t0tot.tsv
    """
    spesh_fs = os.listdir(special_vars_dir)
    for x in ["genesUsed.tsv", "strainsUsed_hg2.tsv", "genesUsed12.tsv", 
            "all_gN.tsv", "t0_gN.tsv", "t0tot.tsv"]:
        if x not in spesh_fs:
            raise Exception(f"Special vars dir must contain {x} but does not.")

    genesUsed = pd.read_table(os.path.join(special_vars_dir, "genesUsed.tsv")).drop(
            labels=["Unnamed: 0"], axis=1)
    strainsUsed = pd.read_table(os.path.join(special_vars_dir, "strainsUsed.tsv")).drop(
            labels=["Unnamed: 0"], axis=1)
    genesUsed12 = pd.read_table(os.path.join(special_vars_dir, "genesUsed12.tsv")).drop(
            labels=["Unnamed: 0"], axis=1)
    all_gN = pd.read_table(os.path.join(special_vars_dir, "all_gN.tsv")).drop(
            labels=["Unnamed: 0"], axis=1)
    t0_gN = pd.read_table(os.path.join(special_vars_dir, "t0_gN.tsv")).drop(
            labels=["Unnamed: 0"], axis=1)
    t0tot = pd.read_table(os.path.join(special_vars_dir, "t0tot.tsv")).drop(
            labels=["Unnamed: 0"], axis=1)

    return genesUsed, strainsUsed, genesUsed12, all_gN, t0_gN, t0tot
    
def export_gene_fit_d(gene_fit_d, op_dir, dbg_prnt=True):
    """
    Args:
        gene_fit_d (python dict):
            'g', 
            'lrRaw', 
            'sd',
            'sumsq',
            'sdNaive',
            'n',
            'nEff',
            'tot',
            'tot0',
            'lr',
            'lrNaive',
            'lrn',
            'lr1',
            'lr2',
            'lrn1',
            'lrn2',
            'tot1',
            'tot1_0',
            'tot2',
            'tot2_0',
            'pseudovar',
            'se',
            't',
            'version',
            'q',
            'genesUsed',
            'strainsUsed',
            'genesUsed12',
            'gN',
            't0_gN',
            'strains',
                used,
                enoughT0
                & multiple others (all_df meta_ix columns)
            'strain_lr',
            'strain_se',
            'high' 
            [pairs]:
                adjDiff:
                    Gene1, Gene2, sysName1, type1, scaffoldId, begin1, end1, strand1, name1, desc1, GC1, 
                    nTA1, locusId, sysName2, type2, begin2, end2, strand2, name2, desc2, GC2, nTA2
                    rfit (float)
                random:
                    Gene1
                    Gene2
                    rfit
                pred:
                    Gene2, Gene1, sysName1, type1, scaffoldId1, begin1, end1, strand1, name1, desc1, GC1, nTA1, 
                    sysName2, type2, scaffoldId2, begin2, end2, strand2, name2, desc2, GC2, nTA2, Sep, bOp
                    rfit
            [cofit] (pandas DataFrame):  
                locusId (str), 
                hitId (str) 
                cofit (float)
                rank (int)
            [specphe]: (Not done)
    """
    for k in gene_fit_d.keys():
        if dbg_prnt:
            print(type(gene_fit_d[k]))
        if type(gene_fit_d[k]) == pd.Series or type(gene_fit_d[k]) == pd.DataFrame:

            if dbg_prnt:
                print(f"exporting pandas object {k} to {op_dir}/{k}.tsv")
            gene_fit_d[k].to_csv(os.path.join(op_dir, k + ".tsv"), sep="\t", index=None)
        else:
            if dbg_prnt:
                print(f"exporting non-pandas object {k}, {type(gene_fit_d[k])} to {op_dir}/{k}.json")
            with open(os.path.join(op_dir, k + ".json"), "w") as g:
                g.write(json.dumps(gene_fit_d[k]))

def test():
    return None

def main():
    args = sys.argv
    print("Needs inp_dir all_pc genes_fp")
    inp_dir = args[1]
    all_pc_fp = args[2]
    genes_fp = args[3]
    genefitresults = export_or_import_genefitresults({}, 
                                      "imp", inp_dir, dbg_print=True)

    exps_df = pd.read_table("tmp/py_exps_df235.tsv")

    all_df, genes_df, has_gene2 = tmp_prep_wrap_up(all_pc_fp, genes_fp)
    gene_fit_d, CrudeOp_df =  start_gene_fit_d(genefitresults, exps_df, all_df, genes_df,
                     has_gene2, meta_ix=7, debug=True)

    genesUsed, strainsUsed, genesUsed12, all_gN, t0_gN, t0tot = tmp_prep_wrap_up2("tmp/special_vars")
    CrudeOp_df = CrudeOp(genes_df)

    gene_fit_d = finish_gene_fit_d(gene_fit_d, genefitresults, genes_df, all_df, exps_df,
                      genesUsed, strainsUsed, genesUsed12,
                      all_gN, t0_gN, t0tot, CrudeOp_df, dbg_prnt=True)
   
    export_gene_fit_d(gene_fit_d, "tmp/ResultStorage2")
    stop(3572)

if __name__ == "__main__":
    main()
