#!python3

import os
import logging
import pandas as pd
import numpy as np
from scipy import stats
import json
import math 
import time
from translate_R_to_pandas import * 

"""
All functions and subroutines:
    analysis_2:
        initialize_gene_fit_d
        FitReadMetrics
        FitQuality
            CrudeOp
            AdjacentPairs
            paircor
        FEBA_Exp_Status
        normalize_per_strain_values
            StrainClosestGenes
                (py_unsplit) from translate_R...
            create_strain_lrn

"""




def analysis_2(GeneFitResults, exps_df, all_df, genes_df, 
               strainsUsed_list, t0tot,
               cfg=None,
               meta_ix=7, debug=False):
    """
    Args:
        GeneFitResults:
            setnameIndex -> ret_d
               ret_d:
                  gene_fit: DataFrame, contains cols:
                    locusId <str>: The locusId to which this row is associated.
                    fit: fitRaw column normalized by Median 
                    fitNaive (float): Median normalized log2 difference between tot0 and tot 
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
                    fitnorm (float): Scaffold normalized fit scores (median and mode)
                    fitnorm1 (float): fit1 + fitnorm - fit
                    fitnorm2 (float): fit2 + fitnorm - fit
                    tot1 (int or nan): For every locusId found in genesUsed12, we give the tot value of first_half_df
                    tot0_1 (int or nan): For every locusId found in genesUsed12, we give the tot0 value of first_half_df
                    tot2 (int or nan): For every locusId found in genesUsed12, we give the tot value of second_half_df
                    tot0_2 (int or nan): For every locusId found in genesUsed12, we give the tot0 value of second_half_df
                  strain_fit: pandas Series (float) per-strain fitness (len(all_df))
                  strain_se: pandas Series (float) (len(all_df))
        cfg (python dict):
            minT0Strain: int
            status_d (python dict):
                min_gMed : (float)
                max_mad12 : (float) 
                min_cor12 : (float) 
                max_gccor : (float) 
                max_adjcor : (float)
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
            tot0_1 (int or nan) dataframe with one column per setindexname
            tot2 (int or nan) dataframe with one column per setindexname
            tot0_2 (int or nan) dataframe with one column per setindexname
            tot (int or nan) dataframe with one column per setindexname
            tot0 (int or nan) dataframe with one column per setindexname
            version (str)

    Description:
        First thing, we create 'gene_fit_d', which is a python dict.
            We take the GeneFitResults, a python dict that
            has keys being the experiment names and dataframes with
            categories (like log ratios) being the values, and we convert it to
            a dict with each category being a key, and those point
            to dataframes with each column being an experiment name,
            and each row being associated with one gene, the genes
            being denoted by a pandas Series within the dictionary
            under the key 'g'. (This pandas series 'g' contains the 
            locusIds (strings)).
            Another thing we do in this creation of gene_fit_d, is 
            we replace the key 'fitnorm' with 'lrn' (for log ratios
            normalized) and 'fit' with 'lr' (for just log ratios).
        Then we initialize the Quality DataFrame. 
            Out of the rows of exps_df, we only take the experiments 
            that are found in the columns of the dataframe of normalized
            log ratios. Meaning we only take the experiments 
            that passed the thresholds necessary to have analysis
            performed on them. We take these rows from the dataframe
            'exps_df' but only the columns "name", "short", and "t0set",
            all other columns we ignore. We find these rows through the column
            "name".
        Next we run FitReadMetrics and FitQuality in order to add 
            more columns to our Quality DataFrame.
        FitReadMetrics:
            We take the subset of all_df with the used experiment names
            as the columns (all rows included), and we compute the sums
            over all rows for each of the experiments for the following:
            nMapped (the total number of reads mapped under that column).
            nPastEnd (the total number of pastEnd reads under that column).
            nGenic (the total number of reads with good gene insertions).
            Good gene insertions means inserted between .1 and .9 as 'f'.
            Then we create a dataframe whose index is the experiment
            names and the columns are 'nMapped', 'nPastEnd' and 'nGenic',
            so the number of rows of the dataframe is the same as the
            number of good experiments, whereas the number of columns
            is fixed to 3.
        In FitQuality we create 12 metrics based on correlations
            and finding adjacent pairs of genes or genes that are
            expected to be in the same operon based on their distance.
            First we get the dataframe crudeOpGenes, which is similar to the genes_df dataframe
            in terms of its columns, but it is a dataframe with pairs of consecutive 
            genes from genes_df, with an extra column 'Sep' denoting the distance
            between the genes, and a column 'bOp' that says whether this distance
            is less than the median distance or not (True if less, False if greater).
            We get another similar dataframe called adj which are the adjacent pairs
            but without the Sep or bOp values. (Can we not use this dataframe?)
            Then we get the matching index between gene_fit_d genes (under the
            column name 'g') and the locusIds in genes_df. Using these matching index,
            we compute the correlation between each column in the normalized log ratios
            dataframe and the GC values of those genes.
            in genes_df. GC_corr is a pandas Series whose length is the number
            of experiments (columns) in gene_fit_d['lrn']; where the index
            labels are the experiment names, and the values are the Pearson 
            correlations. 
            Finally, we compute statistics regarding the experiments and return
            a dataframe with the following labels and meanings:
            "nUsed" (int): Sum over all locusIds of the sum over the locusId (Sum of sum) per experiment
            "gMed" (int): Median of all locusIds over the sums per locusId per experiment
            "gMedt0" (int): Median over the time0 totals per experiment
            "gMean" (float): Mean of all locusIDs over the sums per locusId per experiment
            "cor12" (float): Correlation of normalized log ratios between first and second half 
                             gene insertion locations per experiment
            "mad12" (float): Medians of the Absolute values of the differences between first and second half
                                normalized log ratios.
            "mad12c" (float): Median of some statistic over all experiments
            "mad12c_t0" (float): Same as above but over the time0s
            "opcor" (float): Correlation between normalized log ratios of pairs of genes predicted to be 
                     in the same operon per experiment
            "adjcor" (float): Correlation between normalized log ratios of pairs of adjacent genes per experiment
            "gccor" (float): Correlation between normalized log ratios of genes and their GC percentage per experiment
            "maxFit" (float): The maximum normalized log ratio value per experiment

        We combine the initialized Quality DataFrame, the Fit Read
        Metrics DataFrame and the FitQuality DataFrame and that's
        our current Quality DataFrame, which contains 19 (or possibly 18) columns,
        and the number of rows is equal to the total number of 
        good experiments (so it varies per run).
        Then we run the function FEBA_Exp_Status
        FEBA_Exp_Status:
            For each row in the Quality DataFrame, we check the values under 
            certain columns to decide how to label it's quality.
            The values we check, in this order, are:
            1. Is it a Time0? ("Time0")
            2. Does it have a low Median over the sums over locusIds? ("low_count")
                - param 'min_gMed'
            3. Does it have a high Median of the differences between first and
                second half log ratios? ("high_mad12") - param 'max_mad12' 
            4. Is there a low correlation between the first and second half
                log ratios? ("low_cor12") - param 'min_cor12'
            5. Is the GC correlation high or is the adjacent genes correlation
                high? ("high_adj_gc_cor") - params 'max_gccor' & 'max_adjcor'
            If none of these are True, then the experiment is given the status
            "OK". Otherwise, it returns the first value for which it is 
            True in the above questions, the value it returns is the
            string in parentheses.
            The whole function returns a pandas series the length of which
            is the number of rows in quality_df.
        Now we continue to add on to the quality dataframe by creating
        a column
        called 'u', in which we go through each experiment, and if its 
        status is "OK", then we give it the value True, otherwise it is
        given the value False. So 'u' contains whether the experiment
        has a status that passed. After adding the column 'u' to the 
        Quality DataFrame, it has 20 (or 19) columns total.

        We shift our focus to the 'strains' DataFrame. 
        First, we only take the metadata columns from all.poolcount,
        meaning the columns:
        'barcode', 'rcbarcode', 'scaffold', 'strand', 'pos', 'locusId', 'f'
        and add new columns to it:
        'used': which strains were used to compute Gene Fitness (just the
            strainsUsed List as a column.)
        'enoughT0': which strains had a high enough Mean within T0s

        Now we create two important dataframes:
        strain_lr and strain_se: Both of which have the
            same length (num rows) as all.poolcount. Both simply
            take the values computed in the function
            StrainFitness (in analysis1) and turn them into 
            dictionaries with experiment names as the column 
            names and the values as the columns beneath them.

        Next we normalize the strain fitness values (creating
        the dataframe 'strain_lrn'). We use the function
        normalize_per_strain_values:
            First, for every strain, we find the closest gene center from our list of used genes
            (genes that passed all the thresholds to be used). We call this list 'strainToGene'.
            So for strains that are within a gene, their closest gene will also be the gene
            that they are inserted into.
            Next, we create a dataframe that is the difference between the normalized log
            ratio values per experiment per gene, and the log ratio values per gene (that 
            are not normalized). 
            Then, for each strain, we normalize its fitness values in a complicated way.
            The way in which the fitness values are normalized are the following:
            create_strain_lrn:
                First, for each experiment, we take the per gene difference between 
                the normalized log ratio and the plain old log ratio, and then we 
                map those values onto all the strains using the Strain To Closest 
                Gene series we computed earlier. So for each strain, we take the 
                closest Gene and place the difference between the normalized 
                log fitness and the regular log fitness in a pandas Series we call 
                "per_strain_exp_diff". Then we group the strains by scaffold and
                take the medians of per_strain_exp_diff by these scaffolds, and for each
                strain, instead of having its own gene difference value, it instead now
                has the median of all the strain difference values in the same scaffold
                as it. Next we multiply that entire series by negative 1 and call it
                neg_str_scf_med, for 'negative strain per scaffold median'.
                Now we initialize the numbers we'll add to the original log ratios
                in order to get the normalized strain values. For each value in
                per_strain_exp_diff, if it's 'NA', then we insert the negative
                median of the differences from neg_str_scf_med, otherwise, we leave the
                value as it is (the per_strain_exp_diff value); we call the new series 'sdiff'.
                To get the final normalized log ratios for the strains under this 
                experiment, we simply add the original log ratios per strain to the values
                in sdiff, and that's our column for this experiment.
                The entire normalized log ratio per strain dataframe is one column
                per experiment name.

        So within the function analysis2, we have created several new dataframes
            to the output 'gene_fit_d':
            'q': for quality, with 20/19 columns and one row per used experiment,

            ALL BELOW DATAFRAMES HAVE THEIR NUMBER OF ROWS  = nAllStrains
            'strains': The original strains meta_ix dataframe, with two extra
                        columns 'used' and 'enoughT0'
            Below dataframes have number of columns = nExperimentsUsed 
            'strain_lr': The log ratios per experiment (unnormalized)
            'strain_se': The standard error per strain per experiment
            'strain_lrn': The normalized log ratios per experiment

            This is a pandas Series (also length nAllStrains):
            'strainToGene': Closest gene to strain using index from 'g' pandas Series.

        Function ends and returns gene_fit_d
            
    """

    if cfg is not None:
        minT0Strain = cfg["minT0Strain"] 
    else:
        minT0Strain=3
        cfg = {
                "status_d": None
        }


    gene_fit_d = initialize_gene_fit_d(GeneFitResults, debug=True) 

    # We recompute central_insert_bool_list:
    central_insert_bool_list = [True if (0.1<=x<=0.9) else False for x in all_df['f']]

    # What is q? Quality

    q_col = ["name", "short", "t0set"]
    if "num" in exps_df:
        q_col.append("num")

    # Creating quality datafrme
    # We only get the rows which have enough experiments assoc with genesUsed12
    tmp_name_in_lrn = [True if exps_df['name'].iloc[i] in gene_fit_d['lrn'].head() else False for i \
                        in range(len(exps_df['name']))]
    
    # quality_df is a DataFrame with 4 (or 3) columns and num rows = tmp_name_in_lrn.count(True)
    quality_df = exps_df[tmp_name_in_lrn][q_col]
    quality_df.index = list(quality_df['name'])
    #gene_fit_d['q'] = quality_df
    q_exp_names = quality_df['name']
    for i in range(len(q_exp_names)):
        if not q_exp_names.iat[i] == list(gene_fit_d['lrn'].head())[i]:
            raise Exception(f"Mismatched names in fit: {q_exp_names.iat[i]} != "
                            f"{list(gene_fit_d['lrn'].head())[i]}")

    if debug:
        print("Running FitReadMetrics() and FitQuality()")
    st = time.time()
    # fitreadmet is a dataframe with 3 columns and the num rows = len(q_exp_names)
    fitreadmet = FitReadMetrics(all_df, q_exp_names, central_insert_bool_list)
    print(f"Time to run FitReadMetrics: {time.time() - st} seconds")

    st = time.time()
    # fq_result is a dataframe with 12 columns and num rows = len(q_exp_names)
    fq_result, CrudeOp_df = FitQuality(gene_fit_d, genes_df, prnt_dbg=False)
    print(f"Time to run FitQuality: {time.time() - st} seconds")

    # Since we are concatenating one dataframe with 4 (or 3) cols, one with 3, and one
    # with 12 cols, the overall dataframe gene_fit_d['q'] should have 19 columns
    # (or 18 depending on if 'num' is not in exps_df. It should be so we expect 4)
    gene_fit_d['q'] = pd.concat([quality_df, 
                                 fitreadmet,
                                 fq_result], axis=1)

  
    # HERE
    #DEBUG:
    #gene_fit_d['q'].to_csv("tmp/py_gene_fit_q2.tsv", sep="\t")
    # status is a pandas series of str, should be exa
    status = FEBA_Exp_Status(gene_fit_d['q'], status_d=cfg["status_d"], dbg_prnt=False)

    # We get a list of status is ok + False for the rows of q that surpass length of status
    gene_fit_d['q']['u'] = [True if status.iat[i] == "OK" else False for i in range(len(status))] 

    # Printing out
    for s in ["low_count", "high_mad12", "low_cor12", "high_adj_gc_cor"]:
        if list(status).count(s) > 0:
            logging.info(f"{s}: {gene_fit_d['q']['name'][status == s]}")

    # Creating strains dataframes and values (as opposed to genes)
    strains = all_df.iloc[:,0:meta_ix]
    strains['used'] = strainsUsed_list 
    strains['enoughT0'] = t0tot.mean(axis=1) > minT0Strain
    gene_fit_d['strains'] = strains
    gene_fit_d['strain_lr'] = pd.DataFrame.from_dict(
                            {x: list(GeneFitResults[x]['strain_fit']) for x in GeneFitResults.keys()}
                            )
    gene_fit_d['strain_se'] = pd.DataFrame.from_dict(
                            {x:list(GeneFitResults[x]['strain_se']) for x in GeneFitResults.keys()}
                            )

    # Dataframe, Series
    strain_lrn, strainToGene = normalize_per_strain_values(strains, genes_df, gene_fit_d)
    # Num rows of strain_lrn = len strainToGene  which is nAllStrains
    gene_fit_d['strain_lrn'] = strain_lrn
    gene_fit_d['strainToGene'] = strainToGene

    return gene_fit_d, CrudeOp_df


def initialize_gene_fit_d(GeneFitResults, debug=False):
    """
    We create the initial version of central variable
        'gene_fit_d'. Where we essentially flip the column
        names and the set names of the dataframes, in the sense that
        we go from having a single setindex name pointing to a
        dataframe with columns indicating certain info, to the names
        of those columns pointing to a dataframe with that column's info 
        over all the different set index names. We also replace
        'fitnorm' with 'lrn' for 'log ratios normalized' and
        'fit' with 'lr' (for simply log ratios).

    Args:
        GeneFitResults: (dict) setnameIndex -> ret_d
           ret_d:
               gene_fit: DataFrame, length is nGenesUsed. contains cols:
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
                    tot0_1 (int or nan)
                    tot2 (int or nan)
                    tot0_2 (int or nan)
                    tot (int or nan)
                    tot0 (int or nan)
               strain_fit: pandas Series (float)  Length is nAllStrains
               strain_se: pandas Series (float)  Length is also nAllStrains
    Returns:
        gene_fit_d: (python dict)
            g (pandas Series (str)): pandas Series of locusIds
            For below dataframes, num rows is nGenesUsed

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
            tot0_1 (int or nan) dataframe with one column per setindexname
            tot2 (int or nan) dataframe with one column per setindexname
            tot0_2 (int or nan) dataframe with one column per setindexname
            tot (int or nan) dataframe with one column per setindexname
            tot0 (int or nan) dataframe with one column per setindexname
            version (str)

        
        
    """

    all_ix_names = list(GeneFitResults.keys())
    # This dict will just contain dataframes gene_fit
    fit_locusIds = GeneFitResults[all_ix_names[0]]['gene_fit']['locusId']

    # Why do we replace the name locusId with 'g'? 'g' for gene
    gene_fit_d = {'g': fit_locusIds} 
    other_col_names = list(GeneFitResults[all_ix_names[0]]['gene_fit'].head())
    # other_col_names should be:
    #     fit, fitNaive, fit1, fit2, fitnorm1, fitnorm2, fitRaw
    #     locusId, n, nEff, pseudovar, sumsq, sd, sdNaive, se, t, tot1
    #     tot0_1, tot2, tot0_2, tot, tot0
    other_col_names.remove('locusId')
    if "Unnamed: 0" in other_col_names:
        other_col_names.remove("Unnamed: 0")
    print(other_col_names)

    st = time.time()
    for col_name in other_col_names:
        all_col_values_d = {ix_name: GeneFitResults[ix_name]['gene_fit'][col_name] for ix_name in GeneFitResults.keys()}
        gene_fit_d[col_name] = pd.DataFrame.from_dict(all_col_values_d)
    print(f"Time to create gene_fit_d: {time.time() - st}")

    # We replace fitnorm(1,2) with lrn(1,2)
    # And we replace fit(1,2) with lr(1,2)
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


def FitReadMetrics(all_df, qnames, central_insert_bool_list):
    """
    Args:
        all_df (pandas DataFrame):
        qnames (pandas Series): list<str> (names of set_index_names)
        central_insert_bool_list list<bool>: gene insertion between 0.1 and 0.9 fraction of length 
    
    Returns:
        DataFrame with cols:
            nMapped (int): Sum over all used experiments
            nPastEnd (int): Sum of pastEnd
            nGenic (int): Sum of reads within genes
    
    SubRoutines:
        dataframe.sum(axis=0) -> Sums all the values in each column,
        and returns a pandas series with col_name as index, and values
        being the sum
        

    Description:
        We take the subset of all_df with the used experiment names
        as the columns (all rows included), and we compute the sums
        over all rows for each of the experiments for the following:
        nMapped (the total number of reads mapped under that column).
        nPastEnd (the total number of pastEnd reads under that column).
        nGenic (the total number of reads with good gene insertions).
        Good gene insertions means inserted between .1 and .9 as 'f'.
        Then we create a dataframe whose index is the experiment
        names and the columns are 'nMapped', 'nPastEnd' and 'nGenic',
        so the number of rows of the dataframe is the same as the
        number of good experiments, whereas the number of columns
        is fixed to 3.

    """

    # Returned data type of sum function is Series,
    # In this case we have the index being the exp names, then value is 
    # the sum over all the rows
    nMapped = all_df[qnames].sum(axis=0)
    nPastEnd = all_df[all_df['scaffold']=="pastEnd"][qnames].sum(axis=0)
    nGenic = all_df[central_insert_bool_list][qnames].sum(axis=0)



    frm_df = pd.DataFrame.from_dict({
        "nMapped": list(nMapped),
        "nPastEnd":list(nPastEnd),
        "nGenic":  list(nGenic)
    })

    frm_df.index = nMapped.index
    return frm_df



def FitQuality(gene_fit_d, genes_df, prnt_dbg=False):
    """
    Args:
        gene_fit_d: (python dict) For each key, meaning is the same as input to analysis2()
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
            tot0_1 (int or nan) dataframe with one column per setindexname
            tot2 (int or nan) dataframe with one column per setindexname
            tot0_2 (int or nan) dataframe with one column per setindexname
            tot (int or nan) dataframe with one column per setindexname
            tot0 (int or nan) dataframe with one column per setindexname
            version (str)
        genes_df: 
            Dataframe of genes.GC file
        prnt_dbg: boolean
    Created:
        crudeOpGenes:
            DataFrame with cols 
                'Gene1', 'Gene2', (both str)
                'Sep' (int), 'bOp' - list<bool>,
                'begin1', 'end1', 'begin2', 'end2' (all int)
    
    Returns:
        fit_quality_df:
            Dataframe with cols:
                "nUsed" (int): Sum over all locusIds of the sum over the locusId (Sum of sum) per experiment
                "gMed" (int): Median of all locusIds over the sums per locusId per experiment
                "gMedt0" (int): Median over the time0 totals per experiment
                "gMean" (float): Mean of all locusIDs over the sums per locusId per experiment
                "cor12" (float): Correlation of normalized log ratios between first and second half 
                         gene insertion locations per experiment
                "mad12" (float): Medians of the Absolute values of the differences between first and second half
                            normalized log ratios.
                "mad12c" (float): Median of some statistic over all experiments
                "mad12c_t0" (float): Same as above but over the time0s
                "opcor" (float): Correlation between normalized log ratios of pairs of genes predicted to be 
                         in the same operon per experiment
                "adjcor" (float): Correlation between normalized log ratios of pairs of adjacent genes per experiment
                "gccor" (float): Correlation between normalized log ratios of genes and their GC percentage per experiment
                "maxFit" (float): The maximum normalized log ratio value per experiment



    Description:
        First we get the dataframe crudeOpGenes, which is similar to the genes_df dataframe
        in terms of its columns, but it is a dataframe with pairs of consecutive 
        genes from genes_df, with an extra column 'Sep' denoting the distance
        between the genes, and a column 'bOp' that says whether this distance
        is less than the median distance or not (True if less, False if greater).
        We get another similar dataframe called adj which are the adjacent pairs
        but without the Sep or bOp values. (Can we not use this dataframe?)
        Then we get the matching index between gene_fit_d genes (under the
        column name 'g') and the locusIds in genes_df. Using these matching index,
        we compute the correlation between each column in the normalized log ratios
        dataframe and the GC values of those genes.
        in genes_df. GC_corr is a pandas Series whose length is the number
        of experiments (columns) in gene_fit_d['lrn']; where the index
        labels are the experiment names, and the values are the Pearson 
        correlations. 
        Finally, we compute statistics regarding the experiments and return
        a dataframe with the following labels and meanings:
        "nUsed" (int): Sum over all locusIds of the sum over the locusId (Sum of sum) per experiment
        "gMed" (int): Median of all locusIds over the sums per locusId per experiment
        "gMedt0" (int): Median over the time0 totals per experiment
        "gMean" (float): Mean of all locusIDs over the sums per locusId per experiment
        "cor12" (float): Correlation of normalized log ratios between first and second half 
                         gene insertion locations per experiment
        "mad12" (float): Medians of the Absolute values of the differences between first and second half
                            normalized log ratios.
        "mad12c" (float): Median of some statistic over all experiments
        "mad12c_t0" (float): Same as above but over the time0s
        "opcor" (float): Correlation between normalized log ratios of pairs of genes predicted to be 
                 in the same operon per experiment
        "adjcor" (float): Correlation between normalized log ratios of pairs of adjacent genes per experiment
        "gccor" (float): Correlation between normalized log ratios of genes and their GC percentage per experiment
        "maxFit" (float): The maximum normalized log ratio value per experiment


    SubRoutines:
        CrudeOp: Gives dataframe that predicts whether adjacent genes 
                from genes_df are in same operon
        AdjacentPairs: Gives dataframe with adjacent genes
        df.corrwith: Takes each column of a dataframe and attempts to compute
                    correlation with a given series. Returns a pandas series
                    with index labels being the columns from original dataframe,
                    and values being the correlation. So the length of the
                    new series is the same as the number of the columns
                    in the original dataframe.
        paircor: this returns the correlation (float) between the normalized 
                log ratios of the pairs in one of the paired up dataframes
                crudeOpGenes_df and adjDiff.
    """
    crudeOpGenes_df = CrudeOp(genes_df)

    if prnt_dbg:
        crudeOpGenes_df.to_csv("tmp/py_crudeOpGenes_df.tsv", sep="\t")

    # adj is a dataframe
    adj = AdjacentPairs(genes_df, dbg_prnt=False)
    adjDiff = adj[adj['strand1'] != adj['strand2']]

    match_list = py_match(list(gene_fit_d['g']), list(genes_df['locusId']))

    #GC Correlation is the correlation between the fitnorm values and the GC values
    GC_Corr = gene_fit_d['lrn'].corrwith(genes_df.iloc[match_list]['GC'], method="pearson")

    # First and second half normalized log ratio values
    lrn1 = gene_fit_d['lrn1']
    lrn2 = gene_fit_d['lrn2']

    # Note axis=0 means we take sums over all rows, and the new rows come from the old columns
    # So for each of these, their length is the number of experiments analyzed
    fitQuality_df = pd.DataFrame.from_dict({
        "nUsed": gene_fit_d['tot'].sum(axis=0),
        "gMed": gene_fit_d['tot'].median(axis=0),
        "gMedt0": gene_fit_d['tot0'].median(axis=0),
        "gMean": gene_fit_d['tot'].mean(axis=0),
        "cor12": [lrn1[col_name].corr(lrn2[col_name]) for col_name in lrn1.head()],
        "mad12": (lrn1-lrn2).abs().median(),
        "mad12c": (np.log2(1 + gene_fit_d['tot1']) - np.log2(1 + gene_fit_d['tot2'])).abs().median(),
        "mad12c_t0": (np.log2(1 + gene_fit_d['tot0_1']) - np.log2(1 + gene_fit_d['tot0_2'])).abs().median(),
        # Remember crudeOpGenes_df['bOp'] is a list of bools
        "opcor": [paircor(crudeOpGenes_df[crudeOpGenes_df['bOp']], 
                  gene_fit_d['g'], 
                  gene_fit_d['lrn'][colname], 
                  method="spearman",
                  dbg_prnt=False) for colname in gene_fit_d['lrn']], 
        "adjcor": [paircor(adjDiff, gene_fit_d['g'], gene_fit_d['lrn'][colname], method="spearman", dbg_prnt=False)\
                    for colname in gene_fit_d['lrn']],
        "gccor": GC_Corr,
        "maxFit": gene_fit_d['lrn'].max()
        })
   
    if prnt_dbg:
        fitQuality_df.to_csv("tmp/py_fitQuality_df.tsv", sep="\t")

    return fitQuality_df, crudeOpGenes_df



def FEBA_Exp_Status(quality_df, status_d=None,
                    dbg_prnt=False):
    """
    Args:
        quality_df: A dataframe with cols:
            from FitReadMetrics:
                nMapped (int): Sum over all used experiments
                nPastEnd (int): Sum of pastEnd
                nGenic (int): Sum of reads within genes
            from FitQuality:
                "nUsed" (int): Sum over all locusIds of the sum over the locusId (Sum of sum) per experiment
                "gMed" (int): Median of all locusIds over the sums per locusId per experiment
                "gMedt0" (int): Median over the time0 totals per experiment
                "gMean" (float): Mean of all locusIDs over the sums per locusId per experiment
                "cor12" (float): Correlation of normalized log ratios between first and second half 
                         gene insertion locations per experiment
                "mad12" (float): Medians of the Absolute values of the differences between first and second half
                            normalized log ratios.
                "mad12c" (float): Median of some statistic over all experiments
                "mad12c_t0" (float): Same as above but over the time0s
                "opcor" (float): Correlation between normalized log ratios of pairs of genes predicted to be 
                         in the same operon per experiment
                "adjcor" (float): Correlation between normalized log ratios of pairs of adjacent genes per experiment
                "gccor" (float): Correlation between normalized log ratios of genes and their GC percentage per experiment
                "maxFit" (float): The maximum normalized log ratio value per experiment
            "name": (from exps_df) The name of the experiment (string)
            "short": (from exps_df) The shorthand name (string)
            "t0set": (from exps_df) associated control set (Time0) (string)
            ["num"]: (from_exps_df) experiment number from experiments file

            index labels for dataframe: experiment names (string)
        status_d: Dict containing threshold params 
    Returns:
        status_list (pandas Series(list<str>)): each status is from: {"OK", "Time0", "low_count", "high_mad12", 
                                                    "low_cor12", "high_adj_gc_cor"}
                                And each status corresponds to one experiment in quality_df (each row)
    Description:
        For each row in the Quality DataFrame, we check the values under 
        certain columns to decide how to label it's quality.
        The values we check, in this order, are:
        1. Is it a Time0? ("Time0")
        2. Does it have a low Median over the sums over locusIds? ("low_count")
            - param 'min_gMed'
        3. Does it have a high Median of the differences between first and
            second half log ratios? ("high_mad12") - param 'max_mad12' 
        4. Is there a low correlation between the first and second half
            log ratios? ("low_cor12") - param 'min_cor12'
        5. Is the GC correlation high or is the adjacent genes correlation
            high? ("high_adj_gc_cor") - params 'max_gccor' & 'max_adjcor'
        If none of these are True, then the experiment is given the status
        "OK". Otherwise, it returns the first value for which it is 
        True in the above questions, the value it returns is the
        string in parentheses.
        The whole function returns a pandas series the length of which
        is the number of rows in quality_df.

        Returns status of each experiment -- "OK" is a non-Time0 experiment that passes all quality metrics
        # Note -- arguably min_cor12 should be based on linear correlation not Spearman.
        # 0.1 threshold was chosen based on Marinobacter set5, in which defined media experiments with cor12 = 0.1-0.2
        # clearly worked, and Kang Polymyxin B (set1), with cor12 ~= 0.13 and they barely worked.
    """

    if status_d is not None:
        min_gMed = status_d["min_gMed"]
        max_mad12 = status_d["max_mad12"]
        min_cor12 = status_d["min_cor12"]
        max_gccor = status_d["max_gccor"]
        max_adjcor = status_d["max_adjcor"]
    else:
        min_gMed=50 
        max_mad12=0.5 
        min_cor12=0.1
        max_gccor=0.2 
        max_adjcor=0.25


    if dbg_prnt:
        print(quality_df.columns)
        print(quality_df.shape[0])
        print(quality_df.index)

    status_list = []
    # Each row corresponds to one experiment
    for ix, row in quality_df.iterrows():
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

    return pd.Series(data=status_list, index=quality_df.index) 


def CrudeOp(genes_df, dbg_out_file=None, dbg=False):
    """

    Args:
        genes_df is a dataframe which must have keys:
            locusId, begin, end
    Returns:
        DataFrame with cols 
            Gene1, Gene2, sysName1, type1, scaffoldId1, begin1, end1, strand1, name1, desc1, GC1, nTA1, 
            sysName2, type2, scaffoldId2, begin2, end2, strand2, name2, desc2, GC2, nTA2, Sep, bOp
        In this dataframe, each row compares consecutive genes from genes_df, so all the column
        names that end with '1' or '2' come from genes_df, except 'Gene' substitutes the original 
        'locusId' name. The value 'Sep' computes the distance between the two genes, and 
        'bOp' tests whether this distance is less than the median of 'Sep'. The total
        number of rows in this is the subset of pairs of genes that are on the same
        strand and scaffold.

    Description:

        Crude operon predictions -- pairs of genes that are on the same strand and
        separated by less than the median amount are predicted to be in the same operon
        
        We take the genes dataframe with genes sorted by location and then line them
        up offset by 1, so it would look like this:
                gene1, gene2,
                gene2, gene3,
                gene3, gene4,
                .
                .
                .
                geneX-1, geneX
        Where X is the (total number of genes in genes_df (from genes.GC)) - 1.
        Then we add the other columns of genes_df from the first X rows (omitting
        the last row) to get the gene's location (beginning and ending (begin1 and end1), 
        then we do the same again to get the beginning and ending of Gene2 
        (begin2 and end2). We also use their strand and scaffold columns in order to find
        consecutive genes on the same strand and column. 
        Then we remove all the rows of 2 genes that aren't on the same strand
        and same scaffoldId. 
        Then we compute the distance between the two consecutive genes
        on each remaining row. We load those distances
        into a column called 'Sep'. Then we take the median of that column
        'Sep', (i.e. the median of the distances between consecutive genes)
        and check if for every Sep value, is it greater than or less than
        this median. We load this boolean value into the column called
        'bOp'. Then we return this dataframe. The total number of rows in 
        this is the subset of pairs of genes that are on the same
        strand and scaffold.





    """
    # To assist with first merge we rename the column name locusId to Gene1
    # We offset all the locusIds by 1: First we ignore the last one, then we ignore the first
    # And place them side by side (Gene1, Gene2)
    # Note that the number of rows in g1_g2_df is 1 less than genes_df
    g1_g2_df =  pd.DataFrame.from_dict({
                            "Gene1": list(genes_df['locusId'].iloc[:-1]),
                            "Gene2": list(genes_df['locusId'].iloc[1:])
                            })


    mrg1 = g1_g2_df.merge(
                          genes_df, 
                          left_on="Gene1",
                          right_on="locusId",
                          how="inner")

    del mrg1["locusId"]
    del g1_g2_df


    # Now for the second merge we rename the column name Gene1 to Gene2
    new_df = mrg1.merge(
                        genes_df,
                        suffixes=["1","2"],
                        left_on="Gene2",
                        right_on="locusId",
                        how="inner")
    del mrg1, new_df['locusId']


    if dbg:
        print("CrudeOp new dataframe column names: " + \
                ", ".join(list(new_df.head())))

    if dbg_out_file is not None:
        new_df.to_csv( dbg_out_file + "second", sep="\t")


    st1_eq_st2 = [bool(
                        new_df['strand1'].iloc[i]==new_df['strand2'].iloc[i] and \
                        new_df['scaffoldId1'].iloc[i] == new_df['scaffoldId2'].iloc[i]
                    ) for i in range(new_df.shape[0])]

    print(f"Number of same strand/scaffold rows in CrudeOp predictions: {st1_eq_st2.count(True)}")

    # We remove unused rows
    pred_df = new_df[st1_eq_st2]

    # essentially the distance between the end of one gene and beginning of the next 
    paralmin = []
    for i in range(len(pred_df['begin1'])):
        paralmin.append(min(abs(pred_df['begin1'].iat[i] - pred_df['end2'].iat[i]), 
                            abs(pred_df['end1'].iat[i] - pred_df['begin2'].iat[i]) ))


    pred_df['Sep'] = paralmin
    # Below series is boolean (True/False), if the distance is less than median separation
    pred_df['bOp'] = pred_df['Sep'] < pred_df['Sep'].median()

    if dbg_out_file is not None:
        pred_df.to_csv( dbg_out_file + "third", sep="\t")

    return pred_df





def paircor(pairs, locusIds, values, use="p", method="spearman", names=["Gene1","Gene2"],
            dbg_prnt=False):
    """
    pairs (pandas DataFrame): dataframe with multiple cols (CrudeOp with TRUE cols from bOp)
        
    locusIds (pandas Series (str)): locusIds 
    values (pandas Series): normalized fitness scores 
    use: 
    method: Correlation method ("spearman")
    names (list<str>): "Gene1", "Gene2"
    dbg_prnt (bool)

    Description:
        # are values correlated for pairs? 
        # Intended for operon pairs but would work with other types too
        
        This is run on every experiment, so it will be run as many times
        as there are columns in the final log ratios normalized dataframe.
        We take the Crude Operon pairs (or Adjacent Pairs), but only the ones
        that are predicted to be in the same operon. 
        In our Crude Operon pairs, there are Two genes per row, so we get
        the normalized log ratio for the current two genes and place them
        under names "value1", "value2", which are associated to "Gene1",
        "Gene2" as you'd expect for this experiment. So now we have
        a dataframe with columns "Gene1", "Gene2", "value1", "value2",
        where "Gene1" and "Gene2" are locusIds (str) and the two values
        are their associated normalized log ratio values under the current
        experiment which we don't get in this function.
        Then we get the correlation between "value1" and "value2", i.e.
        the monotonic correlation between the normalized log ratios for the pairs
        under this experiment. We return this value (float)

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

    # This dataframe will have columns Gene1, Gene2, value1
    mrg1 = pairs[names].merge(premrg1, on="Gene1")



    premrg2 = pd.DataFrame.from_dict({
                "Gene2": list(locusIds),
                "value2": list(values)
                })

    mrg2 = mrg1.merge(premrg2, on="Gene2")


    if dbg_prnt:
        print('mrg2')
        print(mrg2)


    # method should be pearson
    res = mrg2['value1'].corr(mrg2['value2'], method=method)

    return res 


def normalize_per_strain_values(strains, genes_df, gene_fit_d):
    """
    Args:
        strains: all_df dataframe but just the metadata columns
                plus a column called 'used' which is StrainsUsed (boolean)
                and a column called enoughT0 which is also boolean
                testing if the means of the t0tot rows are big enough
        genes_df: Dataframe of genes.GC file
        gene_fit_d:
            'g': pandas Series of locusIds (str)
            'strain_lr': Length of this is nUsefulReads < nTotalReads ( = rows of all_df)
            'lrn':
            'lr':
            'strains':
                'scaffold'

    Returns:
        strain_lrn (pandas DataFrame): Normalized FitNorm values (?)
    
    SubRoutines:
        StrainClosestGenes
        create_strain_lrn

    Description:
        First, for every strain, we find the closest gene center from our list of used genes
        (genes that passed all the thresholds to be used). We call this list 'strainToGene'.
        So for strains that are within a gene, their closest gene will also be the gene
        that they are inserted into.
        Next, we create a dataframe that is the difference between the normalized log
        ratio values per experiment per gene, and the log ratio values per gene (that 
        are not normalized). 
        Then, for each strain, we normalize its fitness values in a complicated way.
        The way in which the fitness values are normalized are the following:
        create_strain_lrn:
            First, for each experiment, we take the per gene difference between 
            the normalized log ratio and the plain old log ratio, and then we 
            map those values onto all the strains using the Strain To Closest 
            Gene series we computed earlier. So for each strain, we take the 
            closest Gene and place the difference between the normalized 
            log fitness and the regular log fitness in a pandas Series we call 
            "per_strain_exp_diff". Then we group the strains by scaffold and
            take the medians of per_strain_exp_diff by these scaffolds, and for each
            strain, instead of having its own gene difference value, it instead now
            has the median of all the strain difference values in the same scaffold
            as it. Next we multiply that entire series by negative 1 and call it
            neg_str_scf_med, for 'negative strain per scaffold median'.
            Now we initialize the numbers we'll add to the original log ratios
            in order to get the normalized strain values. For each value in
            per_strain_exp_diff, if it's 'NA', then we insert the negative
            median of the differences from neg_str_scf_med, otherwise, we leave the
            value as it is (the per_strain_exp_diff value); we call the new series 'sdiff'.
            To get the final normalized log ratios for the strains under this 
            experiment, we simply add the original log ratios per strain to the values
            in sdiff, and that's our column for this experiment.
            The entire normalized log ratio per strain dataframe is one column
            per experiment name.

        
        
    """

    # strainToGene is pandas Series that has same length as num strains, 
    # and in which each index points to closest gene index by location
    """
    included_genes = [bool(x in gene_fit_d['g'].values) for x in genes_df['locusId'].values]
    reshuffled_genes = genes_df[included_genes].reset_index()
    print(reshuffled_genes)
    """
    strainToGene = NewStrainClosestGenes(strains, 
                                         gene_fit_d['g'],
                                         genes_df,
                                         dbg_prnt=False)

    # Subtract every value from log ratio normalized matrix by log ratio values.
    lrn_minus_lr = gene_fit_d['lrn'] - gene_fit_d['lr']
    strain_lrn = create_strain_lrn(gene_fit_d['strain_lr'], 
                                   lrn_minus_lr,
                                   strains, strainToGene,
                                   dbg_print=False)



    return strain_lrn, strainToGene

def NewStrainClosestGenes(strains, locusIds_used, genes_df, dbg_prnt=False):
    """
    
    Args:
        strains (pandas DataFrame):
            has all the meta columns from all_df and all the rows beneath them, (including 'scaffold')
            additionally contains columns:
                used: pandas Series(list<bool>): whose length is same as num of Trues in central_insert_bool_list
                enoughT0: Means of a subset of t0tots who pass the minT0 test.
                Length = nAllStrains 
        locusIds_used (pandas Series): contains all locusIds from the main dataframe
                                        in the final output order. 
                                        Length = nGenesUsed
        genes_df (pandas DataFrame): The dataframe created out of genes.GC,
                                must contain cols 'locusId', 'begin', 'end', 'scaffoldId'
                                        Length = nTotalGenes
    Returns:
        Strain_to_closest_gene_series (pandas Series): For each strain, the index (location)
                            within 'locusIds_used' which it is closest to. 
                            Length = nAllStrains


    Description:
        We find the closest gene (labelled by index of locusIds_used) to each
        strain. In order to do this efficiently, we need to sort the values
        in strains and genes_df so we can go one by one and check if the gene
        is close. But if we want to sort, we want to store the original indexes
        since those are what we will be needing (if we reshuffle the dataframes
        and find the closest genes to each strain then it would also take time
        to find the original index). 
        So we create a new column for both genes and strains called 
        'previous_index' in order to store what the indexes originally were.
        Then we group the genes and strains by scaffoldId, and iterate over
        the subsets of the genes and strains that are inside that scaffoldId.
        Every time we get a new scaffoldId, we select the subset of strains
        and genes that are in that scaffoldId, and we sort them in increasing
        position. Then we start by assuming the first strain in the subset
        is the closest to the first gene in it's subset, and check if it's
        closer to that gene than to the next one. As long as it's closer
        to the next gene than to the current gene, we don't finish labelling
        which gene it's closest to. Once it is closer to the current gene
        than to the next one, we mark both their 'previous_index' values
        in a list and move on to the next strain (keeping the current gene
        index). We do this for all scaffolds, thus finding the closest strain
        to each gene within it's scaffold, and return a pandas Series which
        has the same length as the strains DataFrame (nAllStrains), and the 
        value in each is the index of locusIds_used which it is closest to.

    """
    included_genes = [bool(x in locusIds_used.values) for x in genes_df['locusId'].values]
    sub_genes_df = genes_df[["locusId","scaffoldId","begin","end"]][included_genes]
    previous_index_list = []
    
    index_list = list(locusIds_used.index)
    values_list = list(locusIds_used.values)
    for val in sub_genes_df['locusId'].values:
        previous_index_list.append(index_list[values_list.index(val)])
    sub_genes_df['previous_index'] = previous_index_list
    sub_genes_df['midpoint'] = (sub_genes_df['begin'] + sub_genes_df['end'])/2
    strains["previous_index"] = strains.index

    # Reorganizing columns for the following algorithm
    strains = strains[["pos","previous_index","scaffold"]]

    # Now we group per scaffold and get the values
    strain_scf2ix = strains.groupby(by="scaffold").indices
    gene_scf2ix = sub_genes_df.groupby(by="scaffoldId").indices
    closest_gene_list = []
    for scaffoldId in strain_scf2ix.keys():
        print(f"Beginning to find closest used genes to strains for scaffold {scaffoldId}")
        # First we get the subsets of our dataframe that have this scaffold:
        strains_subset = strains.iloc[strain_scf2ix[scaffoldId]]
        genes_subset = sub_genes_df.iloc[gene_scf2ix[scaffoldId]]
        # then we sort them  by the values we care about to allow algorithm
        sorted_strains_subset = strains_subset.sort_values(["pos"], ascending=[True])
        sorted_genes_subset = genes_subset.sort_values(["midpoint"], ascending=[True])
        # For this algorithm, we start by assuming the closest gene is the first
        crnt_closest_gene_ix = 0
        midpoints = sorted_genes_subset['midpoint']
        final_gene_ix = len(midpoints) - 1
        for strain_row_tuple in sorted_strains_subset.itertuples():
            if crnt_closest_gene_ix == final_gene_ix:
                closest_gene_list.append([
                            strains_subset['previous_index'].iat[strain_row_tuple[0]],
                            genes_subset['previous_index'].iat[final_gene_ix]
                            ])        
            else:
                # strain_row_tuple[1] is the position of the current strain
                distance_to_current_gene = abs(strain_row_tuple[1] - midpoints.iat[crnt_closest_gene_ix])
                distance_to_next_gene = abs(strain_row_tuple[1] - midpoints.iat[crnt_closest_gene_ix + 1])
                while distance_to_next_gene < distance_to_current_gene:
                    crnt_closest_gene_ix += 1
                    if crnt_closest_gene_ix == final_gene_ix:
                        break
                    distance_to_current_gene = abs(strain_row_tuple[1] - midpoints.iat[crnt_closest_gene_ix])
                    distance_to_next_gene = abs(strain_row_tuple[1] - midpoints.iat[crnt_closest_gene_ix + 1])

                closest_gene_list.append([
                            strains_subset['previous_index'].iat[strain_row_tuple[0]],
                            genes_subset['previous_index'].iat[crnt_closest_gene_ix]
                            ])

    
    print(f"Finished finding closest genes for all scaffolds")
    strain_index = [x[0] for x in closest_gene_list]
    gene_index = [x[1] for x in closest_gene_list]
    Strain_to_closest_gene_series = pd.Series(gene_index, index=strain_index)
    Strain_to_closest_gene_series.sort_index(inplace=True)
    return Strain_to_closest_gene_series
        



def create_strain_lrn(strain_lr, gdiff, strains, strainToGene, dbg_print=False):
    """ We normalize per strain values
    Args:
        strain_lr:  
            (comes from strain_lr) (float): dataframe with one column per setindexname
                Num rows is nAllStrains len(all_df)
        gdiff:  dataframe (float) with one column per setindexname (same length as main_df-
                                which is equivalent to the number of unique locusIds that are used)
                                (lrn - lr)
                                len: nGenesUsed
        strains: all_df dataframe but just the metadata columns. Needs column 'scaffold'
                    Num rows (length) nAllStrains
        strainToGene pandasSeries<index>: For each strain, the index of the closest
                                            gene center. Length: nAllStrains 
    Returns:
        pandas DataFrame with normalized strain values per experiment
        (Num columns = nExperiments that are used), nRows = nAllStrains.

    Description:
        First, for each experiment, we take the per gene difference between 
        the normalized log ratio and the plain old log ratio, and then we map those values
        onto all the strains using the Strain To Closest Gene series we computed
        earlier. So for each strain, we take the closest Gene and map that value (the difference
        between the normalized log fitness and the regular log fitness) to a list
        we call "per_strain_exp_diff". Then we group the strains by scaffold and
        take the medians of per_strain_exp_diff by these scaffolds, and for each
        strain, instead of having it's own gene difference value, it instead now
        has the median of all the strain difference values in the same scaffold
        as it. Next we multiply that entire series by negative 1 and call it
        neg_str_scf_med, for 'negative strain per scaffold median'.
        Now we initialize the numbers we'll add to the original log ratios
        in order to get the normalized strain values. For each value in
        per_strain_exp_diff, if it's 'NA', then we insert the negative
        median of the differences from neg_str_scf_med, otherwise, we leave the
        value as it is (the per_strain_exp_diff value); we call the new series 'sdiff'.
        To get the final normalized log ratios for the strains under this 
        experiment, we simply add the original log ratios to the values
        in sdiff, and that's our column for this experiment.
        The entire normalized log ratio per strain dataframe is one column
        per experiment name.
        
    """

    # We check that the column names (experiment names) match between the two dataframes    
    for i in range(len(strain_lr.columns)):
        if strain_lr.columns[i] != gdiff.columns[i]:
            raise Exception("Column names are not matching each other."
                            f" {strain_lr.columns[i]} != {gdiff.columns[i]}")

    results = {}
    for exp_name in strain_lr.columns:
        print("currently working on normalizing strain fitness values for"
              f" experiment {exp_name}")
        # experiment log ratio column (pandas Series)
        exp_lr = strain_lr[exp_name]
        # experiment diff between normalized and not (pandas Series)
        exp_diff = gdiff[exp_name]
        # The length of this series will be nAllStrains, with multiple repeats 
        per_strain_exp_diff = exp_diff.iloc[strainToGene]
        per_strain_exp_diff.reset_index(inplace=True, drop=True)
        # We get the medians of the strain log ratios grouped by scaffoldId 
        strain_scaffold_ix2_medians = []
        scaffold_indices = strains.groupby(by="scaffold").indices
        for index_group in scaffold_indices.values():
            crnt_median = exp_lr.iloc[index_group].median()
            for ix in index_group:
                strain_scaffold_ix2_medians.append([ix,crnt_median])
        medians = [x[1] for x in strain_scaffold_ix2_medians]
        indices = [x[0] for x in strain_scaffold_ix2_medians]
        strain_scaffold_medians = pd.Series(medians, index=indices)
        neg_str_scf_med = -1 * strain_scaffold_medians
        
        sdiff = []
        for ix, val in per_strain_exp_diff.iteritems():
            if pd.isnull(val):
                sdiff.append(neg_str_scf_med.iat[ix])
            else:
                sdiff.append(per_strain_exp_diff.iat[ix])
        sdiff = pd.Series(sdiff) 
        results[exp_name] = exp_lr + sdiff

    return pd.DataFrame.from_dict(results)
                
                




        




    '''
    results = {}
    tot_num_cols = len(strain_lr.columns)
    # We iterate over every column in both dataframes strain_lr & gdiff
    for i in range(tot_num_cols):
        strain_lr_set_index_name = list(strain_lr.columns)[i]
        print(f"Currently working on column {strain_lr_set_index_name}, {i + 1}/{tot_num_cols}")
        gdiff_set_index_name = list(gdiff.columns)[i]
        if strain_lr_set_index_name != gdiff_set_index_name:
            raise Exception("Columns are not matching each other."
                            f" {strain_lr_set_index_name} != {gdiff_set_index_name}")
        strain_lr_col = strain_lr[strain_lr_set_index_name]
        gdiff_col = gdiff[gdiff_set_index_name]
        
        # What happens here ??
        # Here we want to copy each value from the gdiff_col,
        # and create a Series the length of all_df
        sdiffGene = [gdiff_col.iat[strainToGene.iat[j]] for j in range(len(strainToGene.values))]
        # Does this require all values in strains['scaffolds'] to be in strain_lr_col?
        grouped_strain_lr = dict(strain_lr_col.groupby(by=strains['scaffold']).indices)

        # We want the median to be the same for every value under a group
        # of indices
        sdiffSc_d = {}
        for group_label, indices in grouped_strain_lr.items():
            current_median = strain_lr_col.iloc[indices].median()
            if dbg_print:
                for ix in indices:
                    if ix in sdiffSc_d:
                        raise Exception("Reoccuring index value (?)")
                    sdiffSc_d[ix] = -1*current_median
            else:
                for ix in indices:
                    sdiffSc_d[ix] = -1*current_median

        sdiffSc = [sdiffSc_d[j] for j in sorted(sdiffSc_d.keys())]
        sdiffSc = pd.Series(sdiffSc, index = sorted(sdiffSc_d.keys()))

        print(len(sdiffGene))
        print(len(sdiffSc))
        sdiff = [sdiffSc[j] if pd.isnull(sdiffGene[j]) else sdiffGene[j] \
                 for j in sdiffSc.index]
        # You can add a series and a list vectorwise, but will make sdiff a series anyways:
        results[strain_lr_set_index_name] = strain_lr_col + pd.Series(sdiff)

    return pd.DataFrame.from_dict(results)
    '''


def AdjacentPairs(genes_df, dbg_prnt=False):
    """
    Args:
        genes_df pandas DataFrame of genes.GC tsv
    Returns:
        DataFrame with the following cols:
            Gene1, Gene2, sysName1, type1, scaffoldId, begin1, end1, strand1, name1, desc1, GC1, 
            nTA1, sysName2, type2, begin2, end2, strand2, name2, desc2, GC2, nTA2
        Length of dataframe is at first the length of genes.GC - 1, because we get all pairs
        of consecutive genes, but then we also remove all consecutive genes that aren't 
        on the same scaffold.

    Description:
        Similar to CrudeOp, we get the consecutive pairs of genes from genes_df and create
        rows (length of genes_df - 1) with all of them, only removing rows that contain a 
        pair of genes that aren't on the same scaffold.
    """
    # get genes in order of scaffoldId and then tiebreaking with increasing begin
    c_genes_df = genes_df.copy(deep=True).sort_values(by=['scaffoldId', 'begin'])

    # We offset the genes with a loop starting at the first
    g1_g2_df = pd.DataFrame.from_dict({
            "Gene1": list(c_genes_df['locusId']),
            "Gene2": list(c_genes_df['locusId'].iloc[1:]) + [c_genes_df['locusId'].iloc[0]]
        })

    # add metadata and only keep pairs with same scaffold
    mg1 = g1_g2_df.merge(c_genes_df, left_on="Gene1", right_on="locusId")
    del mg1['locusId']
    adj = mg1.merge(c_genes_df,
                    left_on="Gene2",
                    right_on="locusId",
                    suffixes = ["1","2"]
                    )
    adj = adj[adj['scaffoldId1'] == adj['scaffoldId2']]
    del(adj['locusId'])
    
    if dbg_prnt:
        adj.to_csv("tmp/py_AdjacentPairsOutput.tsv", sep="\t")

    return adj



def OLDStrainClosestGenes(strains, locusIds_used, genes_df, dbg_prnt=False):
    """ 
    Args:
        strains (pandas DataFrame):
            has all the meta columns from all_df and all the rows beneath them, (including 'scaffold')
            additionally contains columns:
                used: pandas Series(list<bool>): whose length is same as num of Trues in central_insert_bool_list
                enoughT0: Means of a subset of t0tots who pass the minT0 test.
        genes (pandas DataFrame): same as genes_df, but with a switched order of locusIds. 
                                Needs to have columns: 
                                    scaffoldId, begin, end

    Intermediate Vars:
        indexSplit (python dict): group_label (scaffoldId) -> list of values (int or np.nan)
            
    Returns:
        pandas Series: Length of 'strains' (all_df), for each row of all_df, we return the index of 
                        the closest gene from 'genes', by taking halfway between each gene's beginning
                        and ending position and comparing it to the position of the strain barcode insertion.
    Description:
        For each strain (barcode in all.poolcount), find the closest gene as listed, in gene_fit_d['g']
        as an index label from 
        'genes' dataframe.
        returns a list, same length as strain, with corresponding strain rows -> closest row within genes
        If there is no gene on that scaffold, returns NA.

    SubRoutines:
        py_unsplit

    * Below can be optimized with multithreading
    """
    
    # A list of integers parallel to rows of genes_df
    genes_index = list(range(0, genes.shape[0]))

    # Below are both numpy arrays
    unq_strain_scaffolds = strains['scaffold'].unique()
    unq_gene_scaffolds = genes['scaffoldId'].unique()

    # Below are both pandas groupby objects
    strainGroupBy = strains.groupby(by='scaffold')
    geneGroupBy = genes.groupby(by='scaffoldId')

    indexSplit = {}
    # We iterate over the unique scaffold Ids from 'strains'
    for scaffoldId in unq_strain_scaffolds:
        crnt_scf_df = strainGroupBy.get_group(scaffoldId)
        if scaffoldId not in unq_gene_scaffolds:
            crnt_genes_df = pd.DataFrame()
        else:
            crnt_genes_df = geneGroupBy.get_group(scaffoldId)

        nGenes_rows = crnt_genes_df.shape[0]
        nStrains_rows = crnt_scf_df.shape[0]
        if  nGenes_rows == 0:
            indexSplit[scaffoldId] = [np.nan]*nStrains_rows
        elif nGenes_rows == 1:
            indexSplit[scaffoldId] = [geneGroupBy.groups[scaffoldId][0]]*nStrains_rows
        else:
            # We get the centers of all the genes
            crnt_genes_df['pos'] = (crnt_genes_df['begin'] + crnt_genes_df['end']) / 2

            # Now we find the location of the strain and capture the closest gene center
            # This is the part that could be multithreaded/ sorted
            crnt_scaffold_list = []
            if dbg_prnt:
                print(f"Now finding closest gene for {crnt_scf_df.shape[0]} values")
            count = 0

            print(f"Starting to find strain closest genes for scaffold: {scaffoldId}")
            total_rows = crnt_scf_df.shape[0]
            time_stamp = time.time()
            # ix is an int
            for ix, row in crnt_scf_df.iterrows():
                if ix % 5000 == 0 and int(ix) != 0:
                        rows_remaining = total_rows - ix
                        amount_5000_left = rows_remaining/5000
                        print(f"Currently at count {ix} in Strain Closest Genes for"
                              f" scaffoldId {scaffoldId}.\n"
                              f"Number of rows remaining: {rows_remaining}.\n"
                              f"Time Remaining: {(time.time() - time_stamp)*amount_5000_left}"
                                " seconds.\n")
                        time_stamp = time.time()

                gene_pos_minus_strain_pos = (crnt_genes_df['pos'] - row['pos']).abs()
                # we get the index of the minimum value
                crnt_scaffold_list.append(int(gene_pos_minus_strain_pos.idxmin()))
            print("Done finding closest genes to strains.")
        
            if dbg_prnt:
                with open("tmp/py_crnt_scaffold_list.json", "w") as g:
                    g.write(json.dumps([int(x) for x in crnt_scaffold_list], indent=2))

            indexSplit[scaffoldId] = crnt_scaffold_list
    
    new_indexSplit = {}
    for key, vals in indexSplit.items():
        new_indexSplit[key] = [None if val==np.nan else val for val in vals]
    indexSplit = new_indexSplit
        

    if dbg_prnt:
        with open("tmp/py_indexSplit.json", 'w') as g:
            g.write(json.dumps(indexSplit, indent=2))
    
    # py_unsplit returns Series
    recombined_series = py_unsplit(indexSplit, strains['scaffold']) 
    if dbg_prnt:
        recombined_series.to_csv("tmp/py_recombined_series.tsv", sep="\t")

    return recombined_series
