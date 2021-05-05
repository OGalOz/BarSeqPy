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


def analysis_2(GeneFitResults, exps_df, all_df, genes_df, central_insert_bool_list,
               strainsUsed_list, t0tot,
               meta_ix=7, debug=False, minT0Strain=3):
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



    #save_gene_fit_d(gene_fit_d, prnt_dbg=False)

    if debug:
        print("Running FitReadMetrics() and FitQuality()")
    st = time.time()
    fitreadmet = FitReadMetrics(all_df, qnames, central_insert_bool_list)
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


    # Creating strains dataframe
    strains = all_df.iloc[:,0:meta_ix]

    strains['used'] = strainsUsed_list 
    strains['enoughT0'] = t0tot[t0tot > minT0Strain].mean()
    gene_fit_d['strains'] = strains

    gene_fit_d['strain_lr'] = pd.DataFrame.from_dict(
                            {x: list(GeneFitResults[x]['strain_fit']) for x in GeneFitResults.keys()}
                            )
    gene_fit_d['strain_se'] = pd.DataFrame.from_dict(
                            {x:list(GeneFitResults[x]['strain_se']) for x in GeneFitResults.keys()}
                            )
    
    strain_lrn, strainToGene = normalize_per_strain_values(strains, genes_df, gene_fit_d)
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


def FitReadMetrics(all_df, qnames, central_insert_bool_list):
    """
    Args:
        all_df (pandas DataFrame):
        qnames (pandas Series): list<str> (names of set_index_names)
        central_insert_bool_list list<bool>: gene insertion between 0.1 and 0.9 fraction of length 
    
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
        "nGenic": all_df[central_insert_bool_list][qnames].sum(axis=0)
        })
    frm_df.index = list(qnames)
    return frm_df



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


def normalize_per_strain_values(strains, genes_df, gene_fit_d):
    """
    Args:
        strains: all_df dataframe but just the metadata columns
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


    return strain_lrn, strainToGene


def StrainClosestGenes(strains, genes, dbg_prnt=False):
    """ 
    Args:
        strains (pandas DataFrame):
            has all the meta columns from all_df and all the rows beneath them, (including 'scaffold')
            additionally contains columns:
                used: pandas Series(list<bool>): whose length is same as num of Trues in central_insert_bool_list
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

            print(f"Starting to find strain closest genes for scaffold: {scaffoldId}")
            total_rows = s.shape[0]
            time_stamp = time.time()
            for ix, row in s.iterrows():
                if count % 5000 == 0 and count != 0:
                        rows_remaining = total_rows - count
                        amount_5000_left = rows_remaining/5000
                        print(f"Currently at count {count} in Strain Closest Genes for"
                              f" scaffoldId {scaffoldId}.\n"
                              f"Number of rows remaining: {rows_remaining}.\n"
                              f"Time Remaining: {(time.time() - time_stamp)*amount_5000_left}"
                                " seconds.\n")
                        time_stamp = time.time()

                gene_pos_minus_strain_pos = (g['pos'] - row['pos']).abs()
                # we get the index of the minimum value
                crnt_scaffold_list.append(gene_pos_minus_strain_pos.idxmin())
                count += 1
            print("Done finding closest genes to strains.")
            
            if dbg_prnt:
                with open("tmp/py_crnt_scaffold_list.json", "w") as g:
                    g.write(json.dumps([int(x) for x in crnt_scaffold_list], indent=2))

            indexSplit[scaffoldId] = crnt_scaffold_list


    
    recombined_series = py_unsplit(indexSplit, strains['scaffold']) 
    if dbg_prnt:
        recombined_series.to_csv("tmp/py_recombined_series.tsv", sep="\t")

    return recombined_series


def create_strain_lrn(sfit, gdiff, gene_fit_d, strainToGene, dbg_print=False):
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

    if dbg_print:
        print("Creating strain_lrn (strain log ratios normalized).")
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
        grouped_sfit = dict(sfit_col.groupby(by=gene_fit_d['strains']['scaffold']).groups)
        sdiffSc = [(-1*sfit_col[grouped_sfit[group_label]].median() ) \
                        for group_label in grouped_sfit]
        sdiff = sdiffSc if sdiffGene is None else sdiffGene
        results[sfit_set_index_name] = sfit_col + sdiff

    return pd.DataFrame.from_dict(results)


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
