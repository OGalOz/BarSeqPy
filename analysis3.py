
import os
import logging
import pandas as pd
import numpy as np
from scipy import stats
import json
import math 
import time
from translate_R_to_pandas import * 

def analysis_3(gene_fit_d, GeneFitResults, genes_df, all_df, exps_df,
               genesUsed, strainsUsed, genesUsed12,
               t0_gN, t0tot, CrudeOp_df, central_insert_bool_list,
               meta_ix=7, minT0Strain=3, dbg_prnt=False):
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

    # Start Shift
    gene_fit_d['genesUsed'] = genesUsed
    gene_fit_d['strainsUsed'] = strainsUsed
    gene_fit_d['genesUsed12'] = genesUsed12
    gene_fit_d['gN'] = get_all_gN(all_df, central_insert_bool_list, meta_ix)
    gene_fit_d['t0_gN'] = t0_gN
   
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




def get_all_gN(all_df, central_insert_bool_list, meta_ix):

    # We get the subset of the experiments in all_df who have central genes
    tmp_all_df = all_df.iloc[:,meta_ix:][central_insert_bool_list]
    tmp_all_df['locusId'] = all_df['locusId'][central_insert_bool_list]
    # all_gN is a dataframe with unique locusId values with sums
    all_gN = py_aggregate(tmp_all_df, "locusId", func="sum")

    return all_gN









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
    """
    tmp_df = gene_fit_d['q'][gene_fit_d['q']['u']].merge(exps_df, on=["name","short"])
    gene_fit_d['specphe'] = SpecificPhenotypes(gene_fit_d['g'], 
                            tmp_df, gene_fit_d['lrn'][gene_fit_d['q']['u']], 
                            gene_fit_d['t'][gene_fit_d['q']['u']], dbg_prnt=True)
    """

    return gene_fit_d


def AdjacentPairs(genes_df, dbg_prnt=False):
    """
    Args:
        genes_df pandas DataFrame of genes.GC tsv
    Returns:
        DataFrame with the following cols:
            Gene1, Gene2, sysName1, type1, scaffoldId, begin1, end1, strand1, name1, desc1, GC1, 
            nTA1, locusId, sysName2, type2, begin2, end2, strand2, name2, desc2, GC2, nTA2

    Description: We sort the genes and line them up next to each other
                so we get adjacent pairs in a dataframe, e.g.:
                    Gene1, Gene2,
                    Gene2, Gene3,
                    .
                    .
                    .
                    GeneN-1, GeneN
                    GeneN, Gene1
    """
    
    logging.info("Getting adjacent pairs")

    # get genes in order of scaffoldId and then tiebreaking with increasing begin
    c_genes_df = genes_df.sort_values(by=['scaffoldId', 'begin'])

    # We offset the genes with a loop starting at the first
    adj = pd.DataFrame.from_dict({
            "Gene1": list(c_genes_df['locusId']),
            "Gene2": list(c_genes_df['locusId'].iloc[1:]) + [c_genes_df['locusId'].iloc[0]]
        })


    c_genes_df = c_genes_df.rename(columns={"locusId": "Gene1"})

    mg1 = adj.merge(c_genes_df, by="Gene1")

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

    return out_df



def HighFit(gene_fit_d, genes_df, exps_df, min_fit=4, min_t=5, max_se=2, 
            min_gMean=10, max_below=8, dbg_prnt=False):
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
            
    Subroutines:
        py_order: (from translate_R_to_pandas)
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

    #where_are_high_rows = [x[0] for x in where_is_high]
    #where_are_high_cols = [x[1] for x in where_is_high]

    fit_high = []
    t_high = [] 
    experiment_names = []
    locusIds = []
    for x in where_is_high:
            fit_high.append(lrn.iloc[x[0], x[1]])
            t_high.append(t.iloc[x[0], x[1]])
            experiment_names.append(lrn.columns[x[1]])
            if isinstance(gene_fit_d['g'], pd.DataFrame):
                locusIds.append(gene_fit_d['g']['locusId'][x[0]])
            elif isinstance(gene_fit_d['g'], pd.Series):
                locusIds.append(gene_fit_d['g'][x[0]])
            else:
                raise Exception("Cannot recognize type of gene_fit_d['g']")



    high_df = pd.DataFrame.from_dict({
                    # x[0] -> rows from where_is_high
                    "locusId": locusIds,
                    # x[1] -> columns from where_is_high
                    "expName": experiment_names,
                    "fit": fit_high,
                    "t": t_high 
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
    # Adding the two dataframes
    new_high = new_high.append(genes_df[["locusId","sysName","desc"]])
    new_high = new_high.iloc[py_order(high_df['expName'], tie_breaker=(-1*high_df['fit']))]
    
    if dbg_prnt:
        new_high.to_csv("tmp/py_new_high_df.tsv", sep="\t", index=False)

    return new_high


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
    
    Subroutines:
        StrainClosestGenes
        create_strain_lrn
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
                                   gene_fit_d, strainToGene, dbg_prnt=True)


    return strain_lrn


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

    Subroutines:
        py_split (translate_R_to_pandas)

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
            
            if dbg_prnt:
                with open("tmp/py_crnt_scaffold_list.json", "w") as g:
                    g.write(json.dumps([int(x) for x in crnt_scaffold_list], indent=2))

            indexSplit[scaffoldId] = crnt_scaffold_list


    
    recombined_series = py_unsplit(indexSplit, strains['scaffold']) 
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

    Subroutines:
        None
    """

    if dbg_print:
        print(sfit)
        print(gdiff)
  
    print("sfit")
    print(sfit)
    print("gdiff")
    print(gdiff)
    stop(583)


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


def stop(line_num):
    raise Exception(f"Stopped, line {line_num}") 
