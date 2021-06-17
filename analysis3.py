
import os
import logging
import pandas as pd
import numpy as np
from scipy import stats
import json
import math 
import random
import time
from translate_R_to_pandas import * 

def analysis_3(gene_fit_d, GeneFitResults, genes_df, 
               all_df, exps_df,
               genesUsed, strainsUsed, genesUsed12,
               t0_gN, t0tot, CrudeOp_df, 
               meta_ix=7, 
               cfg=None,
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
            tot0_1 (int or nan) dataframe with one column per setindexname
            tot2 (int or nan) dataframe with one column per setindexname
            tot0_2 (int or nan) dataframe with one column per setindexname
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
                    tot0_1, tot2, tot0_2, tot, tot0
                strain_fit: pandas Series (float) with a computation applied to values
                strain_se: pandas Series (float) with a computation applied to values
        strainsUsed pandas Series(list<bool>):  
        CrudeOp_df (pandas DataFrame): Output from function CrudeOp(genes_df)
                Gene2, Gene1, sysName1, type1, scaffoldId1, begin1, end1, strand1, name1, desc1, GC1, nTA1, 
                sysName2, type2, scaffoldId2, begin2, end2, strand2, name2, desc2, GC2, nTA2, Sep, bOp
        minCofitExp (int): The minimum number of good experiments in order for us to 
                            compute the cofitness values.

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

    Description:
        First, we store some variables for outputs:
            genesUsed: list<str> The locusIds, e.g. names, of the genes which
                                passed qualifications for being included
                                in the analysis.
            strainsUsed: list<bool> A list of booleans, which is the same length
                                as the number of rows in all.poolcount, which
                                has True for every strain that we include in
                                computing GeneFitness, and False otherwise.
            genesUsed12: list<str> The locusIds of the genes that passed qualifications
                                    for being included in the analysis as well as
                                    having abundant enough insertions on both 
                                    halves, i.e. enough insertions in the 0.1 ->0.5
                                    fraction of the gene, and enough insertions in
                                    the 0.5 -> 0.9 fraction of the gene.
            t0_gN: A dataframe with the t0 totals over locusIds, may be more
                    genes than nGenesUsed due to not omitting genes whose
                    numbers don't pass minT0GenesUsed. Index label
                    is locusId.
            gN: A dataframe with all the gene totals, ignoring genes that
                don't pass qualifications. Any gene that is in all_df's locusId
                column is included. Same number of rows as t0_gN. Index label
                is locusId.
        Next, in summary, we add two sets of dataframes: DataFrames that compute cofitness
        (meaning correlation between genes), and DataFrames that compute special
        values (i.e. specific situations (experiment and gene) that have exceptional
        fitness, either good or bad). The dataframes that are related to cofitness
        which we add are the following:
            "pairs", "cofit". Where "pairs" contains
            cofitness between specific pairs of genes (adjacent genes,
            genes predicted to be in same operon, and randomly chosen genes), 
            "cofit" contains
            the top cofit genes for each gene. We can define how many
            are included in the 'top cofit' by inputting the variable
            nTopCofit to the function analysis3. If it's None, then
            the program decides how many that is.
        The dataframes related to special values are
            "specphe", and "high"
            "specphe" contains the instances where we found specific 
            phenotypes for strains within experiments (which gene and which 
            experiment).
            "high" contains instances where we find both fitness values AND
            t scores that pass thresholds as well as a number of other
            columns from the experiments dataframe and the genes dataframe.

        The function ends and we return gene_fit_d.
            
    """
    if cfg is not None:
        compute_cofit_bool = cfg["compute_cofit_bool"]
        compute_High_bool = cfg["compute_High_bool"]
        nTopCofit = cfg["nTopCofit"]
        minT0Strain = cfg["minT0Strain"]
        minCofitExp = cfg["minCofitExp"]
    else:
        compute_cofit_bool=True
        compute_High_bool=True
        nTopCofit=None
        minT0Strain=3 
        minCofitExp=5
        cfg = {
                "spec_cfg": None,
                "high_cfg": None
        }


    gene_fit_d['genesUsed'] = genesUsed
    gene_fit_d['strainsUsed'] = strainsUsed
    gene_fit_d['genesUsed12'] = genesUsed12
    gene_fit_d['t0_gN'] = t0_gN
    # gN is a dataframe with as many columns as experiments and as many
    # rows as there are unique locusIds in the all_df column 'locusId'
    gene_fit_d['gN'] = get_all_gN(all_df, meta_ix)
  

    if compute_cofit_bool:
        # (u_true is an int)
        u_true = list(gene_fit_d['q']['u']).count(True)
        if dbg_prnt:
            print(f"u_true: {u_true}")
        if u_true >= minCofitExp:
            logging.info("Computing cofitness with {u_true} experiments")
            gene_fit_d = compute_cofit(gene_fit_d, genes_df, CrudeOp_df, exps_df, 
                                        cfg, nTopCofit=nTopCofit)
        else:
            logging.info(f"Only {u_true} experiments of {gene_fit_d['q'].shape[0]} passed quality filters!")

    if compute_High_bool:
        gene_fit_d['high'] = HighFit(gene_fit_d, genes_df, exps_df, all_df,
                                    high_cfg=cfg["high_cfg"],
                                    dbg_prnt=True)

    return gene_fit_d




def get_all_gN(all_df, meta_ix):
    """
    Description:
        We get the sums over unique locus Ids and reads centrally within
        those genes. For the returned dataframe the index labels are
        the unique 'locusIds', the remaining values are the sums over 
        all the rows that had that locusId (for each column.)
        The remaining columns are the experiment names.

    Returns:
        all_gN (pandas DataFrame): num rows = num unique genes in all_df
                                    'locusId' column. num columns = num
                                    experiments. The values are the sums
                                    over the locusIds.
    """


    central_insert_bool_list = [True if (0.1<=x<=0.9) else False for x in all_df['f']]

    # We get the subset of the experiments in all_df who have central genes
    tmp_all_df = all_df.iloc[:,meta_ix:][central_insert_bool_list]
    tmp_all_df['locusId'] = all_df['locusId'][central_insert_bool_list]
    # all_gN is a dataframe with unique locusId values with sums
    all_gN = tmp_all_df.groupby("locusId").sum()

    return all_gN









def compute_cofit(gene_fit_d, genes_df, CrudeOp_df, exps_df, 
                  cfg,
                  nTopCofit=None):
    """
    Args:

        gene_fit_d: Required keys:
            'lrn' (pandas DataFrame, length is n
            'q' (pandas DataFrame):
                'u': (bool)
            'g' (pandas Series):
            't' (pandas DataFrame float):
        genes_df: genes.GC pandas DataFrame

        CrudeOp_df (pandas DataFrame):
            Gene2, Gene1, sysName1, type1, scaffoldId1, begin1, end1, strand1, name1, desc1, GC1, nTA1, 
            sysName2, type2, scaffoldId2, begin2, end2, strand2, name2, desc2, GC2, nTA2, Sep, bOp

        cfg (python dict):
            spec_cfg (d): None or keys specified in SpecificPhenotypes function




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
    
    Description:
        We compute a few dataframes to add to the output. The outputs
        are described throughout and listed at the bottom of the description.
        First, we get the dataframe 'adj' out of AdjacentPairs. This dataframe
        contains adjacent genes in each row with the location of both.
        Its length is the same as the length of genes_df - 1.
        Then we remove pairs who aren't on the same strand and call the
        dataframe adjDiff.
        We compute the correlation between adjacent genes (the correlation
        is on the normalized log ratio fitness values per gene).
        We do the same as we did for Adjacent Pairs for the Crude Operon
        prediction genes (genes predicted to be on the same operon).
        We do this again for Random pairs of genes, while making sure
        we aren't computing correlation between a gene and itself.
        Then we find the nTopCofit values, i.e. the top cofitness between
        genes for nTopCofit (an integer) values. If nTopCofit isn't given
        as an input, then nTopCofit is computed within the function TopCofit
        automatically (the variable 'n'). This returns a dataframe with
        nTopCofit*nGenesUsed rows, since for every gene in genesUsed,
        we return the top nTopCofit cofitness values and genes.
        Finally, we compute 'SpecificPhenotypes', which asks
        which exact strains under which experiments behaved abnormally,
        or in a way that we are interested in. This returns a dataframe
        with as many rows as abnormal results, which may be very few.
        We return gene_fit_d with all the new dataframes we created:
            "pairs", "cofit", and "specphe". Where "pairs" contains
            cofitness between specific pairs of genes (adjacent,
            predicted to be in same operon, random), "cofit" contains
            the top cofit genes for each gene, and "specphe" contains
            the instances where we found specific phenotypes for 
            strains within experiments (which gene and which 
            experiment).
    """
    print("Computing Cofit!")
    used_experiment_names = list(gene_fit_d['q']['name'][gene_fit_d['q']['u']])

    adj = AdjacentPairs(genes_df)
    adjDiff = adj[adj['strand1'] != adj['strand2']]
    # rfit is correlation between genes on both of these.
    adjDiff['rfit'] = cor12(adjDiff, gene_fit_d['g'], gene_fit_d['lrn'][used_experiment_names])
    CrudeOp_df['rfit'] = cor12(CrudeOp_df, gene_fit_d['g'], gene_fit_d['lrn'][used_experiment_names])

    # We create random correlations between various genes and check values.
    g_len = len(gene_fit_d['g'])
    # We get 'k' random numbers from the list
    sample1 = random.choices(list(range(g_len)), k = g_len*2)
    sample2 = random.choices(list(range(g_len)), k = g_len*2)
    rand1 = gene_fit_d['g'].iloc[sample1]
    rand2 = gene_fit_d['g'].iloc[sample2]
    #rand2 = gene_fit_d['g'].sample(n=len(gene_fit_d['g'])*2, replace=True)
    # Random gene correlation
    random_df = pd.DataFrame.from_dict({
                    "Gene1": rand1.values, 
                    "Gene2": rand2.values 
                })
    # We make sure we aren't computing the correlation between the same two genes.
    random_df = random_df[random_df['Gene1'] != random_df['Gene2']]
    random_df['rfit'] = cor12(random_df, gene_fit_d['g'], gene_fit_d['lrn'][used_experiment_names])

    gene_fit_d['pairs'] = {"adjDiff": adjDiff, 
                            "pred": CrudeOp_df, 
                            "random": random_df }

    gene_fit_d['cofit'] = TopCofit(gene_fit_d['g'], gene_fit_d['lrn'][used_experiment_names],
                                    pre_n=nTopCofit)

    tmp_df = gene_fit_d['q'][gene_fit_d['q']['u']].merge(exps_df, on=["name","short"])

    gene_fit_d['specphe'] = SpecificPhenotypes(gene_fit_d['g'], 
                            tmp_df, gene_fit_d['lrn'][used_experiment_names], 
                            gene_fit_d['t'][used_experiment_names], 
                            spec_cfg = cfg["spec_cfg"],
                            dbg_prnt=True)

    return gene_fit_d



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


def cor12(pairs, usedLocusIds, fitnorm_df, method="pearson"):
    """
    Args:
        pairs (pandas DataFrame): comes from AdjacentPairs: has following column names
            Gene1, Gene2, sysName1, type1, scaffoldId, begin1, end1, strand1, name1, desc1, GC1, 
            nTA1, locusId, sysName2, type2, begin2, end2, strand2, name2, desc2, GC2, nTA2
        usedLocusIds (pandas Series<locusId (str)>) : gene_fit_d['g']
        fitnorm_df (pandas DataFrame all floats): dataframe with one column per setindexname ( Fitness normalized)
        method: "pearson" (or "spearman" but not really)

    Description:
        The input 'pairs' dataframe has locusIds in the columns 'Gene1' and 'Gene2'. We want to find
        those locusId index within gene_fit_d['g'] which is called 'usedLocusIds'.
        We use 'py_match' to find their indices within usedLocusIds and label Gene1's locations as
        i1, and Gene2's locations as i2. Note that since many of the genes in the overall
        genes.GC file (genes_df) are not used, and since the genes in pairs come from that
        file, then many of these genes won't be in 'usedLocusIds', and thus we will have
        many locations in i1 and i2 with NaN values.
        Now we go through all the rows in the pairs dataframe, and for each time we
        have both of the locusIds (for columns 'Gene1' and 'Gene2') included in
        usedLocusIds, then we compute their correlation using the method given
        (default "pearson" correlation).
    """
   
    i1 = py_match(list(pairs["Gene1"]), list(usedLocusIds))
    i2 = py_match(list(pairs["Gene2"]), list(usedLocusIds))
    res = []
    for ix in range(pairs.shape[0]):
        if pd.isnull(i1[ix]) or pd.isnull(i2[ix]):
            res.append(np.nan)
        else:
            res.append(fitnorm_df.iloc[i1[ix]].corr(fitnorm_df.iloc[i2[ix]], method=method))
    
    return res


def TopCofit(locusIds, lrn, pre_n=None, dbg=False, fraction=0.02):
    """
    Args:
        locusIds (pandas Series): is names of genes
        lrn (pd DataFrame): is a matrix of fitness values with columns set name index 
        pre_n (int or None): If int, then we get the top n=pre_n cofitness values for each gene.
                            If None, then we compute the number 'n' in another way, using 'fraction'

    Returns:
        out_df (pandas DataFrame): has columns:
            locusId (str), 
            hitId (str) 
            cofit (float)
            rank (int)

    Description:
        Our overall goal is to get the top 'n' cofitness
        for each gene and put it in a dataframe. 
        'n' can be decided in two ways, first it can be given
        to the function with the argument 'pre_n'.
        Otherwise it will be computed by taking the minimum
        between 0.02 fraction of nGenesUsed and nGenesUsed - 1,
        which will probably be 0.02 fraction of nGenesUsed.
        If 0.02*nGenesUsed is less than 1, then the value
        will be 1.
        Now, we want to compute the correlation between all genes.
        To do this, we transpose the matrix from having columns
        be experiment names to having columns be the index of
        the genes. Then we use a built-in pandas function corr()
        and get a matrix of Gene to Gene, where the diagonal
        will always be 1 (since correlation with yourself is
        1).
        Now we want the top 'n' cofitness values for each gene
        (we have to make sure to remove the genes correlation 
        with itself). How we get the top n this is for each row, 
        we sort it in descending order,
        then we find the location of the top (1,n+1) sorted values
        within the original row and store those locations.
        Then we get the locusIds at those locations, and 
        create a DataFrame with the following columns:
            "locusId": (str) Name of gene
            "hitId": (str) Name of other gene with which cor computed
            "cofit": (float) Value of cofitness
            "rank": (int) Numerical ranking of cofitness related to others
        The number of rows in the dataframe is n*'nGenesUsed', since
        for each gene in genesUsed, we have n rows (the top n
        cofitness values).
        For example, if n = 3, and the locusIds are "a","b","c","d","e",
        you might have:
            locusId  hitId  cofit  rank
            a        b        0.8   1
            a        c        0.7   2
            a        d        0.6   3
            b        d        0.9   1
            b        a        0.8   2   
            .
    """

    logging.info("Computing top cofit")

    if pre_n is None: 
        n = min( max(1, round(len(locusIds) * fraction)) , len(locusIds) - 1)
    else:
        if pre_n > len(locusIds):
            logging.warning(f"Top cofitness value {pre_n} is greater than total number"
                            f" of genes {len(locusIds)}")
            n = len(locusIds) - 1
        else:
            n = pre_n

    print(f"TopCofit number: n: {n}")

    # Number of locusIds must match number of rows in lrn
    if len(locusIds) != lrn.shape[0]:
        raise Exception("Number of genes and number of rows in matrix do not match.")
    
    # We transpose the matrix lrn
    trp = lrn.transpose()
    # We want each row correlated with itself, so the diagonal in the middle is 1
    # cofits is an GxG matrix (dataframe) where G=len(locusIds)=lrn.shape[0]
    cofits = trp.corr()

    nOut = len(locusIds)*n
    if dbg:
        print(f"Making output with {nOut} rows")

    '''
    out_hitId = pd.Series([""]*nOut)
    out_cofit = pd.Series([np.nan]*nOut)
    '''
    out_hitId = [] 
    out_cofit = [] 

    for i in range(len(locusIds)):
        values = cofits.iloc[i,:]
        #print(values)
        sorted_values = values.sort_values(ascending=False)
        #print(sorted_values)
        top_n_indexes = sorted_values.index[1:n+1]
        #print(top_n_indexes)
        top_n_locusIds = locusIds.iloc[top_n_indexes]
        #print(top_n_locusIds)
        top_n_values = sorted_values.values[1:n+1]
        #print(top_n_values)
        out_hitId += list(top_n_locusIds)
        out_cofit += list(top_n_values)

    # We get the other values in this dataframe
    lI_list = []
    rank = []
    for i in range(len(locusIds)):
        lI_list += [locusIds.iat[i]]*n
        rank += list(range(1,n+1))
  


    out_df = pd.DataFrame.from_dict({
        "locusId": lI_list,
        "hitId": out_hitId,
        "cofit": out_cofit,
        "rank": rank
        })

    return out_df



def HighFit(gene_fit_d, genes_df, exps_df, all_df,
            high_cfg=None,
            dbg_prnt=False):
    """
    Args:
       gene_fit_d (python dict):
            lrn (fitness): pandas DataFrame (one col per setindexname) floats (fitness?)
            t (t-score): pandas DataFrame (one col per setindexname) floats (t_score?)
            u (used?): pandasDataFrame (one col per setindexname) floats
            tot: Total number of reads in this locusId
            n: Total number of different strains inserted into this locusId

    Description:
        We find the [row, col] indexes where the 'lrn' and 't' dataframes (fitness and 
        t score dataframes) have values that pass the thresholds of minimum fitness and 
        minimum t score (parameters min_fit and min_t). We create a new dataframe called
        'high_df' which contains the locusId, experiment name, fitness score and t scores
        where these thresholds are passed. The number of rows in these dataframes is equal
        to the number of locations where the thresholds are passed.
        We also compute some other values for each of these high fitness strains,
        those values are standard error and naive standard deviation. We remove
        all high fitness locations where the standard error passes our threshold
        for 'max_se'. We then merge our new dataframe with the quality
        dataframe which gives us values 'maxFit' and 'gMean' per experiment, we use these
        values to further dwindle down our HighFitness values to match
        the more stringent rules:
            the gMean has to be greater than the minimum gMean
            the fitness value has to be greater than [maxFit - maxFitBelow]
        We combine our new dataframe with some columns from the genes dataframe
        we'd like to include ('sysName' and 'desc').
        We create a column called 'nDetected', which takes the total number 
        of times a locusId was inserted into per experiment IF that locusId
        is included in the current HighFitness DataFrame.
        
        and return that dataframe, 
        which now contains
        as many as 19 columns (column number depends on column names
        in exps_df, min number is 15, assuming the column names 
        "Condition_1", "Concentration_1", "Units_1", "Media" are optional.) 


    Returns:
        new_high (pandas DataFrame):
            locusId, expName, fit, t, nReads, nStrains,
            se, sdNaive, name, Group, Condition_1, Concentration_1, 
            Units_1, Media, short, u, maxFit, gMean, sysName, desc,
            nDetected
            
    Subroutines:
        py_order: (from translate_R_to_pandas)
    """

    if high_cfg is not None:
        min_fit = high_cfg["min_fit"]
        min_t = high_cfg["min_t"]
        max_se = high_cfg["max_se"]
        min_reads = high_cfg["min_reads"]
        min_gMean = high_cfg["min_gMean"]
        max_below = high_cfg["max_below"]
        min_strains = high_cfg["min_strains"]
        min_strain_fraction = high_cfg["min_strain_fraction"]
    else:
        min_fit=4 
        min_t=5 
        max_se=2 
        min_reads=10 
        min_gMean=10 
        max_below=8 
        min_strains=2 
        min_strain_fraction=0.5




    logging.info("Starting to compute High Fitness DataFrame")
    lrn = gene_fit_d['lrn']
    t = gene_fit_d['t']
    u = gene_fit_d['q']['u']
    tot = gene_fit_d['tot']
    n_reads = gene_fit_d['n']
    
    # This needs to be two columns: 1 with rows and 1 with columns
    num_rows, num_cols = lrn.shape[0], lrn.shape[1]
    # where is high is a list of [row (int), col(int)] (coming from dataframe, so it's a list whose length
    # is the length of (m x j) for rows and columns in the dataframe.
    where_is_high = []
    for i in range(num_rows):
        for j in range(num_cols):
            if lrn.iloc[i,j] >= min_fit and t.iloc[i,j] >= min_t \
            and tot.iloc[i,j] >= min_reads and n_reads.iloc[i,j] >= min_strains:
                where_is_high.append([i,j])



    logging.info(f"Number of high instances: {len(where_is_high)}")

    if len(where_is_high) == 0:
        new_high = pd.DataFrame.from_dict({
                    "locusId": []
                    })
        return new_high 

    fit_high = []
    t_high = [] 
    experiment_names = []
    locusIds = []
    nReads = []
    nStrains = []
    for x in where_is_high:
            fit_high.append(lrn.iloc[x[0], x[1]])
            t_high.append(t.iloc[x[0], x[1]])
            experiment_names.append(lrn.columns[x[1]])
            locusIds.append(gene_fit_d['g'][x[0]])
            nReads.append(tot.iloc[x[0],x[1]])
            nStrains.append(n_reads.iloc[x[0],x[1]])


    high_df = pd.DataFrame.from_dict({
                    # x[0] -> rows from where_is_high
                    "locusId": locusIds,
                    # x[1] -> columns from where_is_high
                    "expName": experiment_names,
                    "fit": fit_high,
                    "t": t_high,
                    "nReads": nReads,
                    "nStrains": nStrains
                })

    high_df['se'] = high_df['fit']/high_df['t']
    high_df['sdNaive'] = [gene_fit_d['sdNaive'].iloc[x[0], x[1]] for x in where_is_high]
    high_df = high_df[high_df['se'] <= max_se]

    # Which experiments are ok
    fields = "name Group Condition_1 Concentration_1 Units_1 Media short".split(" ")
    fields = [x for x in fields if x in exps_df.columns]
    crnt_exps = exps_df[fields]
    crnt_exps = crnt_exps.merge(gene_fit_d['q'][["name","u","short","maxFit","gMean"]], on="name")
    new_high = high_df.merge(crnt_exps, left_on="expName", right_on="name")
    check_bool = [bool(val >= min_gMean and \
                  new_high['fit'].iloc[ix] >= new_high['maxFit'].iloc[ix] - max_below) \
                  for ix, val in new_high['gMean'].items()]
    
    new_high = new_high[check_bool]
    # Adding the two dataframes
    new_high = new_high.merge(genes_df[["locusId","sysName","desc"]], on="locusId")



    # Compute #strains detected per gene x sample
    newhighlocIds = list(new_high['locusId'])
    strainsUsed = gene_fit_d['strainsUsed']
    u = [bool(all_locId in newhighlocIds and strainsUsed[ix]) \
         for ix, all_locId in all_df['locusId'].items()]
    if u.count(True) == 0:
        logging.info("No strains in 'nDetected'")
    d = all_df[u]
    d = d[[x for x in all_df.columns if x in list(new_high['expName'])]]
    nDetected = (d>0).groupby(by=all_df[u]["locusId"]).sum()
    

    # Reformat table to have only 3 columns, and values lined up that way,.
    locusIds = []
    expNames = []
    values = []
    for row_ix in range(nDetected.shape[0]):
        for col_ix in range(nDetected.shape[1]):
            locusIds.append(nDetected.index[row_ix])
            expNames.append(nDetected.columns[col_ix])
            values.append(nDetected.iloc[row_ix, col_ix])

    shifted_nDetected = pd.DataFrame.from_dict({
                        "locusId": locusIds,
                        "expName": expNames,
                        "nDetected": values
                        })

    new_high = new_high.merge(shifted_nDetected, on=["locusId", "expName"])
    new_high = new_high[new_high["nDetected"]/new_high["nStrains"] > min_strain_fraction]
    new_high = new_high.sort_values(by=['expName', 'fit'], ascending=[True, False])
    if dbg_prnt:
        new_high.to_csv("tmp/py_new_high_df.tsv", sep="\t", index=False)

    logging.info("Finished computing High Fitness DataFrame.")

    return new_high



def SpecificPhenotypes(locusIds, q_sub_exps_df, used_lrn, used_t,
                        spec_cfg=None,
                        dbg_prnt=False):
    """
    Args:
        locusIds (pd.Series<str>): locusIds associated with each row in lrn and t
        q_sub_exps_df (pd.DataFrame): Quality DataFrame merged with experiments DataFrame
        used_lrn (pd.DataFrame): subset of lrn with used experiments
        used_t (pd.DataFrame): subset of t with used experiments
    Returns:
        specific_df (pd.DataFrame):
            Quality DataFrame columns adding columns:    
                        "locusId","lrn","t". 
                        Total num columns should be 23
            Num rows are situations where experiments produced specific results
                    
    Description:
        We want to identify "specific phenotypes", which are cases where a gene is sick.
        We have thresholds which the absolute value of the normalized log ratio (fitness)
        of a gene has to pass, and the absolute value of the t-score also has to pass.
        We also get a list which contains the 95th percentile for each row, we call this
        rowHi (percentile is defined by the input to the function, the default is 95th).
        So we do a test for each location in the fitness (normalized log ratios) dataframe,
        which has a corresponding t-score value, and we check this in relation to the
        95th percentile of each row, to get <row_index, column_index> for the dataframes
        in which their is a "specific phenotype". We store these locations (row, col)
        in the list "which_pass_list", and then use that to get the locusIds, experiment
        name, normalized log ratio, and t score, and export a dataframe which has these
        values, along with the experiment information like 'Group', 'Condition_1', etc
        The output dataframe should be relatively small in terms of row number.
        Its row number will be the number of specific phenotypes found.

    """

    if spec_cfg is not None:
        minT = spec_cfg["minT"]
        minFit = spec_cfg["minFit"]
        percentile = spec_cfg["percentile"]
        percentileFit = spec_cfg["percentileFit"]
        minDelta = spec_cfg["minDelta"]
    else:
        minT=5 
        minFit=1.0 
        percentile=0.95
        percentileFit=1.0 
        minDelta=0.5

    expsFields = set(q_sub_exps_df.columns).intersection(set(["name", "short", "Group", "Condition_1",
                                                        "Concentration_1", "Units_1", "Condition_2",
                                                        "Concentration_2", "Units_2", "Condition_3",
                                                        "Concentration_3", "Units_3", "Condition_4",
                                                        "Concentration_4", "Units_4"]))

    # getting the 95th percent quantile for each row of the absolute values of the dataframe
    # rowHi should be a pandas Series with length nGenesUsed
    abs_used_lrn = used_lrn.abs()
    abs_used_t = used_t.abs()
    rowHi = abs_used_lrn.quantile(q=percentile, axis=1)

    '''
    which_pass_df = (abs_used_lrn > minFit) & (abs_used_t > minT)

    for row_ix in range(used_lrn.shape[0]):
        for col_ix in range(used_lrn.shape[1]):
            # If this value is True, we might change it to False
            if which_pass_df.iloc[row_ix, col_ix]:
                rowHival = rowHi.iat[row_ix]
                if (rowHival >= percentileFit) or \
                    (abs_used_lrn.iloc[row_ix, col_ix] <= rowHival + minDelta):
                   which_pass_df.iloc[row_ix, col_ix] = False 
    '''


    # We find <row, col> locations that pass thresholds
    which_pass_list = []
    for row_ix in range(used_lrn.shape[0]):
        for col_ix in range(used_lrn.shape[1]):
            if (abs_used_lrn.iloc[row_ix, col_ix] > minFit and \
                abs_used_lrn.iloc[row_ix, col_ix] > rowHi.iat[row_ix] + minDelta and 
                rowHi.iat[row_ix] < percentileFit and \
                abs_used_t.iloc[row_ix, col_ix] > minT):
                which_pass_list.append([row_ix, col_ix])

    logging.info(f"Found {len(which_pass_list)} specific phenotypes.")

    if len(which_pass_list) == 0:
        specific_df = pd.DataFrame({
                        "locusId": []
                        })
        return specific_df

    # sp - specific
    sp_locId = locusIds.iloc[[x[0] for x in which_pass_list]]
    related_names = [abs_used_lrn.columns[x[1]] for x in which_pass_list]
    log_ratio_vals = [used_lrn.iloc[x[0], x[1]] for x in which_pass_list]
    t_vals = [used_t.iloc[x[0], x[1]] for x in which_pass_list]

    specific_df = pd.DataFrame.from_dict({
                        "locusId": sp_locId,
                        "name": related_names,
                        "lrn": log_ratio_vals,
                        "t": t_vals
                    })


    op_df = specific_df.merge(q_sub_exps_df[expsFields], on="name")

    return op_df


def stop(line_num):
    raise Exception(f"Stopped, line {line_num}") 
