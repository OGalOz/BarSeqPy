

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
In this file all the dataframes and other data created by FEBA_main/FEBA_Fit is exported
    to a single directory, denoted by the variable name 'op_dir' in the function 
    'FEBA_Save_Tables'. In order to test the function, you need a directory with 
    the right inputs to import gene_fit_d using the function 

Input to FEBA_Save_Tables 'gene_fit_d' is big:
        gene_fit_d (python dict): Contains keys:
            g (pandas Series (str)): pandas Series of locusIds
            lr (pandas DataFrame of float): dataframe with one column per setindexname
            lrNaive (pandas DataFrame of float): dataframe with one column per setindexname
            lr1 (pandas DataFrame of float): dataframe with one column per setindexname
            lr2 (pandas DataFrame of float): dataframe with one column per setindexname
            lrn (pandas DataFrame of float): dataframe with one column per setindexname
            lrn1 (pandas DataFrame of float): dataframe with one column per setindexname
            lrn2 (pandas DataFrame of float): dataframe with one column per setindexname
            fitRaw (pandas DataFrame of float): dataframe with one column per setindexname
            n (pandas DataFrame of int): dataframe with one column per setindexname
            nEff (pandas DataFrame of float): dataframe with one column per setindexname
            pseudovar (pandas DataFrame of float): dataframe with one column per setindexname
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
            sumsq (pandas DataFrame of float): dataframe with one column per setindexname
            sd (pandas DataFrame of float): dataframe with one column per setindexname
            sdNaive (pandas DataFrame of float): dataframe with one column per setindexname
            se (pandas DataFrame of float) Standard Error dataframe with one column per setindexname
            t: (pandas DataFrame of float) t-statistic dataframe with one column per setindexname
            tot1 (pandas DataFrame of int or nan) dataframe with one column per setindexname
            tot0_1 (pandas DataFrame of int or nan) dataframe with one column per setindexname
            tot2 (pandas DataFrame of int or nan) dataframe with one column per setindexname
            tot0_2 (pandas DataFrame of int or nan) dataframe with one column per setindexname
            tot (pandas DataFrame of int or nan) dataframe with one column per setindexname
            tot0 (pandas DataFrame of int or nan) dataframe with one column per setindexname
            version (str)
            genesUsed : 
            strainsUsed : 
            genesUsed12 : 
            gN : 
            t0_gN : 
            strains : 
                used,
                enoughT0
                scaffold
                    & multiple others (all_df meta-columns, i.e. columns that describe metadata
                                        as opposed to the set index names.)
            strain_lr : 
            strain_se : 
            high (pandas DataFrame): dbg@(tmp/py_new_high_df.tsv)
                locusId, expName, fit, t, se, sdNaive, name, Group, Condition_1, Concentration_1, 
                Units_1, Media, short, u, maxFit, gMean, sysName, desc

            Optional Keys (depending on inputs)
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

def FEBA_Save_Tables(gene_fit_d, genes_df, organism_name_str,
                     op_dir, exps_df, writeImage=False, debug=False):
    """
    Args:
        gene_fit_d (python dict): Documentation above function
        genes_df (pandas DataFrame): table genes.GC
        organism_name_str (str): Name of organism
        op_dir (str): Directory to write all saved tables and JSON to.
        exps_df (pandas DataFrame): from FEBA.BarSeq
            Must contain cols:
                name
                short
                
        writeImage (bool): Should we save all the data in one image to 
                            be easily imported into python/R?

    Note: 
        We merge many dataframes on the locusId columns



    Outputs the following dataframes:

    """

    # Setting print options for debugging:
    pd.set_option('display.max_columns', None)

    if not os.path.isdir(op_dir):
        os.mkdir(op_dir)

    for expected_key in ["q","lr","lrn","lrn1","lrn2","t", "genesUsed","g", "lrNaive"]:
        if expected_key not in gene_fit_d:
            raise Exception(f"Missing expected key in gene_fit_d: {expected_key}")

    for name in gene_fit_d['q']['name']:
        if name not in gene_fit_d['lr'].columns:
            raise Exception(f"Name {name} missing from 'lr' object.")
        if name not in gene_fit_d['lrn'].columns:
            raise Exception(f"Name {name} missing from 'lrn' object.")

    for val in ["locusId", "sysName", "desc"]:
        if val not in genes_df.columns:
            raise Exception(f"Column name {val} not in genes_df")

    # Preparing variables that make it simpler to create_tables
    first3_cols = ["locusId", "sysName", "desc"]
    genes_first3 = genes_df[first3_cols]
    final_colnames = list(gene_fit_d['q']['name'] + ' ' + gene_fit_d['q']['short'])

    # WRITING TABLES:
    write_DataFrame_and_log(os.path.join(op_dir, "fit_quality.tsv"), gene_fit_d['q'], df_name="quality")


    #2 Fit genes - All genes, with some having the used column = True
    # used is a boolean list
    used = [(genes_df['locusId'].iat[i] in gene_fit_d['genesUsed']) \
            for i in range(len(genes_df['locusId']))]
    new_genes_df = genes_df.copy(deep=True)
    new_genes_df['used'] = used
    write_DataFrame_and_log(os.path.join(op_dir, "fit_genes.tab"), new_genes_df, df_name = "Fit genes")
    del new_genes_df, used

    #3 Fit Log Ratios unnormalized 
    pre_merge = gene_fit_d['lr']
    pre_merge['locusId'] = gene_fit_d['g']
    # below how is 'inner' by default, which is the fitting merge type
    tmp_df = genes_first3.merge(pre_merge, on="locusId") 
    write_DataFrame_and_log(os.path.join(op_dir, "fit_logratios_unnormalized.tab"), 
                            tmp_df, df_name = "log ratios unnormalized")


    #4 Log Ratios Unnormalized Naive (Can put into 'extract...' function)
    pre_merge = gene_fit_d['lrNaive'].copy(deep=True)
    pre_merge['locusId'] = gene_fit_d['g']
    tmp_df = genes_first3.merge(pre_merge, on="locusId") 
    write_DataFrame_and_log(os.path.join(op_dir, "fit_logratios_unnormalized_naive.tab"), 
                            tmp_df, df_name = "log ratios unnormalized naive")


    #5 Fit Logratios 
    pre_merge = gene_fit_d['lrn'].copy(deep=True)
    pre_merge['locusId'] = gene_fit_d['g']
    tmp_df = genes_first3.merge(pre_merge, on="locusId") 
    write_DataFrame_and_log(os.path.join(op_dir, "fit_logratios.tab"), 
                            tmp_df, df_name = "fit logratios")

    #6 Fit Log Ratios 1st half (Can put into 'extract...' function)
    pre_merge = gene_fit_d['lrn1'].copy(deep=True)
    pre_merge['locusId'] = gene_fit_d['g']
    tmp_df = genes_first3.merge(pre_merge, on="locusId") 
    write_DataFrame_and_log(os.path.join(op_dir, "fit_logratios_half1.tab"), 
                            tmp_df, df_name = "fit logratios 1st half")


    #7 Fit Log Ratios 2nd half (Can put into 'extract...' function)
    pre_merge = gene_fit_d['lrn2'].copy(deep=True)
    pre_merge['locusId'] = gene_fit_d['g']
    tmp_df = genes_first3.merge(pre_merge, on="locusId") 
    write_DataFrame_and_log(os.path.join(op_dir, "fit_logratios_half2.tab"), 
                            tmp_df, df_name = "fit logratios 2nd half")



    print(genes_df)
    #8 Fit Log Ratios Good (?)
    genes_in_g_bool = [bool(genes_df['locusId'].iat[i] in gene_fit_d['g'].values) for i \
                        in range(genes_df.shape[0])]
    f3col_genes_df = genes_df[first3_cols][genes_in_g_bool]
    f3col_genes_df['comb'] = f3col_genes_df['sysName'] + ' ' + f3col_genes_df['desc']
    tmp_df = f3col_genes_df.copy(deep=True)
    # q is quality, u is used
    if list(gene_fit_d['q']['u']).count(True) == 0:
        logging.warning("***Warning: 0 'OK' experiments.")
        tmp_df = tmp_df.sort_values(by='locusId')
    else:
        used_q_rows = gene_fit_d['q'][gene_fit_d['q']['u']] 
        used_names = used_q_rows['name'] 
        lrn_copy = gene_fit_d['lrn'].copy(deep=True)
        lrn_copy = lrn_copy[used_names]
        lrn_copy['locusId'] = gene_fit_d['g']
        tmp_new = tmp_df.merge(lrn_copy, on="locusId")
        rename_columns = list(used_q_rows['name'] + ' ' + used_q_rows['short'])
        rename_d = {val: rename_columns[ix] for ix, val in enumerate(list(tmp_new.columns[4:]))}
        tmp_new = tmp_new.rename(columns=rename_d)

    write_DataFrame_and_log(os.path.join(op_dir, "fit_logratios_good.tab"), 
                            tmp_new, df_name = "fit logratios good")
    del tmp_new, lrn_copy

    
    #9 Gene Counts 
    pre_merge = gene_fit_d['tot'].copy(deep=True) 
    pre_merge['locusId'] = gene_fit_d['g']

    tmp_df = f3col_genes_df.merge(pre_merge, on="locusId") 
    write_DataFrame_and_log(os.path.join(op_dir, "gene_counts.tab"), 
                            tmp_df, df_name = "gene counts")
    

    #10 Fit T Scores
    extract_gene_fit_d_category_to_tsv_basic(gene_fit_d['t'],
                                             gene_fit_d['g'],
                                             genes_first3,
                                             final_colnames,
                                             os.path.join(op_dir, "fit_t.tab"),
                                             "fit t")

    #11 Fit standard error
    extract_gene_fit_d_category_to_tsv_basic(gene_fit_d['se'],
                                             gene_fit_d['g'],
                                             genes_first3,
                                             final_colnames,
                                             os.path.join(op_dir, "fit_standard_error_obs.tab"),
                                             "fit standard error")


    #12 Fit Standard Error Naive
    extract_gene_fit_d_category_to_tsv_basic(gene_fit_d['sdNaive'],
                                             gene_fit_d['g'],
                                             genes_first3,
                                             final_colnames,
                                             os.path.join(op_dir, "fit_standard_error_naive.tab"),
                                             "fit standard error naive")

    #13 Strain Fit
    logging.info("Getting order of scaffolds to print Strain Fit.")
    tmp_df = gene_fit_d['strains'].join(gene_fit_d['strain_lrn'])
    tmp_df.sort_values(by=['scaffold','pos'])
    write_DataFrame_and_log(os.path.join(op_dir,"strain_fit.tab"), 
                            tmp_df, 
                            df_name="Strain Fit")

    #14 expsUsed (subset of original exps file with used experiments
    write_DataFrame_and_log(os.path.join(op_dir,"expsUsed.tab"), 
                            exps_df, 
                            df_name="expsUsed")
    
    #15 Cofit
    if 'cofit' in gene_fit_d and gene_fit_d['cofit'] is not None:
        # Why do we repeat the three columns sysName, locusId and desc
        # with hitSysName, hitId, and hitDesc etc?
        tmp_df = f3col_genes_df.merge(gene_fit_d['cofit'], on="locusId")
        pre_merge_df = pd.DataFrame.from_dict({
            "hitId" : genes_df["locusId"],
            "hitSysName" : genes_df["sysName"],
            "hitDesc" : genes_df["desc"]
        })
        tmp_df = tmp_df.merge(pre_merge_df)
        tmp_df.sort_values(by=["locusId", "rank"], inplace=True, axis=0)
    else:
        logging.warning("Cofit not found in gene_fit_d")
        tmp_df = pd.DataFrame.from_dict({
            "locusId": [""],
            "sysName": [""],
            "desc": [""],
            "cofit": [""],
            "rank":[""],
            "hitId": [""],
            "hitSysName": [""],
            "hitDesc": [""]
            })
    write_DataFrame_and_log(os.path.join(op_dir, "cofit.tab"), 
                            tmp_df, 
                            df_name="cofit")




    #16 specphe - specific phenotypes
    if "specphe" in gene_fit_d and gene_fit_d["specphe"] is not None:
        #print(f3col_genes_df)
        #print(f3col_genes_df.dtypes)
        #print(gene_fit_d['specphe'])
        #print(gene_fit_d['specphe'].dtypes)
        tmp_df = f3col_genes_df.merge(gene_fit_d['specphe'], on="locusId")
    else:
        tmp_df = pd.DataFrame.from_dict({
                "locusId": [""],
                "sysName": [""],
                "desc": [""],
                "short": [""],
                "Group": [""],
                "Condition_1": [""],
                "Concentraion_1": [""],
                "Units_1": [""],
                "Condition_2": [""],
                "Concentration_2": [""],
                "Units_2": [""],
            })
    print(tmp_df.head(6))
    write_DataFrame_and_log(os.path.join(op_dir, "specific_phenotypes.tab"), 
                            tmp_df, 
                            df_name="specific phenotypes")



    # 17 Strong - 
    # We create the dataframe 'strong.tab'
    # we find which normalized log ratios are greater than 2 and 
    #             't' scores are greater than 5. We store results in one list
    #             'which_are_strong' which is list<[col_name (str), row_ix (int)]>
    create_strong_tab(gene_fit_d, genes_df, exps_df, op_dir, debug=debug)


    #18 High
    # High Fitness
    write_DataFrame_and_log(os.path.join(op_dir, "high_fitness.tab"), 
                            gene_fit_d['high'], 
                            df_name="high fitness")

    #19 HTML Info
    html_info_d = {
            "organism_name": organism_name_str,
            "number_of_experiments": len(gene_fit_d['q']['short']) - \
                                     list(gene_fit_d['q']['short']).count("Time0"),
            "number_of_successes": list(gene_fit_d['q']['u']).count(True),
            "version": gene_fit_d['version'],
            "date": str(datetime.now())
    }

    with open(os.path.join(op_dir, "html_info.json"), 'w') as g:
        g.write(json.dumps(html_info_d, indent=2))

    
    logging.info("Finished exporting all tables and files to " + op_dir)

    return 0


def extract_gene_fit_d_category_to_tsv_basic(input_df,
                                             genes_locusId,
                                             genes_first3,
                                             final_colnames,
                                             output_filepath,
                                             df_log_name):
    """
    Args:
        inp_df: A standard DataFrame coming out of gene_fit_d, length is
                nGenesUsed
        genes_locusId: pandas Series with locusId (str)
        genes_first3 DataFrame: Columns of genes.GC ["locusId", "sysName", "desc"]
        final_colnames: list(gene_fit_d['q']['name'] + ' ' + gene_fit_d['q']['short'])
        output_filepath: Path to write out the TSV
        df_log_name: What name of output to report

    Subroutines: 
    """

    pre_merge = input_df.copy(deep=True)
    pre_merge['locusId'] = genes_locusId
    tmp_df = genes_first3.merge(pre_merge, on="locusId") 
    write_DataFrame_and_log(output_filepath, 
                            tmp_df, 
                            df_name=df_log_name)
    


    return None


def create_strong_tab(gene_fit_d, genes_df, exps_df, op_dir, debug=False):
    """
    Description:
        We create the dataframe 'strong.tab'
    # we find which normalized log ratios are greater than 2 and 
    #             't' scores are greater than 5. We store results in one list
    #             'which_are_strong' which is list<[col_name (str), row_ix (int)]>
    We end up with a dataframe with a single row for every strong fit and t score
    value, and we add other informational columns like 'sysName', 'desc' (description)
    and 'short'. After all we have a dataframe with the following columns:
        locusId, name (experiment name), t (t score), lrn (fitness score
        (log ratio normalized)), sysName, desc, and short.

    
    """
    which_are_strong = []
    abs_lrn = gene_fit_d['lrn'].abs()
    abs_t = gene_fit_d['t'].abs()
    for col in gene_fit_d['lrn'].columns:
        for row_ix in range(gene_fit_d['lrn'].shape[0]):
            if abs_lrn[col].iloc[row_ix] > 2 and \
                abs_t[col].iloc[row_ix] > 5:
                which_are_strong.append([col, row_ix])
    strong_t = [gene_fit_d['t'][x[0]].iloc[x[1]] for x in which_are_strong]
    strong_lrn = [gene_fit_d['lrn'][x[0]].iloc[x[1]] for x in which_are_strong]
    locusIds = gene_fit_d['g']
    stronglocusIds = locusIds.iloc[[x[1] for x in which_are_strong]]
    name_col = [x[0] for x in which_are_strong]


    if len(which_are_strong) > 0:
        strong_df = pd.DataFrame.from_dict({
                    "locusId": stronglocusIds,
                    "name": name_col,
                    "t": strong_t,
                    "lrn": strong_lrn
                    })
        print(strong_df)
        sysName = []
        desc = []
        pre_merge1 = genes_df[["locusId", "sysName", "desc"]]
        strong_df = strong_df.merge(pre_merge1, on="locusId")
        pre_merge2 =  exps_df[["name", "short"]]
        strong_df = strong_df.merge(pre_merge2, on="name")
        """
        for locusId in strong_df["locusId"]:
            if debug:
                print(f"current locusId: {locusId}")
                print(f"type of locusId: {type(locusId)}")
            sysName.append(genes_df[genes_df["locusId"] == locusId]["sysName"].values[0])
            desc.append(genes_df[genes_df["locusId"] == str(locusId)]["desc"].values[0])

        strong_df["sysName"] = sysName
        strong_df["desc"] = desc

        short = []
        for exp_name in strong_df["name"]:
            short.append(exps_df[exps_df["name"] == exp_name]["short"].values[0])

        strong_df["short"] = short
        """
    else:
        strong_df = pd.DataFrame.from_dict({
                    "locusId": [],
                    "name": [],
                    "t": [],
                    "lrn": [],
                    "sysName": [],
                    "desc": [],
                    "short": [] 
        })
    write_DataFrame_and_log(os.path.join(op_dir, "strong.tab"), 
                            strong_df, 
                            df_name="strong")



def stop(line_num):
    raise Exception(f"Stopped, line {line_num}") 

def write_DataFrame_and_log(op_fp, df, df_name=None):
    """
    Args: 
        op_fp (str): Output file path to write dataframe to
        df (pandas DataFrame)
        df_name (str or None): If not None, report name of dataframe written
    """
    df.to_csv(op_fp, sep="\t", index=False)
    if df_name is not None:
        logging.info(f"Wrote DataFrame {df_name} to {op_fp}")
    else:
        logging.info(f"Wrote DataFrame {op_fp}")



def test_import_gene_fit_d(inp_dir):
    """
    inp_dir (str): Path to directory containing the following files:
        
    """

    return None


def main():
    args = sys.argv

    if args[-1] != "1":
        print("Incorrect args.")
        print("Should be:\n"
              "python3 FEBA_Save_Tables.py gene_fit_d_dir genes_fp exps_df_fp"
              "organism_name_str op_dir 1")
        sys.exit(1)
    else:
        fn_ph, gfd_dir, genes_fp, exps_df_fp, org_str, op_dir, num_ph = args
        gene_fit_d = test_import_gene_fit_d(gfd_dir)
        genes_df = pd.read_table(genes_fp)
        exps_df = pd.read_table(exps_df_fp)
        FEBA_Save_Tables(gene_fit_d, genes_df, org_str,
                     op_dir, exps_df, writeImage=False)
        sys.exit(0)


if __name__ == "__main__":
    main()
    

def combine_dataframes_on_column(df1, df2, combine_col_name, optional_str=None,
                                 debug=False):
    """
    Args:
        df1 (pandas DataFrame): must include column {combine_col_name} as 
                                well as at least one other column. But for
                                all other columns, they should not be the 
                                same as the columns of df2.
        df2 (pandas DataFrame): must include column {combine_col_name} as 
                                well as at least one other column. But for
                                all other columns, they should not be the 
                                same as the columns of df1.
        combine_col_name (str): Name of column to merge on

    Description:
        For each value in combine_col_name of the first dataframe, we
        find the row number that same value exists in for the second,
        and combine all the other values on that row for both dataframes
        and add that to a list which we eventually turn into a dataframe
        with all the rows from both that overlap.

        How this is generally used within the program is that it takes
        the 3 columns of the genes dataframe "locusId", "sysName", "desc",
        and takes a dataframe with values associated with a "locusId" column
        as well, and combines the two where the "locusId" value is the same.
        So we can look at the gene's name, it's sysName and it's basic description
        along with numerical values associated with it in the same row.
    """

    logging.info(f"Merging two dataframes on column: {combine_col_name}.")
    if debug:
        print("First dataframe dtypes:")
        print(df1.dtypes)
        print("Second dataframe dtypes:")
        print(df2.dtypes)
        print("First dataframe head:")
        print(df1.head)
        print("Second dataframe head:")
        print(df2.head)

    if optional_str is not None:
        logging.info(f"Merging to create dataframe {optional_str}")

    if (combine_col_name not in df1) or (combine_col_name not in df2):
        raise Exception(f" Field to combine on {combine_col_name} not in one of the dataframes.")


    df1_combine_srs = df1[combine_col_name]
    df2_combine_srs = df2[combine_col_name]
    # These are lists
    df1_other_cols = list(df1.columns)
    df1_other_cols.remove(combine_col_name)
    df2_other_cols = list(df2.columns)
    df2_other_cols.remove(combine_col_name)

    combined_rows = []
    for tup in df1_combine_srs.iteritems():
        if tup[1] in df2_combine_srs.values:
            df1_row_num = tup[0]
            df2_row_num = list(df2_combine_srs).index(tup[1])
            row_d = {combine_col_name: tup[1]}
            for col_name in df1_other_cols:
                row_d[col_name] = df1[col_name].iloc[df1_row_num]
            for col_name in df2_other_cols:
                row_d[col_name] = df2[col_name].iloc[df2_row_num]
            combined_rows.append(row_d)

    op_df_dict = {}
    if len(combined_rows) > 0:
        all_cols = combined_rows[0].keys()
        for col_name in all_cols:
            op_df_dict[col_name] = []
        for row_d in combined_rows:
            for col_name in all_cols:
                op_df_dict[col_name].append(row_d[col_name])

    op_df = pd.DataFrame.from_dict(op_df_dict)

    return op_df


def create_R_PDFs():
    """
    In this function we create all the PDFs that are created by R
    using the original R code.
    """



