

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
            tot1_0 (pandas DataFrame of int or nan) dataframe with one column per setindexname
            tot2 (pandas DataFrame of int or nan) dataframe with one column per setindexname
            tot2_0 (pandas DataFrame of int or nan) dataframe with one column per setindexname
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
        We want to merge many dataframes on the locusId columns
    """

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


    first3_cols = ["locusId", "sysName", "desc"]
    genes_first3 = genes_df[first3_cols]
    print(genes_first3.head())
    final_colnames = list(gene_fit_d['q']['name'] + ' ' + gene_fit_d['q']['short'])


    # WRITING TABLES:
    write_DataFrame_and_log(os.path.join(op_dir, "fit_quality.tsv"), gene_fit_d['q'], df_name="quality")


    #2 Fit genes - should be good
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
    """
    combine_dataframes_on_column(genes_first3, 
                                pre_merge, 
                                'locusId', 
                                optional_str="log ratios unnormalized",
                                debug=False)
    #tmp_df = genes_df[first3_cols].append(pre_merge)
    tmp_df.reindex(columns = first3_cols + final_colnames)
    """
    write_DataFrame_and_log(os.path.join(op_dir, "fit_logratios_unnormalized.tab"), 
                            tmp_df, df_name = "log ratios unnormalized")


    #4 Log Ratios Unnormalized Naive (Can put into 'extract...' function)
    pre_merge = gene_fit_d['lrNaive'].copy(deep=True)
    pre_merge['locusId'] = gene_fit_d['g']
    tmp_df = genes_first3.merge(pre_merge, on="locusId") 
    write_DataFrame_and_log(os.path.join(op_dir, "fit_logratios_unnormalized_naive.tab"), 
                            tmp_df, df_name = "log ratios unnormalized naive")


    #5 Fit Logratios (Can put into 'extract...' function)
    pre_merge = gene_fit_d['lrn'].copy(deep=True)
    pre_merge['locusId'] = gene_fit_d['g']
    tmp_df = genes_first3.merge(pre_merge, on="locusId") 
    write_DataFrame_and_log(os.path.join(op_dir, "fit_logratios.tab"), 
                            tmp_df, df_name = "fit logratios")

    #6 Fit Log Ratios 1st half (Can put into 'extract...' function)
    pre_merge = gene_fit_d['lrn'].copy(deep=True)
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




    #8 Fit Log Ratios Good (?)
    tmp_df = genes_df[first3_cols]
    genes_in_g_bool = [bool(genes_df['locusId'].iat[i] in gene_fit_d['g']) for i \
                        in range(len(genes_df['locusId']))]
    tmp_df = tmp_df[genes_in_g_bool]
    tmp_df['comb'] = pd.Series([tmp_df['sysName'].iat[i] + ' ' + \
                                tmp_df['desc'].iat[i] for i in range(len(tmp_df['sysName']))])
    # q is quality, u is used
    if list(gene_fit_d['q']['u']).count(True) == 0:
        logging.warning("***Warning: 0 'OK' experiments.")
        tmp_df = tmp_df[py_order(list(tmp_df['locusId']))]
    else:
        u_l = list(gene_fit_d['q']['u'])
        cols_to_keep = [i for i in range(len(u_l)) if u_l[i]]
        if isinstance(gene_fit_d['g'], pd.DataFrame):
            pre_merge_df =gene_fit_d['g'].copy(deep=True)
        elif isinstance(gene_fit_d['g'], pd.Series):
            pre_merge_df = pd.DataFrame.from_dict({
                "locusId": gene_fit_d['g']
                })
        else:
            raise Exception("Could not recognize type of gene_fit_d['g']:"
                            f" {type(gene_fit_d['g'])}")
        pre_merge_df.append(gene_fit_d['lrn'].iloc[:,u_l])
        tmp_df = tmp_df.append(pre_merge_df)

    write_DataFrame_and_log(os.path.join(op_dir, "fit_logratios_good.tab"), 
                            tmp_df, df_name = "fit logratios good")
    

    
    #9 Gene Counts (?)
    tmp_df = genes_df[first3_cols]
    genes_in_g_bool = [bool(genes_df['locusId'].iat[i] in gene_fit_d['g']) for i \
                        in range(len(genes_df['locusId']))]
    tmp_df = tmp_df[genes_in_g_bool]
    tmp_df['comb'] = pd.Series([tmp_df['sysName'].iat[i] + ' ' + \
                                tmp_df['desc'].iat[i] for i in range(len(tmp_df['sysName']))])

    pre_merge = gene_fit_d['tot'].copy(deep=True) 
    pre_merge['locusId'] = gene_fit_d['g']

    tmp_df = genes_first3.merge(pre_merge, on="locusId") 
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
    # py_order from translate_R_to_pandas.py
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
        tmp_df = genes_df[first3_cols].merge(gene_fit_d['cofit'], on="locusId")
        tmp_df["hitSysName"] = genes_df["sysName"]
        tmp_df["hitId"] = genes_df["locusId"]
        tmp_df["hitDesc"] = genes_df["desc"]
        tmp_df.sort_values(by=["locusId", "rank"], inplace=True, axis=0)
    else:
        logging.warning("Cofit not found in gene_fit_d")
        tmp_df = pd.DataFrame.from_dict({
            "locusId": [""],
            "sysName": [""],
            "desc": [""],
            "hitId": [""],
            "cofit": [0],
            "rank":[0],
            "hitSysName": [""],
            "hitDesc": [""]
            })
    write_DataFrame_and_log(os.path.join(op_dir, "cofit.tab"), 
                            tmp_df, 
                            df_name="cofit")




    #16 specphe - specific phenotypes
    if "specphe" in gene_fit_d and gene_fit_d["specphe"] is not None:
        tmp_df = genes_df[first3_cols].join(gene_fit_d['specphe'], on="locusId")
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
    write_DataFrame_and_log(os.path.join(op_dir, "specific_phenotypes.tab"), 
                            tmp_df, 
                            df_name="specific phenotypes")


    # 17 Strong - we find which normalized log ratios are greater than 2 and 
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
            "number_of_experiments": len(gene_fit_d['q']['short']) - list(gene_fit_d['q']['short']).count("Time0"),
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
        inp_df: A standard DataFrame coming out of gene_fit_d
        genes_locusId: pandas Series with locusId (str)
        genes_first3 DataFrame: Columns of genes.GC ["locusId", "sysName", "desc"]
        final_colnames: list(gene_fit_d['q']['name'] + ' ' + gene_fit_d['q']['short'])
        output_filepath: Path to write out the TSV
        df_log_name: What name of output to report

    Subroutines: 
        combine_dataframes_on_column
    """

    pre_merge = input_df.copy(deep=True)
    pre_merge['locusId'] = genes_locusId
    tmp_df = genes_first3.merge(pre_merge, on="locusId") 
    write_DataFrame_and_log(output_filepath, 
                            tmp_df, 
                            df_name=df_log_name)
    


    return None


def create_strong_tab(gene_fit_d, genes_df, exps_df, op_dir, debug=False):
    which_are_strong = []
    for col in gene_fit_d['lrn'].columns:
        for row_ix in range(gene_fit_d['lrn'].shape[0]):
            if abs(gene_fit_d['lrn'][col].iloc[row_ix]) > 2 and \
                abs(gene_fit_d['t'][col].iloc[row_ix]) > 5:
                which_are_strong.append([col, row_ix])
    strong_t = [gene_fit_d['t'][x[0]].iloc[x[1]] for x in which_are_strong]
    strong_lrn = [gene_fit_d['lrn'][x[0]].iloc[x[1]] for x in which_are_strong]
    if isinstance(gene_fit_d['g'], pd.DataFrame):
        locusIds = gene_fit_d['g']['locusId']
    else:
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
        strong_df = strong_df.astype({"locusId": str})

        sysName = []
        desc = []
        for locusId in strong_df["locusId"]:
            if debug:
                print(f"current locusId: {locusId}")
                print(f"type of locusId: {type(locusId)}")
            sysName.append(genes_df[genes_df["locusId"] == str(locusId)]["sysName"].values[0])
            desc.append(genes_df[genes_df["locusId"] == str(locusId)]["desc"].values[0])

        strong_df["sysName"] = sysName
        strong_df["desc"] = desc

        short = []
        for exp_name in strong_df["name"]:
            short.append(exps_df[exps_df["name"] == exp_name]["short"].values[0])

        strong_df["short"] = short

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




    

    







"""

FEBA_Save_Tables = function(fit, genes, org="?",
		 topdir="data/FEBA/html/",
		 dir = paste(topdir,org,sep="/"),
		 writeImage=TRUE,
		 FEBAdir="src/feba",
		 template_file=paste(FEBAdir,"/lib/FEBA_template.html",sep=""),
		 expsU=expsUsed,
		 ... # for FEBA_Quality_Plot
		 ) {

    # ARGS:
    #
    #
    #

	if(!file.exists(dir)) dir.create(dir);

	for (n in words("q lr lrn lrn1 lrn2 t")) {
	    if (is.null(fit[[n]]) || !is.data.frame(fit[[n]])) {
	        stop("Invalid or missing ",n," entry");
	    }
	}
	if (is.null(fit$genesUsed)) stop("Missing genesUsed");
	if (is.null(fit$g)) stop("Missing g -- versioning issue?");

	if(!all(names(fit$lr) == fit$q$name)) stop("Name mismatch");
	if(!all(names(fit$lrn) == fit$q$name)) stop("Name mismatch");

	nameToPath = function(filename) paste(dir,filename,sep="/");
	wroteName = function(x) cat("Wrote ",nameToPath(x),"\n",file=stderr());

	writeDelim(fit$q, nameToPath("fit_quality.tab"));
	wroteName("fit_quality.tab");

	writeDelim(cbind(genes, used=genes$locusId %in% fit$genesUsed), nameToPath("fit_genes.tab"));
	wroteName("fit_genes.tab");

        ###
    
        #3
	d = merge(genes[,c("locusId","sysName","desc")], cbind(locusId=fit$g,fit$lr));
	names(d)[-(1:3)] = paste(fit$q$name,fit$q$short);
	writeDelim(d, nameToPath("fit_logratios_unnormalized.tab"));
	wroteName("fit_logratios_unnormalized.tab");

        #4
	d = merge(genes[,c("locusId","sysName","desc")], cbind(locusId=fit$g,fit$lrNaive));
	names(d)[-(1:3)] = paste(fit$q$name,fit$q$short);
	writeDelim(d, nameToPath("fit_logratios_unnormalized_naive.tab"));
	wroteName("fit_logratios_unnormalized_naive.tab");

        ###
        #5
	d = merge(genes[,c("locusId","sysName","desc")], cbind(locusId=fit$g,fit$lrn));
	names(d)[-(1:3)] = paste(fit$q$name,fit$q$short);
	writeDelim(d, nameToPath("fit_logratios.tab"));
	wroteName("fit_logratios.tab");
        
        #6
	d = merge(genes[,c("locusId","sysName","desc")], cbind(locusId=fit$g,fit$lrn1));
	names(d)[-(1:3)] = paste(fit$q$name,fit$q$short);
	writeDelim(d, nameToPath("fit_logratios_half1.tab"));
	wroteName("fit_logratios_half1.tab");

        ###
        #7
	d = merge(genes[,c("locusId","sysName","desc")], cbind(locusId=fit$g,fit$lrn2));
	names(d)[-(1:3)] = paste(fit$q$name,fit$q$short);
	writeDelim(d, nameToPath("fit_logratios_half2.tab"));
	wroteName("fit_logratios_half2.tab");


        #8
	d = genes[genes$locusId %in% fit$g, c("locusId","sysName","desc")];
	d$comb = paste(d$sysName, d$desc); # for MeV
	if (sum(fit$q$u) == 0) {
		cat("Warning: 0 OK experiments\n");
                d = d[order(d$locusId),]; # ensure same order as other tables
	} else {
		d = merge(d, cbind(locusId=fit$g,fit$lrn[,fit$q$u]));
		names(d)[-(1:4)] = paste(fit$q$name,fit$q$short)[fit$q$u];
	}
	writeDelim(d, nameToPath("fit_logratios_good.tab"));
	cat("Wrote fitness for ",sum(fit$q$u), " successful experiments to ", nameToPath("fit_logratios_good.tab"),"\n",
	    file=stderr());

        ###
        #9
	d = genes[genes$locusId %in% fit$g, c("locusId","sysName","desc")];
	d$comb = paste(d$sysName, d$desc); # for MeV
        d = merge(d, cbind(locusId=fit$g, fit$tot));
        names(d)[-(1:4)] = paste(fit$q$name,fit$q$short);
	writeDelim(d, nameToPath("gene_counts.tab"));
        wroteName("gene_counts.tab");
    
        #10
	d = merge(genes[,c("locusId","sysName","desc")], cbind(locusId=fit$g,fit$t));
	names(d)[-(1:3)] = paste(fit$q$name,fit$q$short);
	writeDelim(d, nameToPath("fit_t.tab"));
	wroteName("fit_t.tab");

        #11
	d = merge(genes[,c("locusId","sysName","desc")], cbind(locusId=fit$g,fit$se));
	names(d)[-(1:3)] = paste(fit$q$name,fit$q$short);
	writeDelim(d, nameToPath("fit_standard_error_obs.tab"));
	wroteName("fit_standard_error_obs.tab");

        #12
	d = merge(genes[,c("locusId","sysName","desc")], cbind(locusId=fit$g,fit$sdNaive));
	names(d)[-(1:3)] = paste(fit$q$name,fit$q$short);
	writeDelim(d, nameToPath("fit_standard_error_naive.tab"));
	wroteName("fit_standard_error_naive.tab");

        #13
	writeDelim(cbind(fit$strains,fit$strain_lrn)[order(fit$strains$scaffold, fit$strains$pos),],
		nameToPath("strain_fit.tab"));
	wroteName("strain_fit.tab");


        ### Skipping:

	FEBA_Quality_Plot(fit$q, nameToPath("fit_quality.pdf"), org, ...);
	wroteName("fit_quality.pdf");

	if(is.null(fit$pairs)) {
		paste("No data for cofitness plot\n");
		unlink(nameToPath("cofitness.pdf"));
	} else {
		FEBA_Cofitness_Plot(fit$pairs, nameToPath("cofitness.pdf"), org);
		wroteName("cofitness.pdf");
	}

	pdf(nameToPath("fit_quality_cor12.pdf"),
		pointsize=10, width=6, height=6,
		title=paste(org,"Fitness Cor12 Plots"));
	for (i in 1:nrow(fit$q)) {
	    n = as.character(fit$q$name[i]);
	    changers = fit$g[abs(fit$t[[n]]) >= 3];
	    plot(fit$lrn1[[n]], fit$lrn2[[n]],
	    		  main=sprintf("%s %s #%d (gMed=%.0f rho12=%.3f)\n%s",
			  	org, n, fit$q$num[i], fit$q$gMed[i], fit$q$cor12[i], fit$q$short[i]),
	    		  xlab="First Half", ylab="Second Half",
			  col=ifelse(fit$g %in% changers, 2, 1));
	    eqline(); hline(0); vline(0);
	}
         
        ##





        # Start Skip

	dev.off();
	wroteName("fit_quality_cor12.pdf");

	labelAll = sprintf("%s #%d gMed=%.0f rho12=%.2f %30.30s",
		      sub("^set","",fit$q$name), fit$q$num, fit$q$gMed, fit$q$cor12, fit$q$short);
        labelAll = ifelse(fit$q$short=="Time0", paste(labelAll, fit$q$t0set), labelAll);

	use = fit$q$short != "Time0";
	if(sum(use) > 2) {
	    lrClust = hclust(as.dist(1-cor(fit$lrn[,as.character(fit$q$name)[use]], use="p")));
	    pdf(nameToPath("fit_cluster_logratios.pdf"),
		pointsize=8, width=0.25*pmax(8,sum(use)), height=8,
		title=paste(org,"Cluster Logratios"));
	    plot(lrClust, labels=labelAll[use], main="");
	    dev.off();
	    wroteName("fit_cluster_logratios.pdf");
	}

	if (ncol(fit$gN)-1 >= 3) { # at least 3 things to cluster
	    countClust = hclust(as.dist(1-cor(log2(1+fit$gN[fit$gN$locusId %in% fit$genesUsed,-1]))));
	    pdf(nameToPath("fit_cluster_logcounts.pdf"),
		pointsize=8, width=pmax(5,0.25*nrow(fit$q)), height=8,
		title=paste(org,"Cluster Log Counts"));
	    # Some Time0s may be missing from fit$q
	    d = match(names(fit$gN)[-1], fit$q$name);
	    labelAll2 = ifelse(is.na(d), paste("Time0", sub("^set","",names(fit$gN)[-1])), labelAll[d]);
	    plot(countClust, labels=labelAll2, main="");
	    dev.off();
	    wroteName("fit_cluster_logcounts.pdf");
	}

        d = table(genes$scaffoldId[genes$locusId %in% fit$genesUsed]);
	maxSc = names(d)[which.max(d)];
	if (is.null(maxSc)) stop("Invalid scaffoldId?");
	beg = ifelse(fit$g %in% genes$locusId[genes$scaffold==maxSc],
	    genes$begin[match(fit$g, genes$locusId)], NA);

	pdf(nameToPath("fit_chr_bias.pdf"), pointsize=10, width=6, height=6,
	          title=paste(org,"Chromosome Bias"));
	for (i in 1:nrow(fit$q)) {
	    n = as.character(fit$q$name[i]);
	    plot(beg, pmax(-2,pmin(2,fit$lr[[n]])),
	    		  main=sprintf("%s %s #%d (gMed=%.0f rho12=%.3f)\n%s",
			  	org, sub("^set","",n), fit$q$num[i], fit$q$gMed[i], fit$q$cor12[i], fit$q$short[i]),
	    		  xlab="Position on Main Scaffold",
			  ylab="Fitness (Unnormalized)",
			  ylim=c(-2,2), col="darkgrey");
	    o = order(beg);
	    lines(beg[o], (fit$lr[[n]] - fit$lrn[[n]])[o], col="darkgreen", lwd=2);
	    hline(0,lty=1,col=1);
	}
	dev.off();
	wroteName("fit_chr_bias.pdf");

        # Done skip

        #14 expsUsed
	if (!is.null(expsU)) {
		writeDelim(expsU, nameToPath("expsUsed"));
		wroteName("expsUsed");
	}

        #15 Cofit
	if (is.null(fit$cofit)) {
	    d = data.frame(locusId="",sysName="",desc="",hitId="",cofit=0,rank=0,hitSysName="",hitDesc="");
	} else {
	    d = merge(genes[,words("locusId sysName desc")], fit$cofit, by="locusId");
	    d = merge(d, data.frame(hitId=genes$locusId, hitSysName=genes$sysName, hitDesc=genes$desc));
	    d = d[order(d$locusId,d$rank),];
	}
	writeDelim(d, nameToPath("cofit"));
	wroteName("cofit");

        #16 specphe
	if (is.null(fit$specphe)) {
	   d = data.frame(locusId="",sysName="",desc="",name="",short="",Group="",Condition_1="",Concentration_1="",Units_1="",
	                  Condition_2="",Concentration_2="",Units_2="");
	   d = d[0,];
	} else {
	   d = merge(genes[,words("locusId sysName desc")], fit$specphe, by="locusId");
	}
	writeDelim(d, nameToPath("specific_phenotypes"));
	wroteName("specific_phenotypes");

        #17 Strong
        d = which(abs(fit$lrn) > 2 & abs(fit$t) > 5, arr.ind=T);
        if (nrow(d) >= 1) {
	  out = data.frame(locusId=fit$g[d[,1]], name=names(fit$lrn)[d[,2]], lrn=fit$lrn[d], t=fit$t[d]);
	  out = merge(genes[,words("locusId sysName desc")], merge(expsU[,c("name","short")], out));
	  writeDelim(out, nameToPath("strong.tab"));
	  wroteName("strong.tab");
	}

        #18 High
        # High Fitness
        writeDelim(fit$high, nameToPath("high_fitness.tab"));
        wroteName("high_fitness.tab");

	if(writeImage) {
	    img = format(Sys.time(),"fit%Y%b%d.image"); # e.g., fit2013Oct24.image
	    expsUsed = expsU;
	    save(fit, genes, expsUsed, file=nameToPath(img));
	    wroteName(img);
	    unlink(nameToPath("fit.image"));
	    file.symlink(img, nameToPath("fit.image"));
	    cat("Created link for ",nameToPath("fit.image"),"\n", file=stderr());
	}

        #19 HTML:
	if(!is.null(template_file)) {
	    FEBA_Save_HTML(nameToPath("index.html"), template_file,
			   list(ORG=org,
			        NEXPS=sum(fit$q$short != "Time0"),
				NSUCCESS=sum(fit$q$u),
				VERSION=fit$version,
				DATE=date()));
	    wroteName("index.html");
	}
}
"""

