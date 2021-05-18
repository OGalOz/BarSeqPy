#!python3
# A command-line wrapper for running FEBA.R (Written by Morgan Price)
# Can be invoked from the command-line to analyze the fitness experiments and save the results
# (i.e., FEBA_Fit() and FEBA_Save_Tables()) - the two main functions from FEBA.R
#   used to do computations.

import pandas as pd
import numpy as np
import os
import logging
import sys
import json
import datetime
import re
from translate_R_to_pandas import *
from data_prep1 import data_prep_1
from data_prep2 import data_prep_2
from analysis1 import analysis_1 
from analysis2 import analysis_2
from analysis3 import analysis_3
from FEBA_main import FEBA_Fit 
from FEBA_Save_Tables import FEBA_Save_Tables 
#, FEBA_Save_Tables 


"""
Main functions in file:
   data_prep_1 
"""


def RunFEBA(org_str, data_dir, FEBA_dir, start_point,
            debug_bool=False, meta_ix=7):
    """ Entrypoint into running the FEBA analysis program

    Args:
        org_str: (str) Name of organism
        data_dir: (str) Path to directory which contains the 
                    following files: 'all.poolcount', 'genes',
                            'exps', 'pool' - all TSV files.
                            Optionally contains the following files:
                                strainusage.barcodes.json - json list
                                strainusage.genes.json - json list
                                strainusage.genes12.json - json list
                                ignore_list.json - json list ( list of str 
                                            with sample-index name to ignore )
                    All these files are changed depending on the input.
        FEBA_dir: (str) Path to directory which contains the 
                    following files: 'desc_short_rules'
        start_point (int): Where you want to start running the program
        debug_bool: Whether you'd like to print the dataframes
                as a test to the data_dir before running FEBA_Fit
   
    Functions:
        applyRules: Replaces unnecessary parts in condition names if they match 
                values in input table 'desc_short_rules' in FEBA_dir

    Note:
        all.poolcount has to have the same number of columns that aren't index names
            as the value 'meta_ix', and must have columns:
             "locusId", "f"
        exps file must have columns:
            "SetName", "Index", "Date_pool_expt_started", "Description"
            column added: "num"
        genes must have columns:
            "scaffoldId", "locusId", "sysName", "desc", "begin", "end"

    """
    
    if start_point == 1:
        # Part 1 - Data Preparation 1
        res_dp1 = data_prep_1(data_dir,
                          FEBA_dir, 
                          debug_bool=False,
                          meta_ix=meta_ix)

        logging.info("\n\n\nRunning Section 1 - Data Prep 1\n\n\n")
        exps_df, all_df, genes_df, genesUsed_list, ignore_list = res_dp1

        breakpoint1(exps_df, all_df, genes_df, genesUsed_list, ignore_list,
                    to_file=True)
        # End Part 1 - Data Preparation 1

    if start_point <= 2:

        if start_point == 2:
            res_from_dir = import_start_point2_data_from_dir("tmp/BP1")
            all_df, exps_df, genes_df, genesUsed_list, ignore_list = res_from_dir

        logging.info("\n\n\nRunning Section 2 - Data Prep 2\n\n\n")
        # Part 2 - Data Preparation 2; Control Experiments
        res_dp2 = data_prep_2(exps_df, all_df, genes_df, genesUsed_list,
                    ignore_list, minSampleReads=2*10e4,
                    meta_ix=7, 
                    okDay=True, okLane=True,
    	            minT0Strain=3, minT0Gene=30,
                    dbg_prnt=True,
                    dbg_lvl=10,
                    export_vars_bool=True)
    
        all_df, exps_df, genes_df, genesUsed_list = res_dp2[0]
        strainsUsed_list, genesUsed_list12, t0_gN, t0tot = res_dp2[1]
        central_insert_bool_list, expsT0, num_vars_d = res_dp2[2]
        breakpoint2("tmp/BP2", genesUsed_list, 
                strainsUsed_list, genesUsed_list12, 
                t0_gN, t0tot, central_insert_bool_list,
                all_df, exps_df, genes_df,
                expsT0, num_vars_d)
    if start_point <= 3:    
        if start_point == 3:
            res_from_dir = import_start_point3_data_from_dir("tmp/BP2")
            genesUsed_list, strainsUsed_list, genesUsed_list12 = res_from_dir[0:3]
            central_insert_bool_list, t0_gN, t0tot = res_from_dir[3:6]
            all_df, exps_df, genes_df, expsT0 = res_from_dir[6:]

        logging.info("\n\n\nRunning Section 3 - Analysis 1\n\n\n")
        GeneFitResults = analysis_1(all_df, exps_df, genes_df,
                                expsT0, t0tot, 
                                genesUsed_list, genesUsed_list12, strainsUsed_list, 
                                central_insert_bool_list, 
                                minGenesPerScaffold=10, meta_ix=7,
                                debug=False, nDebug_cols=4,
                                starting_debug_col=10)
        breakpoint3("tmp/BP3", GeneFitResults)

    if start_point <= 4:
        if start_point == 4:
            # Import GeneFitResults
            GeneFitResults = export_or_import_genefitresults(None, "imp", "tmp/BP3", dbg_print=False)
            # We can use results from previous imports for the rest
            res_from_dir = import_start_point3_data_from_dir("tmp/BP2")
            genesUsed_list, strainsUsed_list, genesUsed_list12 = res_from_dir[0:3]
            central_insert_bool_list, t0_gN, t0tot = res_from_dir[3:6]
            all_df, exps_df, genes_df, expsT0 = res_from_dir[6:]
      
        logging.info("\n\n\nRunning Section 4 - Analysis 2\n\n\n")
        gene_fit_d, CrudeOp_df = analysis_2(GeneFitResults, exps_df, all_df, 
                                            genes_df, central_insert_bool_list,
                                            strainsUsed_list, t0tot, 
                                            meta_ix=7, debug=False, minT0Strain=3)

        breakpoint4("tmp/BP4", gene_fit_d, CrudeOp_df)

    if start_point <= 5:
        if start_point == 5:
            gene_fit_d, CrudeOp_df = import_gene_fit_d_and_CrudeOp("tmp/BP4") 
            GeneFitResults = export_or_import_genefitresults(None, 
                                                            "imp", 
                                                            "tmp/BP3", 
                                                            dbg_print=False)
            # We can use results from previous imports for the rest
            res_from_dir = import_start_point3_data_from_dir("tmp/BP2")
            genesUsed_list, strainsUsed_list, genesUsed_list12 = res_from_dir[0:3]
            central_insert_bool_list, t0_gN, t0tot = res_from_dir[3:6]
            all_df, exps_df, genes_df, expsT0 = res_from_dir[6:]

        logging.info("\n\n\nRunning Section 5- Analysis 3\n\n\n")
        gene_fit_d = analysis_3(gene_fit_d, GeneFitResults, genes_df, all_df, exps_df,
               genesUsed_list, strainsUsed_list, genesUsed_list12,
               t0_gN, t0tot, CrudeOp_df, central_insert_bool_list,
               meta_ix=7, minT0Strain=3, dbg_prnt=False)
        
        breakpoint5("tmp/BP5", gene_fit_d)

    if start_point <= 6:
        if start_point == 6:
            gene_fit_d = import_gene_fit_d_and_CrudeOp("tmp/BP5", CrudeOp_bool=False) 
            res_from_dir = import_start_point3_data_from_dir("tmp/BP2")
            genesUsed_list, strainsUsed_list, genesUsed_list12 = res_from_dir[0:3]
            all_df, exps_df, genes_df, expsT0 = res_from_dir[6:]
            gene_fit_d['genesUsed'] = genesUsed_list

        logging.info("\n\n\nRunning Section 6 - FEBA Save Tables\n\n\n")
        FEBA_Save_Tables(gene_fit_d, genes_df, org_str,
                         "tmp/Output", exps_df, writeImage=False)

    
    '''
    # Note that strainsUsed_list could be None or a list of bools 
    # genesUsed_list could be None or a list of str
    # genesUsed12_list could be None or a list of str
    gene_fit_d = FEBA_Fit(exps_df, all_df, genes_df, dir=data_dir,
    	           strainsUsed=strainsUsed_list, genesUsed=genesUsed_list, 
                   genesUsed12=genesUsed12_list,
                   minSampleReads=0, debug=True, dbg_lvl=10,
                   debug_cols=20)

    FEBA_Save_Tables(gene_fit_d, genes_df, organism_name_str,
                     op_dir, exps_df, writeImage=False)
    '''
    '''
    FEBA_Save_Tables(fit_df, genes_df, org, expsU=exps, dir=dir, FEBAdir=FEBAdir);
    with open(os.path.join(data_dir, ".FEBA.success"),'w') as g:
        g.write(str(datetime.datetime.now()))
    '''

    sys.exit(0)


def breakpoint1(exps_df, all_df, genes_df, genesUsed_list,
                ignore_list, to_file=True, op_dir=None):
    """

    Description:
        We take basic info from all the data frames (column number, row number)
        and log it. We also take the length of the lists and log it.
        We get number of unique locusIds in all_df and genes_df
        If to_file is set to True, then we also print the dataFrames to the tmp
        dir.
    """
    
    logging.info("Reached Breakpoint 1.")
    logging.info(f"For exps_df, Row|col number is {exps_df.shape[0]}|{exps_df.shape[1]}")
    logging.info(f"For all_df, Row|col number is {all_df.shape[0]}|{all_df.shape[1]}")
    logging.info(f"For genes_df, Row|col number is {genes_df.shape[0]}|{genes_df.shape[1]}")
    
    for x in [[ignore_list, 'ignore_list'], 
            [genesUsed_list, 'genesUsed_list']]:
        if x[0] is not None:
            if isinstance(x[0], list):
                logging.info(f"Length of {x[1]} is {len(x[0])}")
            else:
                raise Exception("Expected {x[1} to be list or NoneType, instead {type(x[0])}")
    
    nUnq_in_genes = len(genes_df['locusId'].unique())
    nUnq_in_all = len(all_df['locusId'].unique())
    logging.info(f"unique locusIds in genes_df|all_df: {nUnq_in_genes}|{nUnq_in_all}")

    if to_file:
        if op_dir == None:
            op_dir = os.path.join("tmp","BP1")
        exps_df.to_csv(os.path.join(op_dir,"py_exps_df.tsv"),sep="\t", index=False)
        all_df.to_csv(os.path.join(op_dir,"py_all_df.tsv"),sep="\t", index=False)
        genes_df.to_csv(os.path.join(op_dir,"py_genes_df.tsv"),sep="\t", index=False)
        logging.info("Wrote the dataframes to py_(dataframe) in the tmp dir")
        with open(os.path.join(op_dir,"py_genesUsed_list.json"), "w") as g:
            g.write(json.dumps(genesUsed_list, indent=2))
        with open(os.path.join(op_dir,"py_ignore_list.json"), "w") as g:
            g.write(json.dumps(ignore_list, indent=2))
        logging.info("Wrote the lists to py_(listname).json in the tmp dir")

     


def start_at_analysis1(inp_data_dir):
    """
    Args:
        inp_data_dir (str): Directory which contains all files needed
                            to run analysis_1
    """







#def data_prep_1(data_dir, FEBA_dir, debug_bool=False, meta_ix=7):
""" The first phase of data preparation for the BarSeqR Computations
    Args:
        data_dir: (str) Path to directory which contains the 
                    following files: 'all.poolcount', 'genes',
                            'exps', 'pool' - all TSV files.
                            Optionally contains the following files:
                                strainusage.barcodes.json - json list
                                strainusage.genes.json - json list
                                strainusage.genes12.json - json list
                                ignore_list.json - json list ( list of str 
                                            with sample-index name to ignore )
                    All these files are changed depending on the input.

        FEBA_dir: (str) Path to directory which contains the 
                    following files: 'desc_short_rules'
        debug_bool: Whether you'd like to print the dataframes
                as a test to the data_dir before running FEBA_Fit

    Returns:
        list<exps_df, all_df, genes_df, 
            strainsUsed_list, genesUsed_list, genesUsed12_list>
            
            exps_df (pandas DataFrame): Must contain cols: (Variable)
      
            all_df (pandas DataFrame): Must contain cols:
            
            genes_df (pandas DataFrame): Must contain cols:
                scaffold, begin

            strainsUsed_list (py list or None):

            genesUsed_list (py list or None):

            genesUsed12_list (py list or None):
                
       
    Description:
        Within data_prep1 we perform the following functions:
          getDataFrames:
            We import the tables genes, all, exps, rules using a dict to say which 
              data type is in each column.
            We might remove the rows who have 'Drop' set to True (if drop_exps==True).
            We remove the spaces from the values in 'Group', 'Condition_1', 'Condition_2'
            We check that the right column names exist in each of the tables.
          checkLocusIdEquality:
            We check all the locusIds in all_df are also present in genes_df
            If debugging we also print the number of unique locusIds in each.
          check_exps_df_against_all_df:
            We check that the index names in all.poolcount are equivalent to the 
                'SetName' + '.' + 'Index' in exps
          prepare_set_names:  
            We replace the SetNames from the complicated version to a simpler one,
            remove the period in between SetName and Index in all.poolcount columns,
            and make the 'names' column in the experiments file and the all.poolcount columns
            have the same values. For example, we move column name from Keio_ML9_set2.IT004 to 
            set2IT004, and rename the values in the Experiments file similarly.
          get_special_lists:
            We get the lists from the files in data_dir if they are there.
            Otherwise we return their values as None.

        If debug_bool we print out resultant exps, all, genes to 'tmp' dir
          
"""

'''

    genes_df, all_df, exps_df, rules_df = getDataFrames(data_dir, FEBA_dir, dbg_lvl=2)

    # Makes no changes to the variables
    checkLocusIdEquality(all_df, genes_df, debug_bool=debug_bool)
    # We check that SetNames and Indexes in experiments file match all.poolcount file
    check_exps_df_against_all_df(exps_df, all_df, meta_ix, debug_bool=debug_bool)
    # We make it so the names are cleaner and create 'names', 'num', 'short' in exps_df
    exps_df, all_df, replace_col_d = prepare_set_names(exps_df, all_df, rules_df, debug_bool=debug_bool)

    strainsUsed_list, genesUsed_list, genesUsed12_list, ignore_list = get_special_lists(data_dir, all_df,
                                                                    replace_col_d, debug_bool=debug_bool)

    if debug_bool:
        exps_df.to_csv("tmp/py_test1_exps_fp.tsv", sep="\t")
        all_df.to_csv("tmp/py_test1_all_fp.tsv", sep="\t")
        genes_df.to_csv("tmp/py_test1_genes_fp.tsv", sep="\t")

    return [exps_df, all_df, genes_df, strainsUsed_list, genesUsed_list, genesUsed12_list]
'''


def get_special_lists(data_dir, all_df, replace_col_d, debug_bool=False):
    """
    Args:
        replace_col_d: Dict mapping original all_df experiment name to replacement name
        data_dir

    Returns:
        strainsUsed_list
        genesUsed_list
        genesUsed12_list
        ignore_list: List<str> New names for the experiments we want to ignore.

    Description: We get the lists from the files in data_dir if they are there.
                Otherwise we return their values as empty lists
    """

    strainsUsed_list = []
    genesUsed_list = []
    genesUsed12_list = []
    ignore_list = []
    # list of booleans
    strainsUsed_fp = os.path.join(data_dir, "strainusage.barcodes.json")
    # list of locusIds
    genesUsed_fp = os.path.join(data_dir, "strainusage.genes.json")
    # list of locusIds
    genesUsed12_fp = os.path.join(data_dir, "strainusage.genes12.json")
    # list of extra ignored experiments
    ignore_list_fp = os.path.join(data_dir, "ignore_list.json")

    # We check that all the optional files have read access
    can_parse_extra = True
    for x in [strainsUsed_fp, 
              genesUsed_fp,
              genesUsed12_fp]:
        if not (os.path.isfile(x) and os.access(x, os.R_OK)):
            can_parse_extra = False

    if can_parse_extra:
        # (Intersection of all_df['barcode'] and barcodesUsed_list) -> strainsUsed_list
        barcodesUsed_list = json.loads(open(strainsUsed_fp).read())
        strainsUsed_list = [x for x in all_df['barcode'] if x in barcodesUsed_list]
        genesUsed_list = json.loads(open(GenesUsed_fp).read())
        genesUsed12_list = json.loads(open(GenesUsed12_fp).read())
        logging.info(f"Loaded {len(list(barcodesUsed))} strains and "
                      f"{len(genesUsed_list)} genes to include in the"
                        "analysis\n")

    if os.path.isfile(ignore_list_fp) and os.access(ignore_list_fp, os.R_OK):
        pre_ignore_list = json.loads(open(ignore_list_fp).read())
        for x in pre_ignore_list:
            if x in replace_col_d:
                ignore_list.append(x)
            else:
                raise Exception(f"Avoid list contains experiment {x} but experiment name"
                                " not found in all.poolcount."
                                f" Possible names: {', '.join(list(replace_col_d.keys()))}")


        ignore_list = [replace_col_d[x] for x in ignore_list]


    return strainsUsed_list, genesUsed_list, genesUsed12_list, ignore_list


def check_exps_df_against_all_df(exps_df, all_df, meta_ix, debug_bool=True):
    experiment_names_test = [exps_df['SetName'].iat[i] + "." + exps_df['Index'].iat[i] for i in \
                        range(len(exps_df['SetName']))]
    index_names = list(all_df.head())[meta_ix:]

    # Number of rows:
    if len(index_names) != exps_df.shape[0]:
        raise Exception(f"Number of data columns in {all_fp} does not match"
                        f" number of rows in {exps_fp}\n"
                        f"{len(index_names)} != {exps_df.shape[0]}")
    for i in range(len(index_names)):
        if index_names[i] not in experiment_names_test:
            raise Exception(f"Column names in {all_fp} do not match names from"
                            f"{exps_fp} at index {i}")


    if debug_bool:
        logging.info("Checking length and names passed")


def prepare_set_names(exps_df, all_df, rules_df, debug_bool=False):
    """


    Description:
        We replace the SetNames from the complicated version to a simpler one,
        remove the period in between SetName and Index in all.poolcount columns,
        and make the 'names' column in the experiments file and the all.poolcount columns
        have the same values. For example, we move column name from Keio_ML9_set2.IT004 to 
        set2IT004, and rename the values in the Experiments file similarly.
    """

    # Below is a numpy array, not a series
    uniqueSetNames_nparray = exps_df['SetName'].unique()
    # shortSetNames is numpy ndarray, shortNamesTranslation_d is a dict which contains
    #   conversions from original names to short names.
    shortSetNames, shortNamesTranslation_d = ShortSetNames(uniqueSetNames_nparray)

    if debug_bool:
        logging.debug("uniqueSetNames:")
        logging.debug(uniqueSetNames_nparray)
        logging.debug("shortSetNames")
        logging.debug(shortSetNames)
        logging.debug("Above 2 arrays should be the same length.")


    # We concatenate the string of the set name and the index column
    # But first we need to find the original location of the set name
    # match_list is a list of indeces (int) for each element in the first list
    # where it is found in the second list.
    match_list = match_ix(list(exps_df['SetName']), list(uniqueSetNames_nparray)) 
    # We apply the match list to shortSetNames_list to recreate the original SetName order
    #       just with the newly created 'short' setNames.
    short_names_srs = shortSetNames[match_list]
    
    if debug_bool:
        print("short_names_srs: (shortSetNames[match_list])")
        print(short_names_srs)
        print("original set Names:")
        print(exps_df['SetName'])
        print('match_list')
        print(match_list)
        # If there are 3 unique set names and 100 items in exps_df['SetName'],
        # then match_list will contain 100 items with only 3 different values (0, 1, 2)

    # expNamesNew ends up being a list<str>
    expNamesNew = [] 
    for i in range(len(short_names_srs)):
        if not short_names_srs[i] in [None, np.nan]:
            expNamesNew.append(short_names_srs[i] + exps_df['Index'][i])
        else:
            expNamesNew.append(exps_df['Index'][i])

    if debug_bool:
        print('expNamesNew:')
        print(expNamesNew)




    exps_df['num'] = range(1, exps_df.shape[0] + 1)
    exps_df['short'] = pd.Series(applyRules(rules_df, list(exps_df['Description'])))

    if debug_bool:
        print("exps_df of col 'short':")
        print(exps_df['short'])

    # We remove the "." in the names of the values. Just SetNameIndex now
    replace_col_d = {list(all_df.head())[7 + i]: expNamesNew[i] for i in range(len(expNamesNew))}
    if debug_bool:
        print('replace_col_d')
        print(replace_col_d)
        print('original all_df col names:')
        print(list(all_df.columns))
    all_df = all_df.rename(columns=replace_col_d)
    if debug_bool:
        print('after replacement all_df col names:')
        print(list(all_df.columns))
    
    exps_df['name'] = expNamesNew

    return exps_df, all_df, replace_col_d



def checkLocusIdEquality(all_df, genes_df, debug_bool=False):
    """ We check all the locusIds in all_df are also present in genes_df

    
    Description:
        We check all the locusIds in all_df are also present in genes_df
        If debugging we also print the number of unique locusIds
    """

    if debug_bool:
        logging.debug("Original locusId col")
        logging.debug(all_df['locusId'])

    # below both are pandas series
    unique_all_locusIds = all_df['locusId'].dropna().unique()
    unique_genes_locusIds = genes_df['locusId'].dropna().unique()

    if debug_bool:
        # All
        logging.debug("Unique All Locus Ids: ")
        logging.debug(unique_all_locusIds)
        logging.debug("Number of Unique All Locus Ids: ")
        logging.debug(len(unique_all_locusIds))
        # Genes
        logging.debug("Unique Gene Locus Ids: ")
        logging.debug(unique_genes_locusIds)
        logging.debug("Number of Unique Gene Locus Ids: ")
        logging.debug(len(unique_genes_locusIds))
   

    # Checking if every locusId from all.poolcount also exists in genes
    not_found_locusIds = []
    for x in unique_all_locusIds:
        if x not in unique_genes_locusIds:
            not_found_locusIds.append(x)
    if len(not_found_locusIds) > 0:
        raise Exception("The following locusIds were not found in the genes file."
                        " (All locusIds from all.poolcount must also be in the genes"
                        " file.)"
                        "', '".join(not_found_locusIds))


def getDataFrames(data_dir, FEBA_dir, drop_exps=False, dbg_lvl=0):
    """
    Args:
        data_dir: (str) Path to directory which contains the 
                    following files: 'all.poolcount', 'genes',
                            'exps', 'pool' - all TSV files.
                            Optionally contains the following files:
                                strainusage.barcodes.json - json list
                                strainusage.genes.json - json list
                                strainusage.genes12.json - json list
                    All these files are changed depending on the input.

        FEBA_dir: (str) Path to directory which contains the 
                    following files: 'desc_short_rules'
        drop_exps (bool): Should we drop all experiments that have Drop=True
                        already?

    Returns:
        genes_df (pandas DataFrame): Contains columns:
            locusId, sysName, type, scaffoldId, begin, end, strand, name, desc, GC, nTA
        all_df (pandas DataFrame): Contains columns:
            barcode, rcbarcode, scaffold, strand, pos, locusId, f, setName1, ..., setNameN
        exps_df (pandas DataFrame): Contains columns:
                Index (str)
                Date_pool_expt_started (str)
                Description (str)
                SetName (Str)
                Group (str)
                Drop (bool)
                [Condition_1]
                [Condition_2]
            
        rules_df (pandas DataFrame): Contains columns:
            V1 (str): Original string to replace
            V2 (str): String to replace V1 by

    Description:
        We import the tables using a dict to say which data type is in each column.
        We might remove the rows who have 'Drop' set to True (if drop_exps==True).
        We remove the spaces from the values in 'Group', 'Condition_1', 'Condition_2'
        We check that the right column names exist in each of the tables.

    To Do:
        Should we strip all of the column names when we import them?
    """

    data_files = os.listdir(data_dir)
    for x in ["all.poolcount", "genes", "exps", "pool"]:
        if x not in data_files:
            raise Exception("Input data_dir to RunFEBA must include files:\n"
                            "all.poolcount, genes, exps, and pool."
                            " Currently missing: " + x)


    all_fp = os.path.join(data_dir,"all.poolcount")
    genes_fp = os.path.join(data_dir,"genes")
    exps_fp = os.path.join(data_dir,"exps")
    pool_fp = os.path.join(data_dir,"pool")
    short_rules_fp = os.path.join(FEBA_dir, "desc_short_rules.tsv")

    # Checking access permissions
    for x in [all_fp, genes_fp, exps_fp, pool_fp]:
        if not os.access(x, os.R_OK):
            raise Exception("To run, program requires read permission to file " + x)

    # Read tsv files into dataframes, making sure columns locusId and scaffoldId read as stings
    genes_dtypes = {
            'locusId': str,
            'sysName': str,
            'type': int,
            'scaffoldId': str,
            'begin': int,
            'end': int,
            'strand': str,
            'name': str,
            'desc': str,
            'GC': float,
            'nTA': int
            }
    genes_df = pd.read_table(genes_fp, dtype=genes_dtypes)
    #barcode	rcbarcode	scaffold	strand	pos	locusId	f
    all_dtypes = {
            'barcode': str,
            'rcbarcode': str,
            'scaffold': str,
            'strand': str,
            'pos': int,
            'locusId': str,
            'f': float
            }
    all_df = pd.read_table(all_fp, dtype=all_dtypes) 
    
    exps_dtypes = {
            'SetName': str,
            'Index': str,
            'Date_pool_expt_started': str,
            "Description": str,
            "Group": str,
            "Drop": str,
            "Condition_1": str,
            "Condition_2": str
    }
    exps_df = pd.read_table(exps_fp, dtype=exps_dtypes)

    # We update the 'Drop' experiments
    if 'Drop' in exps_df:
        new_drops = []
        remove_indeces = []
        for ix, value in exps_df['Drop'].items():
            if isinstance(value, str):
                if value.strip().upper() == "TRUE":
                    new_drops.append(True)
                    remove_indeces.append(ix)
                else:
                    new_drops.append(False)
            else:
                new_drops.append(False)
        exps_df['Drop'] = new_drops

    else:
        exps_df['Drop'] = [False]*exps_df.shape[0]

    if drop_exps:
        # Removing Drop rows
        exps_df.drop(remove_indeces, axis=0, inplace=True)

    # Remove trailing spaces:
    for x in ["Group", "Condition_1", "Condition_2"]:
        if x in exps_df:
            # We take the entire column (pandas Series) and strip the spaces
            exps_df[x] = exps_df[x].str.strip()


    rules_dtypes = {
            "V1": str,
            "V2": str
    }
    rules_df = pd.read_table(short_rules_fp, keep_default_na=False, dtype=rules_dtypes)


    for x in ["scaffoldId", "locusId", "sysName", "desc", "begin", "end"]:
        if x not in genes_df.columns:
            raise Exception(f"Genes table must include header {x}")
    # Checking exps table
    for x in ["SetName", "Index", "Date_pool_expt_started", "Description"]:
        if x not in exps_df.columns:
            raise Exception(f"Experiments table must include header {x}")
    for x in ["scaffold", "locusId", "f", "pos"]:
        if x not in all_df.columns:
            raise Exception(f"All.PoolCount file must include header {x}")

    if dbg_lvl > 1:
        print(genes_df)
        print(all_df)
        print(exps_df)
        print(rules_df)


    return [genes_df, all_df, exps_df, rules_df]






def breakpoint2(op_dir, genesUsed, 
                strainsUsed, genesUsed12, 
                t0_gN, t0tot, central_insert_bool_list,
                all_df, exps_df, genes_df, expsT0, num_vars_d):
    """
    """
    if not os.path.isdir(op_dir):
        os.mkdir(op_dir)

    # Exporting json
    for x in [["genesUsed.json",genesUsed],
              ["strainsUsed.json", strainsUsed],
              ["genesUsed12.json", genesUsed12],
              ["central_insert_bool_list.json", central_insert_bool_list],
              ["expsT0.json", expsT0],
              ["num_vars_d", num_vars_d]
              ]:
        with open(os.path.join(op_dir, x[0]), 'w') as g:
            g.write(json.dumps(x[1], indent=2))

    t0_gN.to_csv(os.path.join(op_dir, "t0_gN.tsv"), sep="\t", index=False)
    t0tot.to_csv(os.path.join(op_dir, "t0tot.tsv"), sep="\t", index=False)
    all_df.to_csv(path_or_buf=os.path.join(op_dir, "all_df.tsv"), sep="\t", index=False)
    exps_df.to_csv(path_or_buf=os.path.join(op_dir, "exps_df.tsv"), sep="\t", index=False)
    genes_df.to_csv(path_or_buf=os.path.join(op_dir, "genes_df.tsv"), sep="\t", index=False)

    logging.info(f"Breakpoint2: Exported all vars to {op_dir}")


def breakpoint3(op_dir, GeneFitResults):
    export_or_import_genefitresults(GeneFitResults, "exp", op_dir, dbg_print=False)

    logging.info(f"Breakpoint3: Exported all vars to {op_dir}")


def breakpoint4(op_dir, gene_fit_d, CrudeOp_df):
    # We print out gene_fit_d and CrudeOp_df to a directory
    export_gene_fit_d(gene_fit_d, op_dir, dbg_prnt=True)
    CrudeOp_df.to_csv(os.path.join(op_dir, "CrudeOp_df.tsv"),
                      sep="\t", index=False)
    logging.info(f"\n\nExported values for breakpoint4 to dir {op_dir}----\n\n")
    

def breakpoint5(op_dir, gene_fit_d):

    export_gene_fit_d(gene_fit_d, op_dir, dbg_prnt=True)
    logging.info(f"\n\nExported values for breakpoint5 to dir {op_dir}----\n\n")
    


def import_gene_fit_d_and_CrudeOp(data_dir, CrudeOp_bool=True):
   
    all_files = os.listdir(data_dir)
    gene_fit_d = {}
    for f in all_files:
        if ".tsv" in f:
            gene_fit_d[''.join(f.split('.')[:-1])] = pd.read_table(
                                                        os.path.join(
                                                            data_dir, f), 
                                                            dtype={'locusId':str}
                                                        )
        elif ".json" in f:
            gene_fit_d[''.join(f.split('.')[:-1])] = json.loads(open(
                                                        os.path.join(
                                                        data_dir, f)).read())
        else:
            raise Exception(f"Did not recognize file type of {f}")
  
    
    for k in gene_fit_d.keys():
        if isinstance(gene_fit_d[k], pd.DataFrame):
            if len(gene_fit_d[k].columns) == 1:
                col_name = gene_fit_d[k].columns[0]
                print(f"Key {k} in gene_fit_d associated with DataFrame with"
                      f" only a single column: {col_name}."
                      f" Replacing dataframe with Series: ")
                gene_fit_d[k] = gene_fit_d[k][col_name]


    if CrudeOp_bool:
        CrudeOp_df = gene_fit_d['CrudeOp_df']
        del gene_fit_d['CrudeOp_df']
        return [gene_fit_d, CrudeOp_df]
    else:
        return gene_fit_d


    
    
    

def import_start_point2_data_from_dir(data_dir):
    # Incomplete 
    logging.info(f"Importing data from {data_dir}")
    all_fp = os.path.join(data_dir,"py_all_df.tsv")
    exps_fp = os.path.join(data_dir,"py_exps_df.tsv")
    genes_fp = os.path.join(data_dir,"py_genes_df.tsv")
    all_df, exps_df, genes_df = get_all_exps_genes_df(all_fp, 
                                                    exps_fp, 
                                                    genes_fp)

    with open(os.path.join(data_dir,"py_genesUsed_list.json")) as g:
        genesUsed_list = json.loads(g.read())
    with open(os.path.join(data_dir,"py_ignore_list.json")) as g:
        ignore_list = json.loads(g.read())
    logging.info("Imported outputs from breakpoint 1")


    return [all_df, exps_df, genes_df, genesUsed_list, ignore_list]


def import_start_point3_data_from_dir(data_dir):
    """
    """
    # Incomplete: genesUsed, strainsUsed, genesUsed12, and central_insert_bool_list
    # not actually imported
    genesUsed = json.loads(open(os.path.join(data_dir, "genesUsed.json")).read())
    strainsUsed = json.loads(open(os.path.join(data_dir, "strainsUsed.json")).read())
    genesUsed12 = json.loads(open(os.path.join(data_dir, "genesUsed12.json")).read())
    central_insert_bool_list = json.loads(open(
                            os.path.join(data_dir, "central_insert_bool_list.json")).read())
    t0_gN  = pd.read_table(os.path.join(data_dir, "t0_gN.tsv"), dtype={"locusId": str})
    t0tot  = pd.read_table(os.path.join(data_dir, "t0tot.tsv"), dtype={"locusId": str})
    all_fp  = os.path.join(data_dir, "all_df.tsv")
    exps_fp  = os.path.join(data_dir, "exps_df.tsv")
    genes_fp  = os.path.join(data_dir, "genes_df.tsv")
    all_df, exps_df, genes_df = get_all_exps_genes_df(all_fp, 
                                                    exps_fp, 
                                                    genes_fp)
    expsT0 = json.loads(open(os.path.join(data_dir, "expsT0.json")).read())

    return [genesUsed, strainsUsed, genesUsed12, central_insert_bool_list,
            t0_gN, t0tot, all_df, exps_df, genes_df, expsT0]








def getUniqueValuesFromDataFrame(dataframe, col_nm, drop_na=True):
    """
    Args:

        dataframe is pandas dataframe, 
        col_nm is string name of column
    Returns:
        Returns a pandas Series with only unique values
        Removes None values if they exist
    """
    current_column = dataframe[col_nm].replace('nan',None)
    if drop_na:
        current_column = current_column.dropna()
    x = current_column.unique()
    return x


def ShortSetNames(set_names_nparray, dbg_lvl=0):
    """ Using a table with rules, shorten the names of these sets
    Args:
        set_names_nparray (numpy.ndarray):  Array of string, unique set names from exps file
    Returns:
        set_names_nparray (numpy.ndarray): Edited set Names to be 
            in the format setX* or testX*

    This might convert 
        [ Keio_ML9_set2, Keio_ML9_set2, Keio_ML9_set2, ..., Keio_ML9_set3, Keio_ML9_set3,..., Keio_ML9_set3]
        to 
        [ set2, set2, set2, ..., set3, set3, ..., set3]
    """
    set_names_nparray = np.copy(set_names_nparray)

    # Below returns a TRUE/FALSE vector indicating which 
    # elements of the character vector contain a match (i.o.w a simple name)
    simple = [bool(re.search(r"(set|test)[0-9A-Z]+[0-9A-Z0-9]*$", x)) for x in set_names_nparray]

    if dbg_lvl > 0:
        if len(simple) > 0:
            logging.debug("simple names: \n" + ",".join(list([str(x) for x in simple])))
        else:
            logging.debug("No simple names found.")

   
    # We edit the values of set_names_nparray who are true for simple
    # by removing anything before 'set' or 'test'
    # We count the number of values that were false
    nleft = 0
    simple_set_names = []
    for i in range(len(simple)):
        if simple[i]:
            new_set_name = re.sub("^.*(set|test)", "\\1", set_names_nparray[i]) 
            set_names_nparray[i] = new_set_name
            simple_set_names.append(new_set_name)
        else:
            nleft += 1

    if dbg_lvl > 0:
        logging.debug("fixed set_names:\n" + ",".join(list(set_names_nparray)))
    
    candidates = []
    for x in "A.B.C.D.E.F.G.H.I.J.K.L.M.N.O.P.Q.R.S.T.U.V.W.X.Y.Z".split("."):
        candidates.append("set" + x)

    if dbg_lvl > 0:
        logging.debug(candidates)

    # get the elements in candidates that are not in set_names_nparray[simple]
    candidates = [x for x in candidates if x not in simple_set_names]
    if (nleft > len(candidates)):
        raise Exception(f"Too many unexpected set names: {nleft}.\n To fix this, contact developer "
                        "and say to change the number of possible extensions in list candidates (A.B...Z).")

    # Get the non-simple values from set_names_nparray
    oldComplex = [x for x in set_names_nparray if x not in simple_set_names]
    if dbg_lvl > 0:
        logging.debug("oldComplex:\n" + ",".join(oldComplex))

    cnd_ix = 0 
    translation_dict = {}
    for i in range(len(simple)):
        if not simple[i]:
            logging.info(f"Set {set_names_nparray[i]} simplified to {candidates[cnd_ix]}")
            translation_dict[set_names_nparray[i]] = candidates[cnd_ix]
            set_names_nparray[i] = candidates[cnd_ix]
            cnd_ix += 1

    
    crnt_unq = list(pd.Series(set_names_nparray).unique())
    repeats = []
    for x in list(set_names_nparray):
        if x in crnt_unq:
            crnt_unq.remove(x)
        else:
            repeats.append(x)

    if not (len(repeats) == 0):
        raise Exception("Non-unique set names! :\n" + \
                        ", ".join(repeats))
    else:
        logging.debug("Finished running short set names")
        if dbg_lvl > 0:
            logging.debug("Final set names list: " + ", ".join(set_names_nparray))

    return set_names_nparray, translation_dict


def get_all_exps_genes_df(all_fp, exps_fp, genes_fp):

    all_dtypes = {
            'barcode': str,
            'rcbarcode': str,
            'scaffold': str,
            'strand': str,
            'pos': int,
            'locusId': str,
            'f': float
            }
    all_df = pd.read_table(all_fp, dtype=all_dtypes) 
    
    #barcode	rcbarcode	scaffold	strand	pos	locusId	f
    exps_dtypes = {
            'SetName': str,
            'Index': str,
            'Date_pool_expt_started': str,
            "Description": str,
            "Group": str,
            "Drop": bool,
            "Condition_1": str,
            "Condition_2": str
    }
    exps_df = pd.read_table(exps_fp, dtype=exps_dtypes)

    genes_dtypes = {
            'locusId': str,
            'sysName': str,
            'type': int,
            'scaffoldId': str,
            'begin': int,
            'end': int,
            'strand': str,
            'name': str,
            'desc': str,
            'GC': float,
            'nTA': int
            }
    genes_df = pd.read_table(genes_fp, dtype=genes_dtypes)


    return [all_df, exps_df, genes_df] 


def export_or_import_genefitresults(genefitresults, typ, dir_path, dbg_print=False):
    """
    This function is mainly for debugging purposes ( results stored at 'tmp/ResultStorage')
    Args:
        typ (str): One of "exp" (export) or "imp" (import)
        genefitresults (d) or nothing: 
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
            # st_ix_nm - set index name
            new_st_ix_nm = f.split("py_")[1]
            if "_gene_fit" in new_st_ix_nm:
                new_st_ix_nm = new_st_ix_nm.split("_gene_fit")[0]
            elif "_strain_fit" in new_st_ix_nm:
                new_st_ix_nm = new_st_ix_nm.split("_strain_fit")[0]
            elif "_strain_se" in new_st_ix_nm:
                new_st_ix_nm = new_st_ix_nm.split("_strain_se")[0]
            setindexnames[new_st_ix_nm] = 1
        genefitresults = {}
        for setindexname, nan in setindexnames.items():
            if dbg_print:
                print(f"Importing results for {setindexname}")
            ret_d = {}
            ret_d['gene_fit'] = pd.read_table(
                                    os.path.join(dir_path, "py_" + setindexname + "_gene_fit.dftsv"),
                                        dtype=input_d, index_col=0)
            # These are pandas Series, so we need to take the column
            ret_d['strain_fit'] = pd.read_table(
                                os.path.join(dir_path, "py_" + setindexname + "_strain_fit.dftsv"),
                                index_col=0)['0']
            ret_d['strain_se'] = pd.read_table(
                                os.path.join(dir_path, "py_" + setindexname + "_strain_se.dftsv"),
                                index_col=0)['0']
            genefitresults[setindexname] = ret_d

        return genefitresults

    else:
        raise Exception(f"Cannot recognize type {typ}")

    if dbg_print:
        print("Finished Exporting results")

    return None

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




def stop(line_num):
    raise Exception(f"Stopped, line {line_num}") 

def main():
    args = sys.argv
    logging.basicConfig(level=logging.DEBUG)
    if args[-1] != "1":
        help_str = "Usage:\npython3 RunFEBA.py org_str data_dir FEBA_dir start_point 1"
        help_str += "\n start_point is an int which refers to where to start running the"
        help_str += " program. start_point = 1 is how it would be fully run."
        print(help_str)
        sys.exit(0)
    else:
        org_str = args[1]
        data_dir = args[2]
        FEBA_dir = args[3]
        start_point = args[4]
        RunFEBA(org_str, data_dir, FEBA_dir, int(start_point), debug_bool=True)



if __name__ == "__main__":
    main()



