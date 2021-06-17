import os, logging, json, re
import pandas as pd
import numpy as np
from translate_R_to_pandas import *


def data_prep_1(data_dir, FEBA_dir, debug_bool=False, meta_ix=7, cfg=None):
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
        meta_ix (int): The number of meta column indeces in all.poolcount
        cfg (python dict): The default and config variables required:
                            drop_exps (bool): Do we drop the 'Drop' experiments
                                              from the experiments dataframe
                                              already?
                            okControls (bool): Are we defining controls by
                                               the method where it's written
                                               into the Experiments file?
                                
                                            
                        

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
              data type is in each column. The dataframes we get are called:
                genes_df, all_df, exps_df, rules_df
            Within exps_df:
              We optionally remove the rows who have 'Drop' set to True (if drop_exps==True).
              We strip (remove the spaces from) the values in 'Group', 
              'Condition_1', 'Condition_2'
            We check that the right column names exist in each of the tables.
          
          checkLocusIdEquality:
            We check all the locusIds in all_df are also present in genes_df
            If debugging we also print the number of unique locusIds in each.
          check_exps_df_against_all_df:
            We check that the index names in all.poolcount are equivalent to the 
                'SetName' + '.' + 'Index' in exps
          prepare_set_names:  
            We replace the SetNames from their original version to a simplified standard one,
            remove the period in between SetName and Index in all.poolcount columns,
            and make the 'names' column in the experiments file and the all.poolcount columns
            have the same values. For example, we move column name from Keio_ML9_set2.IT004 to 
            set2IT004, and rename the values in the Experiments file similarly.
          get_special_lists:
            We get the lists from the files in data_dir if they are there,
            otherwise we return their values as empty lists. The lists we
            look for are genesUsed, which should be a list of locusIds
            from this genome that we are using, and ignore_list, which is a list
            of experiment names to ignore (columns from all.poolcount).
        If debug_bool is set to true we print out resultant exps, all, genes to 'tmp' dir
        We return the following variables: 
            'exps_df' (The experiments dataframe)
            'all_df' (The barcodes and locations dataframe)
            'genes_df' (The total genes dataframe)
            'genesUsed_list' (A python list of locusIds that we will use)
            'ignore_list' (A python list of experiment names to ignore)
          
    """

    genes_df, all_df, exps_df, rules_df = getDataFrames(data_dir, FEBA_dir, 
                                                        drop_exps=cfg['drop_exps'],
                                                        okControls = cfg['okControls'],
                                                        dbg_lvl=0)

    # Makes no changes to the variables
    checkLocusIdEquality(all_df, genes_df, debug_bool=debug_bool)

    # We check that SetNames and Indexes in experiments file match all.poolcount file
    check_exps_df_against_all_df(exps_df, all_df, meta_ix)

    # We make it so the names are cleaner and create 'names', 'num', 'short' in exps_df
    exps_df, all_df, replace_col_d = prepare_set_names(exps_df, all_df, rules_df, 
                                                        okControls=cfg['okControls'],
                                                        meta_ix=meta_ix,
                                                        debug_bool=debug_bool)

    genesUsed_list, ignore_list = get_special_lists(data_dir, all_df,
                                                    replace_col_d, debug_bool=debug_bool)

    if debug_bool:
        exps_df.to_csv("tmp/py_test1_exps_fp.tsv", sep="\t")
        all_df.to_csv("tmp/py_test1_all_fp.tsv", sep="\t")
        genes_df.to_csv("tmp/py_test1_genes_fp.tsv", sep="\t")

    return [exps_df, all_df, genes_df, genesUsed_list, ignore_list]


def getDataFrames(data_dir, FEBA_dir, drop_exps=False, 
                  okControls=False, dbg_lvl=0):
    """
    Args:
        data_dir: (str) Path to directory which contains the 
                    following files: 'all.poolcount', 'genes',
                            'exps' - all TSV files.
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
        exps_df (pandas DataFrame): Must contains columns:
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
        In exps_df:
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


    all_fp = os.path.join(data_dir, "all.poolcount")
    genes_fp = os.path.join(data_dir, "genes")
    exps_fp = os.path.join(data_dir, "exps")
    short_rules_fp = os.path.join(FEBA_dir, "desc_short_rules.tsv")

    # Checking access permissions
    for x in [all_fp, genes_fp, exps_fp]:
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
            "Condition_2": str,
            "control_group": str,
            "control_bool": str
    }
    exps_df = pd.read_table(exps_fp, dtype=exps_dtypes)

    # We update the 'Drop' experiments
    if 'Drop' in exps_df:
        new_drops = []
        for ix, value in exps_df['Drop'].items():
            if not isinstance(value, str):
                if pd.isna(value):
                    new_drops.append(False)
                else:
                    raise Exception(f"Value in 'Drop' not string: {value}")
            elif str(value).strip().upper() == "TRUE":
                new_drops.append(True)
            elif value.strip().upper() == "FALSE":
                new_drops.append(False)
            else:
                raise Exception(f"Cannot recognize Drop value in row {ix}:"
                                f" {value}")
        exps_df['Drop'] = new_drops
    else:
        exps_df['Drop'] = [False]*exps_df.shape[0]

    """
    if drop_exps:
        # Removing Drop rows
        exps_df.drop(remove_indeces, axis=0, inplace=True)
    """

    # Remove trailing spaces:
    for x in ["Group", "Condition_1", "Condition_2", "control_bool"]:
        if x in exps_df:
            # We take the entire column (pandas Series) and remove the spaces
            #   from either end
            exps_df[x] = exps_df[x].str.strip()


    rules_dtypes = {
            "V1": str,
            "V2": str
    }
    rules_df = pd.read_table(short_rules_fp, keep_default_na=False, dtype=rules_dtypes)

    # Checking genes.GC
    for x in ["scaffoldId", "locusId", "sysName", "desc", "begin", "end"]:
        if x not in genes_df.columns:
            raise Exception(f"Genes table must include header {x}")
    # Checking exps table
    for x in ["SetName", "Index", "Date_pool_expt_started", "Description"]:
        if x not in exps_df.columns:
            raise Exception(f"Experiments table must include header {x}")
    if okControls:
        for x in ["control_group", "control_bool"]:
            if x not in exps_df.columns:
                raise Exception("If okControls is set To True, then "
                                f"experiments table must include header {x}")

    # Checking all_df
    for x in ["scaffold", "locusId", "f", "pos"]:
        if x not in all_df.columns:
            raise Exception(f"All.PoolCount file must include header {x}")

    if dbg_lvl > 1:
        print(genes_df)
        print(all_df)
        print(exps_df)
        print(rules_df)


    return [genes_df, all_df, exps_df, rules_df]



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



def check_exps_df_against_all_df(exps_df, all_df, meta_ix):
    """
    We make sure that all the experiment names left in the all_df dataframe
        are the same as the experiment names in the rows of the experiments
        dataframe.
    """

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


    logging.debug("There are the same experiment names in all_df and exps_df.")




def prepare_set_names(exps_df, all_df, rules_df, 
                      okControls=False, meta_ix=7, debug_bool=False):
    """


    Description:
        We replace the SetNames from the complicated version to a simpler one,
        remove the period in between SetName and Index in all.poolcount columns,
        and make the 'names' column in the experiments file and the all.poolcount columns
        have the same values. For example, we move column name from Keio_ML9_set2.IT004 to 
        set2IT004, and rename the values in the Experiments file similarly.
        We also add multiple new columns to exps_df:
            "num", "short", "name", "t0set"
        We also make sure that any experiment with its "Group" being "Time0" has
        its short as "Time0" as well.
        We initialize the 't0set' column as being the date + the set name (lane).

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
        logging.info("short_names_srs: (shortSetNames[match_list])")
        logging.info(short_names_srs)
        logging.info("original set Names:")
        logging.info(exps_df['SetName'])
        logging.info('match_list')
        logging.info(match_list)
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
        logging.info('expNamesNew:')
        logging.info(expNamesNew)

    exps_df['num'] = range(1, exps_df.shape[0] + 1)
    # We replace certain strings with others using the 'rules' table.
    exps_df['short'] = applyRules(rules_df, list(exps_df['Description']))

    if okControls:
        if not "control_bool" in exps_df.columns:
            raise Exception("Using manual control label but no column "
                            "'control_bool' in Experiments file!")
        else:
            for ix, val in exps_df["control_bool"].iteritems():
                if val.strip().upper() == "TRUE":
                    exps_df["short"].loc[ix] = "Time0"
                else:
                    # Should not be a Time0 short
                    if exps_df["short"].loc[ix].upper() == "TIME0":
                        raise Exception("Description of experiment indicates Time0, but"
                                        f" value in control_bool is not 'True', instead '{val}'.")
    

    if debug_bool:
        logging.info("exps_df of col 'short':")
        logging.info(exps_df['short'])

    # We remove the "." in the names of the values. Just SetNameIndex now
    replace_col_d = {list(all_df.head())[meta_ix + i]: expNamesNew[i] for i in range(len(expNamesNew))}
    if debug_bool:
        logging.info('replace_col_d')
        logging.info(replace_col_d)
        logging.info('original all_df col names:')
        logging.info(list(all_df.columns))
    all_df = all_df.rename(columns=replace_col_d)
    if debug_bool:
        logging.info('after replacement all_df col names:')
        logging.info(list(all_df.columns))
    
    exps_df['name'] = expNamesNew



    # updating short to include Groups with Time0
    num_time_zero = 0
    for ix, val in exps_df['Group'].items():
        if val.strip().upper() == "TIME0":
            num_time_zero += 1
            exps_df.loc[ix, 'short'] = "Time0"

    # Updating column 't0sets' which refers to the date and SetName
    exps_df['t0set'] = [exps_df['Date_pool_expt_started'].iat[ix] + " " + \
                        val for ix, val in exps_df['SetName'].items()]

    if okControls:
        if not "control_group" in exps_df.columns:
            raise Exception("Using manual control label but no column "
                            "'control_group' in Experiments file!")
        else:
            for ix, val in exps_df["control_group"].iteritems():
                exps_df['t0set'].loc[ix] = val

    if debug_bool:
        logging.info('exps_df short: ')
        logging.info(exps_df['short'])
        logging.info('exps_df t0set: ')
        logging.info(exps_df['t0set'])
        logging.info(f"Total number of time zeros: {num_time_zero}")

    return exps_df, all_df, replace_col_d


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




def get_special_lists(data_dir, all_df, replace_col_d, debug_bool=False):
    """
    Args:
        replace_col_d: Dict mapping original all_df experiment name to replacement name
        data_dir

    Returns:
        genesUsed_list list<str>: LocusIds of genes to use
        ignore_list: List<str> New names for the experiments we want to ignore.

    Description: We get the lists from the files in data_dir if they are there.
                Otherwise we return their values as empty lists. The lists we
                look for are genesUsed, which should be a list of locusIds
                from this genome that we are using, and ignore_list, which is a list
                of experiment names to ignore (columns from all.poolcount)
    """

    genesUsed_list = []
    ignore_list = []
    # list of locusIds
    genesUsed_fp = os.path.join(data_dir, "strainusage.genes.json")
    # list of extra ignored experiments
    ignore_list_fp = os.path.join(data_dir, "ignore_list.json")

    if os.path.isfile(genesUsed_fp) and os.access(genesUsed_fp, os.R_OK):
        genesUsed_list = json.loads(open(GenesUsed_fp).read())
        logging.info(f"Loaded {len(genesUsed_list)} genes to include in the "
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


    return genesUsed_list, ignore_list


def applyRules(rules_df, desc_str_list):
    """
    We replace str value in V1 with value in V2
    Args:
        rules_df: data frame with cols:
                    V1,   V2
        desc_str_list: list<str>
    Returns:
        new_desc_list: list<str>
    """
    new_desc_list = []
    for j in range(len(desc_str_list)):
        new_desc_list.append(desc_str_list[j])
        for i in range(0, rules_df.shape[0]):
            new_desc_list[-1] = new_desc_list[-1].replace(rules_df["V1"].iloc[i], 
                                                        rules_df["V2"].iloc[i])
    return new_desc_list 
