import os, logging, json
from og_util import debug_print 
import pandas as pd
from translate_R_to_pandas import py_aggregate, py_table
import statistics

def data_prep_2(exps_df, all_df, genes_df,
                genesUsed_list,
                ignore_list, minSampleReads=2*10e4,
                meta_ix=7, 
                okDay=True, okLane=False,
                minGenesPerScaffold=10,
	        minT0Strain=3, minT0Gene=30,
                dbg_prnt=False,
                dbg_lvl=10,
                minLengthGenesUsed12=100,
                export_vars_bool=False
                ):
    """
    Args:
        ignore_list list<str>: List of shortened experiment names which we will 
                    not be using within the analysis.
        genesUsed_list (list<str>): List of locusIds we want to use
        okDay (bool): use Time0 from another day on the same lane
        okLane (bool):  compare to Time0 from another lane

    Description:
        We find which experiments aren't fit to be analyzed (maybe too few
            reads total).

    """
    # We find the indeces to ignore (info inside func) (ignore is list<str>)
    all_df, exps_df = set_up_ignore(ignore_list, all_df, 
                                            exps_df, minSampleReads,
                                            meta_ix=meta_ix, dbg_prnt=dbg_prnt)
    
    # central_insert_bool_list is a list of booleans
    central_insert_bool_list = get_central_insert_bool_list(all_df, dbg_prnt=dbg_prnt)

    # expsT0 is a dict that stores Dates and Set Names of Time0 experiments to their
    #       related experiment names. ( expsT0 could be set to {} since it's updated
    # in the next function entirely anyways).
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
    # But only has one column for each date, and all the experiments associated with
    # that date and are t0 sets are summed up.
    t0tot = create_t0tot(expsT0, all_df, dbg_prnt=True)

    # All the locusIds from all_df which include central insertions (many repeats) 
    indexBy = createIndexBy(all_df, central_insert_bool_list)


    t0_gN = createt0gN(t0tot, central_insert_bool_list, indexBy, debug_print_bool=True) 


    # strainsUsed will be a list of booleans with length being
    # total number of strains (num rows of all.poolcount)
    strainsUsed_list = createStrainsUsed(t0tot, minT0Strain, central_insert_bool_list)

    # This int below might be the size of the resulting tables 't' and 'fitness'
    nUniqueUsableLocusIds = getUniqueUsableLocusIds(all_df, strainsUsed_list)     

    genesUsed_list = getGenesUsedList(t0tot, strainsUsed_list, all_df, minT0Gene, genesUsed_list)

    # genesPerScaffold is a dict {scaffoldId (str): num different locusIds in that scaffoldId}
    genesPerScaffold = getGenesPerScaffold(genes_df, genesUsed_list)

    # smallScaffold and smallLocusIds are both list<str>
    smallScaffold, smallLocusIds = get_smallScaffold(genesPerScaffold, minGenesPerScaffold,
                                                     genes_df)

    # refining genesUsed_list 
    genesUsed_list = [x for x in genesUsed_list if x not in smallLocusIds]
    genesUsed_list =  remove_genes_if_not_in_genes_df(genesUsed_list, genes_df)


    print_info2(central_insert_bool_list, all_df, strainsUsed_list, genesUsed_list)

    # genesUsed_list12 is a list of locusIds that have t0tot sums with enough reads
    genesUsed_list12 = get_GenesUsed12(minT0Gene, strainsUsed_list, all_df,
                                  t0tot, minLengthGenesUsed12=minLengthGenesUsed12)

    logging.info(f"For cor12, using {len(genesUsed_list12)} genes. ");

    check_if_every_t0set_is_in_t0tot(exps_df, t0tot)




    return [[all_df, exps_df, genes_df, genesUsed_list], 
            [strainsUsed_list, genesUsed_list12, t0_gN, t0tot],
            [central_insert_bool_list, expsT0]]



def set_up_ignore(ignore, all_df, exps_df, minSampleReads, meta_ix=7, dbg_prnt=False):
    """ Setting up the index (columns of all.poolcount) names to avoid doing analysis
    Args:
        ignore: list of str with sample-index name to ignore (could have len 0)
        all_df: Data frame of all.poolcount
        exps_df: Data frame of experiments file
            Must contain cols: 'name', 'Drop'
            
        minSampleReads: int
        meta_ix: Start of where the indeces become sample/index names
    
    Returns:
        all_df, exps_df, ignore (list<str>, where str is name of indeces
                                we are ignoring)

    Description: 
        We update the experiments to ignore by performing the following tests:
        1. We take all the columns of experiments in all_df (ignoring metadata), 
            and take the sum over each column. We check, for each experiment,
            that the sum is greater than the value 'minSampleReads'.
        2. If the Drop column is True in exps_df then we ignore that column
        Then we remove the indeces from all_df (where they are column names) & 
            exps_df (where they are under the column 'name')
    """
    # Creating a list to ignore out of the all.poolcount indexes 
    #   (names are updated though?)
    if len(ignore) == 0: 
        logging.info("Length of ignore list is 0") 
        # metacol is ignored 
        # We select all the columns related to experiments
        # And get the sum over the columns
        tot = all_df.iloc[:,meta_ix:].sum(axis=0)
        # We figure out the columns for which the sum of barcodes
        # found is less than minSampleReads
        ignore = []
        for c in tot.keys():
            if tot[c] < minSampleReads:
                ignore.append(c)
                logging.info(f"Ignoring experiment name: {c}."
                             f"Sum of reads: {tot[c]}")

    # The 'Drop' column means if Drop=TRUE then ignore sets column
    for ix, val in exps_df['Drop'].items():
        if bool(val):
            if exps_df['name'][ix] not in ignore:
                ignore.append(exps_df['name'][ix])

    # updating the data frames
    if(len(ignore) > 0):
        print("Ignoring " + ", ".join(ignore))
        # List of booleans related to rows with values that aren't ignored
        exps_keep =  [(not (val in ignore)) for ix, val in exps_df['name'].items()]
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

        return [all_df, new_exps_df]
    else:
        print("Not ignoring any samples")

    return [all_df, exps_df]



def get_central_insert_bool_list(all_df, dbg_prnt=False):
    """
    Description:
        We look at the value 'f' for each barcode. 'f' is the percent
        within the gene that the transposon was inserted. For example,
        if a gene has length 900 base pairs, and the transposon was
        inserted at position 300, then 'f' would be .333.
        So if the value 'f' is between 0.1 and 0.9, then we keep that
        barcode (the value in central_insert_bool_list is True).
    """

    # this is a list of booleans over all rows of all_df if their f is 0.1<f<0.9
    central_insert_bool_list = [True if (0.1<=x<=0.9) else False for x in all_df['f']]

    num_central_insert_bool_list = central_insert_bool_list.count(True)

    if dbg_prnt:
        logging.info(f"{num_central_insert_bool_list} is the number of strains with central "
                      "insertions in the gene,\n"
                      "which is equivalent to the number of 'Trues' in central_insert_bool_list.")

    return central_insert_bool_list


def createExpsT0(exps_df, debug_print_bool=False):
    """
    Args: exps_df:
        data frame with cols:
            short (str): string explaining if Time0 or not
            t0set (str): is date + space + setName for ALL experiments in exps_df,
                not only just the t0sets

    Returns 
        expsT0: dict mapping t0set name 'date setName' - > list<set+Index (str (experiment name)) that's related>
            for every actual Time0 name

    Description:
        We create a dataframe which only holds experiments that are 'Time0' experiments
            i.e. 'Control' Experiments.
        Then we create a dict that stores the names of the experiments that are related to
            that time0 and return it.
        On any given day, there were a couple of experiments started that were the controls.
            We save the experiments that are related to any given day (and set)
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



def update_expsT0_and_exps_df_with_nont0sets(expsT0, exps_df, okLane, okDay,
                              print_bool=False, dbgp=False):
    """
    Args:
        expsT0: dict mapping t0set name 'date setName' - > list<set+Index (str) that's related>
            for every actual Time0 name
        exps_df: dataframe of exps file with additional col headers. Requires:
                    't0set', 'Date_pool_expt_started', 'SetName', 'short' 
                    for this function
        okLane: bool Assume True - we can use Time0 from another lane
        okDay: bool Assume True
        print_bool: to print all the vars


        nont0sets: list of exps_df 't0set' values that don't have 'Time0' as their 'short',
                   

    Returns:
        exps_df: (Updated t0set col to just be date instead of date + setname)
        expsT0: (Updated keys to just be date instead of date + setname) 
            updated values to be pandas Series with indeces


    Description:
        Gets a list of t0set values (date setname) which don't have 'Time0' as their short,
            and it iterates through them. 
        For each nont0set, we have to find a corresponding Time0 set to compare it to. If okDay
        is set to True, we choose a Time0 from the same SetName but a different day. If okLane
        is set to True, we choose a Time0 from another lane but the same day.
        We set the exps_df['t0set'] value of that experiment to the newly chosen Time0 date - 
            which, points to a list of experiments that are associated with that Time0 in 
            expsT0
        
    """

    if dbgp:
        print("A1 Original exps_df t0set:")
        print(exps_df['t0set'])
        print("A1 Original expsT0:")
        print(expsT0)

    # nont0sets is a list of str date + setname
    nont0sets = get_nont0_sets(exps_df, debug_print_bool=True)

    if print_bool:
        with open("tmp/py_nont0sets.json", "w") as g:
            g.write(json.dumps(nont0sets, indent=2))

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
        if len(date_list) == 0:
            raise Exception(f"No date associated with nont0set date+setname value '{datesetname}'")
        else:
            associated_date = date_list[0]

        if print_bool:
            debug_print(associated_date, "associated_date")

        # unique set names over current datesetname 
        unique_applicable_set_names = list(exps_df[u]['SetName'].unique())
        if len(unique_applicable_set_names) > 0:
            associated_setname = unique_applicable_set_names[0]
        else:
            raise Exception("No SetName associated with date setname value: {datesetname}")

        # Day
        t0_date_experiments = exps_df[exps_df['Date_pool_expt_started'] == associated_date][exps_df['short'].str.upper() == "TIME0"]
        # Lane (SetName)
        t0_setName_experiments = exps_df[exps_df['SetName'] == associated_setname][exps_df['short'].str.upper() == "TIME0"]

        if okLane and t0_date_experiments.shape[0] > 0:
            if datesetname in expsT0:
                del expsT0[datesetname]
            logging.info(f"Using Time0 from other lanes instead for {datesetname}")
            logging.info("Experiments affected:\n" + "\n".join(list(exps_df['name'][u])))
            for ix in range(len(u)):
                if u.iat[ix]:
                    exps_df['t0set'].iat[ix] = associated_date
            expsT0[associated_date] = list(exps_df['name'][exps_df['Date_pool_expt_started'] == associated_date][exps_df['short'].str.upper() == "TIME0"])
        elif (okDay and t0_setName_experiments.shape[0] > 0 ):
            if datesetname in expsT0:
                del expsT0[datesetname]
            newt0sets = t0_setName_experiments['t0set']
            # Arbitrarily choosing the first one
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
        unique_nont0sets list<str>: list of exps_df t0set values that don't have Time0 as their short,
    Description:
        Get all experiment's t0set strings (Date + set) that don't have 'Time0' as their short. In other
            words, get all sets that aren't time0's.
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

    Description:
        
        
    """

    # We prepare to sum the values for all the pertinent setname-indexes for each datesetname
    # in expsT0.keys
    t0tot = {} #{date: pd_series([sum1, sum2, ...]) for date in expsT0.keys()}
    for date, exp_list in expsT0.items():
        
        t0tot[date] = all_df[exp_list].sum(axis=1)

    # We recreate t0tot as a DataFrame
    t0tot = pd.DataFrame.from_dict(t0tot)

    if dbg_prnt:
        t0tot.to_csv("tmp/py_t0tot.tsv", sep= "\t")

    return t0tot


def createIndexBy(all_df, central_insert_bool_list, print_bool=False):
    """
    indexBy is a panda Series of all the locusIds which
        have insertions in the important regions (keeps indexes)
    Args:
        all_df: Dataframe of all.poolcount
        central_insert_bool_list: A pandas series of booleans the length 
                   of all_df which marks which strains have
                   insertions in the central 80% of a gene
    Returns:
        indexBy: panda Series with all the locusIds which
            have insertions in the important regions
            it's length should be the same length as the
            number of Trues in central_insert_bool_list - comes from
            all_df. Note- locusIds are NOT unique.
    """

    # All the locusIds which include insertions in the important regions
    indexBy = all_df['locusId'][central_insert_bool_list]
    if print_bool:
        debug_print(indexBy, 'indexBy')

    return indexBy


def stop(line_num):
    raise Exception(f"Stopped, line {line_num}") 


def createt0gN(t0tot, central_insert_bool_list, indexBy, debug_print_bool=False):
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
        central_insert_bool_list: A pandas series of booleans the length 
                   of all_df which marks which strains have
                   insertions in the central 80% of a gene
        indexBy: panda Series with all the locusIds which
            have insertions in the important regions
            it's length should be the same length as the
            number of Trues in central_insert_bool_list - locusIds are not unique 
    Returns:
        t0gN:
            A dataframe with the same number of columns
            as t0tot + 1 for locusIds. Row number depends on the 
            number of unique locusIds in indexBy as well as 
            the genes with central insertions.
            It's length should be the same length as the number of 
            unique locusIds
    Description:
        We get a dataframe which sums the time0 dates
        over the places where the locusId is the same
        and only keeps those insertions that are central.
        The number of rows in this is the number of unique 
        locusIds which had a central insertion in them.
        The values are sums over those same parameters.
    """

    t0_gN = t0tot[central_insert_bool_list]
    t0_gN['locusId'] = indexBy
    
    t0_gN = t0_gN.groupby(["locusId"], as_index=False).sum()

    if debug_print_bool: 
        t0_gN.to_csv("tmp/py_t0_gN.tsv", index=False, sep="\t")


    print_log_info1(t0_gN)

    return t0_gN


def print_log_info1(t0_gN):
    """
    Description:
        We print out the number of central reads per t0 set
            in millions.
    """

    logging.info("Central Reads per t0set:\n")
    # We iterate over the set names
    setnames = list(t0_gN.keys())
    setnames.remove('locusId')
    for k in setnames:
        try:
            logging.info(f"{k}: {t0_gN[k].sum()}")
        except Exception:
            logging.info(f"Couldn't print value for key {k}")


def createStrainsUsed(t0tot, minT0Strain, central_insert_bool_list):
    """ Create the variable strainsUsed - uses existing var if not None

    Args:
        t0tot: A Dataframe which contains datesetname: [sum1, sum2, 
                    ...] for datesetname in expsT0.keys(),
                e.g. A dataframe with timezeros datesetnames
                The number of rows in the data frame is equal
                to the number of rows in all_df
                Does not contain cols besides datesetnames
        minT0Strain: int, minimum mean value for total number of
                    barcodes read for a sample name.
        central_insert_bool_list: A pandas series of booleans the length 
                       of all_df which marks which strains have
                       insertions in the central 80% of a gene
        strainsUsed: either list of booleans or None
    Returns:
        strainsUsed: list of boolean the length of total number of strains in all_df
    Description:
        We make strainsUsed a list which contains True or False values for 
          each strain in all_df such that both the strain has an insertion
          centrally in a gene (meaning .1<f<.9) AND that the average number 
          of insertions over the t0 totals is greater than the integer minT0Strain.
    """


    # strainsUsed will be a list of booleans with length being
    # total number of strains.
    strainsUsed = []
    for i in range(len(central_insert_bool_list)):
        if central_insert_bool_list[i] and t0tot.iloc[i,:].mean() >= minT0Strain:
            strainsUsed.append(True)
        else:
            strainsUsed.append(False)

    return strainsUsed


def getUniqueUsableLocusIds(all_df, strainsUsed):
    """
    Description:
        We get the unique locus Ids where we can use the strain
    """


    unique_usable_locusIds = all_df['locusId'][strainsUsed].unique()
    num_unique_usable_locusIds = len(unique_usable_locusIds)
    if num_unique_usable_locusIds < 10:
        raise Exception("Less than ten usable locusIds, program designed to stop.")
    else:
        logging.info(f"Unique number of usable locusIds: {num_unique_usable_locusIds}")
    return num_unique_usable_locusIds



def getGenesUsedList(t0tot, strainsUsed, all_df, minT0Gene, genesUsed_list,
                 debug_print_bool=False):
    """ We create the variable genesUsed_list
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
        all_df (pandas DataFrame): Uses col locusId
        minT0Gene: (int) 
        genesUsed_list: list of locusIds to be used (could be empty)
    Returns:
        genesUsed_list: list of unique locusIds such that their mean Time0 values
                    is greater than minT0Gene

    Description:
        We only take the t0 totals over the used strains.
        Then we sum up the t0 totals over the locusIds.
        Then we take the means of the entire column

    """

    # genesUsed_list is a potentially empty list of locusIds to be used
    pre_t0_gn_used = t0tot[strainsUsed]
    pre_t0_gn_used['locusId'] = list(all_df['locusId'][strainsUsed])

    if len(genesUsed_list)==0:
        # t0_gN_used is  
        t0_gN_used = py_aggregate(pre_t0_gn_used, 
                                  'locusId',
                                  func='sum'
                                 )
        if debug_print_bool:
            t0_gN_used.to_csv("tmp/py_t0_gN_used.tsv", index=False, sep="\t")
        # n0 is a pandas series with a mean for each rows in t0_gN_used
        n0 = t0_gN_used.iloc[:,t0_gN_used.columns != 'locusId'].mean(axis=1)
        # Below we take the mean over the whole n0
        logging.info(f"Time0 reads per gene: mean {statistics.mean(n0)}"
                     f"median: {statistics.median(n0)} "
                     f" ratio: {statistics.mean(n0)/statistics.median(n0)}")

        # Below is boolean list of locations where the row mean passes minT0Gene
        genesUsedpre = [(n0.iloc[i] >= minT0Gene) for i in range(n0.shape[0])]
        #print(genesUsedpre[:100])
        genesUsed_list = t0_gN_used['locusId'][genesUsedpre]
        if debug_print_bool:
            genesUsed_list.to_csv("tmp/py_genesUsed.tsv", sep="\t")


    return genesUsed_list



def getGenesPerScaffold(genes_df, genesUsed):
    """
    Args:
        genes_df: Dataframe of genes.GC
        genesUsed: list<locusId (str)>
    Returns:
        genesPerScaffold (python dict):
            genesPerScaffold is a dict with scaffoldId (str) -> number of locusIds from genesUsed
                                                                found in that scaffold.
    Description:
        We get a python dictionary with scaffoldIds pointing to the number of genes 
          in that scaffoldId in the genes_df.
    """

    #We iterate over every row of genes_df and find locations of genesUsed locusIds
    rows_with_locus_Ids_in_genesUsed_bool = [genes_df['locusId'].iat[i] in genesUsed \
                                    for i in range(len(genes_df['locusId']))]

    genesPerScaffold = py_table(list(genes_df['scaffoldId'][rows_with_locus_Ids_in_genesUsed_bool]
                                    ))

    return genesPerScaffold


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
    Description:
        We get all scaffoldIds who have less than the minimum number of locusIds in them.
        We also get all the locusIds in those scaffoldIds.
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

    for x in genes_not_in_genes_df:
        genesUsed_list.remove(x)


    if len(genesUsed_list) < 10 or (not genes_in_genes_df_bool):
        logging.info("genesUsed_list")
        logging.info(genesUsed_list)
        raise Exception(f"Less than 10 genes left, exiting program: {len(genesUsed_list)}")
    
    if len(genes_not_in_genes_df) > 0:
        logging.critical("Gene Locus Ids not in the genes.GC file: \n"
                        ", ".join(genes_not_in_genes_df) + "\n")

    return genesUsed_list 


def print_info2(central_insert_bool_list, all_df, strainsUsed, genesUsed):
    """
    Args:
        central_insert_bool_list: list<bool>
        all_df: DataFrame of all.poolcount
        strainsUsed: list<bool>
        genesUsed: list<locusId (str)>
    Description:
        We print out
    """
    
    # We count the number of Trues in central_insert_bool_list
    num_true_central_insert_bool_list = central_insert_bool_list.count(True)

    num_unique_locus_Ids = len(all_df['locusId'][central_insert_bool_list].unique())

    logging.info(f"Using {str(len(strainsUsed))} of {num_true_central_insert_bool_list} genic strains.")
    logging.info(f"Using {len(genesUsed)} of {num_unique_locus_Ids} genes with data.")

    return None

def get_GenesUsed12(minT0Gene, strainsUsed, all_df,
                    t0tot, minLengthGenesUsed12=100):
    """
    We get the locusIds which have insertions both under 0.5 and over
        0.5 within the gene (percentage of length) and with values
        over the minT0Gene
    Args:
        minT0Gene: int
        strainsUsed: list<bool> Length of all_df
        all_df: Dataframe needs col (f)
        t0tot: data frame where column names are 'date setname'
                and linked to a list of sums over the indexes that relate
                to that setname, with the list length being equal to the
                total number of strains (barcodes) in all.poolcount 
                (total number of rows is same as all.poolcount)
    Returns:
        genesUsed12: list of locusIds that have both high f (>0.5) and low f (<0.5)
                    insertions with enough abundance of insertions on both sides,
                    where the abundance is coming from the t0tot dataframe

    """

    minT0GeneSide = minT0Gene/2

    # d1 captures t0tot whose strains have f < 0.5 and True in strainsUsed
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
    if (len(genesUsed12) < minLengthGenesUsed12):
        raise Exception(
                f"Length of genesUsed12 is less than {minLengthGenesUsed12}."
                f" Value: {len(genesUsed12)}"
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
        crt (pandas DataFrame): A dataframe (from t0tot) with the locusId only holding 
                                unique values and the value for every other column is 
                                the sum over where the locusId used to be the same.
        crt_row_min_bool list<bool>: A boolean for each row of the aggregated 
                                     dataFrame values where the value is True 
                                     if the minimum value in that row
                                     is greater than the minimum T0 value needed
                        
    """
    crtt0tot = t0tot[stUsed_and_good_f]
    crtt0tot['locusId'] = all_df['locusId'][stUsed_and_good_f]
    # crt is a dataframe with unique locusIds and summed up columns for the rest of the values
    crt = py_aggregate(crtt0tot,
                      'locusId',
                      'sum')

    # Get all columns and rows besides locusId and take their minimum
    # Returns a pandas series with minimum of each row 
    crt_mins = crt.loc[:, crt.columns != 'locusId'].min(axis=1)
    #print(crt_mins)
    crt_row_min_bool = [bool(x >= minT0GeneSide) for x in list(crt_mins)]

    return crt, crt_row_min_bool


def check_if_every_t0set_is_in_t0tot(exps_df, t0tot):
    """
    Args:
        exps_df:
            Dataframe of FEBABarSeq.tsv
        t0tot: data frame where column names are 'date'
                and linked to a list of sums over the indexes that relate
                to that setname, with the list length being equal to the
                total number of strains (barcodes) in all.poolcount
    """

    # We check if every t0set is in t0tot
    #{datesetname:[] for datesetname in expsT0.keys()}
    incorrect_sets = []
    for t0set in exps_df['t0set'].array:
        if t0set not in t0tot.head():
            incorrect_sets.append(t0set)

    if len(incorrect_sets) > 0:
        raise Exception("incorrect t0sets: \n" + ", ".join(incorrect_sets))



