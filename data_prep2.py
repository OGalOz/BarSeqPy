import os, logging, json
from og_util import debug_print 
import pandas as pd
from translate_R_to_pandas import py_aggregate, py_table
import statistics

def data_prep_2(exps_df, all_df, genes_df,
                genesUsed_list,
                ignore_list, 
                meta_ix=7, 
                dbg_prnt=False,
                dbg_lvl=10,
                export_vars_bool=False,
                cfg=None):
    """
    Args:
        exps_df:
        all_df:
        genes_df:
        ignore_list list<str>: List of shortened experiment names which we will 
                    not be using within the analysis.
        genesUsed_list (list<str>): List of locusIds we want to use
        cfg:
            minSampleReads (int): What is the minimum number of reads for an 
                                 experiment to be used in the analysis?
            minGenesPerScaffold (int): What is the minimum number of genes
                                        in a scaffold for the scaffold to
                                        be used in the analysis?
            minT0Strain (int): What is the minimum mean per strain
                                from the T0 (control) experiments? For example,
                                there are 4 T0 experiments; we go through
                                each strain and take the mean over those
                                4 T0 experiments. If the mean is less
                                than minT0Strain, then we don't use that
                                strain, otherwise we keep it.
            minT0Gene (int): What is the minimum MEAN of a specific
                             gene over all the Time0s (aggregated for
                             locusIds). In other words, suppose there are 3
                             control (Time0) groups. Each of those 3 control
                             groups have two experiments that contribute to
                             them. We take the sum of the 2 experiments for 
                             each control group and create a dataframe with
                             as many rows as nStrainsUsed and with 3 columns,
                             one for each strain. Then we take the sum of those 
                             2-experiment sums over the locusIds, so now we have 
                             a dataframe whose number of rows is the number of 
                             unique locusIds. Then we take the average of
                             those 3 controls per locusId, and check if
                             that is greater than minT0Gene. If it is,
                             then we keep that locusId, otherwise, we 
                             don't use that locusId.
            minLengthGenesUsed12 (int): The minimum total number of genes
                            that have enough abundance of insertions on
                            both sides.
            okDay (bool): use Time0 from another day on the same lane
            okLane (bool):  compare to Time0 from another lane

    Description:
        We first get the configuration variables out of the configuration
        dict if they are there, otherwise we make them the default values.
        Then we run the following functions:
        set_up_ignore:
            We update the experiments to ignore by performing the following tests:
            1. We take all the columns of experiments in all_df (ignoring metadata columns), 
                and take the sum over each column. We check, for each experiment,
                that the sum is greater than the value 'minSampleReads'. In other words,
                we take the sum of all the reads over all the strains for each
                experiment and check if there were enough of them, enough of them
                meaning the number is greater than the number in minSampleReads.
            2. If the Drop column is True in exps_df then we ignore that column.
                (For each row in the experiments file, we have a column called 
                'Drop', and if it is indicated to be True, then we ignore that
                row - we get the name of the experiment to ignore because on
                that row there is also the column 'name', which indicates
                the name of the experiment correlated with all_df). In data_prep1
                we just prepare the Drop Column to contain boolean values.
            3. For each experiment name we choose to ignore, we remove the 
                column from all_df (where they are column names) & the row from
                exps_df (where the name is under the column 'name')
            Note that we update BOTH 'exps_df' and 'all_df' here.
        get_central_insert_bool_list:
            We look through all the rows of all_df, each of which represents a
            strain as a barcode, and information related to that barcode,
            like insertion location and amount of times it appears per experiment.
            We look at the value 'f' for each row. 'f' is the fraction of location
            (from 0 to 1) within the gene that the transposon was inserted. 
            For example, if a gene has length 900 base pairs, and the 
            transposon was inserted at position 300, then 'f' would be .333.
            So if the value 'f' is between 0.1 and 0.9, then we keep that
            barcode (the value in central_insert_bool_list is True).
            We return a list the length of all_df (nAllStrains).
            The list is called 'central_insert_bool_list'.
        createExpsT0:
            The overall function returns a dict. First,
            we create a dataframe out of exps_df which only holds experiments (rows)
            that have their 'short' column as 'Time0', i.e. 'Control' Experiments.
            Then we take the 't0set' or 'control_group' name of those experiments and
            create a dict which maps the control_group -> list of experiments in that control
            group which are actually control experiments. Note, we do not include any
            non-control experiments in this dict.
            We return this dict, which is called 'expsT0'.
        create_t0tot:
            We create the data frame t0tot.
            First we take expsT0, which is a python dict
            which maps T0 group to list of experiment 
            names which belong to it (but only the controls,
            the true time0s, not any experiment to be
            compared to it). Then for each T0 group,
            we sum the experiments related to it 
            over all the reads. So we end up with a 
            dataframe that contains as many columns
            as T0 groups, and the number of rows in that
            column is as many as in all_df.
            We return this data frame called 't0tot'.
        createt0gN:
            We get a dataframe (t0_gN) which sums the time0 names
            over the places where the locusId is the same
            and only keeps those insertions that are central.
            (Aggregate t0tot over locusId)
            The number of rows in this is the number of unique 
            locusIds which had a central insertion in them.
            The values are aggregate sums over those same parameters.
            The column names are the same as t0tot, plus the column
            locusId.
            We return the dataframe 't0_gN'.
        createStrainsUsed:
            We make strainsUsed a list which contains True or False values for 
            each strain in all_df such that both the strain has an insertion
            centrally in a gene (meaning .1<f<.9) AND that the mean 
            of insertions over the t0 totals (t0tot) is greater than the 
            integer minT0Strain.
            We return the variable named 'strainsUsed_list'
        getGenesUsedList:
            We take t0_gN, which is the time0 totals summed over locusIds, 
            and we take the mean for each row over the Time0 reads.
            So now we have a series with row number = unique LocusIds,
            and values are the mean of the Time0 reads over that locusId.
            There are no longer as many columns as there are Time0 groups,
            (now there is only one column).
            Then we filter out the locusIds where the mean over the Time0
            reads is less than the integer threshold 'minT0Gene'.
            We store these initial locusIds (strings) as genesUsed list.
            Then we filter out genes that belong to scaffolds which
            have too few total genes on them. In other words, if a scaffold 
            has fewer genes on it than the integer 'minGenesPerScaffold', then we 
            won't use those genes in the analysis.
            Then we check that all the locusIds in the current genesUsed list
            are also in the genes_df (dataframe from genes.GC)
            We return this list called 'genesUsed_list'.
        get_GenesUsed12:
            We get the locusIds which have enough insertions both under 0.5 and over
            0.5 within the gene (percentage of length), where enough means values
            over minT0Gene/2. Then we also make sure all those genes are also
            in our original genesUsed_list, which have other thresholds, like
            belonging to large enough scaffolds.
            If the total number of remaining locusIds
            is less than minLengthGenesUsed12, then we raise an Exception.
            We return this list called 'genesUsed_list12'
        check_that_every_t0set_is_in_t0tot:
           We make sure every t0set value in the exps_df column 't0set'
           is also a column name in t0tot.

        Then we update strainsUsed_list to only include strains that 
        were inserted in genes that are included in genesUsed_list.
        We also create a temporary variable strainsUsed_list12 that 
        only contains strains that were inserted in genes that are 
        in genesUsed_list12 (good insertions for both halves).
        Then we create all the important logging integers generated
        during this phase of the analysis:
            nAllStrains = number of rows ( all.poolcount )
            nAllStrainsCentral = number of rows in all.poolcount with 0.1<f<0.9
            nAllStrainsCentralGoodGenes = number of rows in all.poolcount with 0.1<f<0.9
                                          AND all locusIds are in the list 'GenesUsed'
                                            (27659 in Keio) - also known as nUsefulReads
                                            This is the same as nStrainsUsed
            nAllStrainsCentralGoodGenes12 = number of rows in all.poolcount with 0.1<f<0.9
                                          AND all locusIds are in the list 'GenesUsed12'
                                            (27659 in Keio) - also known as nUsefulReads
            nStrainsUsed  = nAllStrainsCentralGoodGenes, just another name.
            nTotalGenes = number of rows (genes.GC)
            nGenesUsed = (len(genesUsed)) number of rows in genes.GC that we actually use
                        which is equivalent to the number of unique genes
                        that have good insertions in them. (1355 in Keio)
                        Which is the same as the output fitness and t score dataframes
            nGenesUsed12 = (len(genesUsed12)) number of locusIds with a good amount of 
                            insertions in both halves of 'f' from all_df. Both df1 and df2 
                            in GeneFitness() have this number of rows.
            nExperiments = number of rows in experiments file

        We print these out to the console, and store them in a dict called 'num_vars_d'.
        Finally we return the variables that are used in the future:
            all_df, exps_df, genes_df, genesUsed_list, 
            strainsUsed_list_new, genesUsed_list12, t0_gN, t0tot, expsT0
            And num_vars_d for debugging.
            
    """

    # Preparing config variables:
    if cfg is not None:
        minSampleReads= cfg["minSampleReads"] 
        minGenesPerScaffold = cfg["minGenesPerScaffold"]
        minT0Strain = cfg["minT0Strain"]
        minT0Gene = cfg["minT0Gene"]
        minGenesAllowed = cfg['minGenesAllowed']
        minLengthGenesUsed12 = cfg["minLengthGenesUsed12"]
        okControls = cfg["okControls"]
        okDay = cfg["okDay"]
        okLane = cfg["okLane"]
    else:
        minSampleReads = 2*10e4
        minGenesPerScaffold = 10
        minT0Strain = 3 
        minT0Gene = 30
        minGenesAllowed = 100
        minLengthGenesUsed12 = 100
        okControls = False
        okDay = True
        okLane = False



    # We find the indeces to ignore (info inside func) (ignore is list<str>)
    # Note that in this function we change all_df and exps_df
    all_df, exps_df = set_up_ignore(ignore_list, all_df, 
                                    exps_df, minSampleReads,
                                    meta_ix=meta_ix, dbg_prnt=dbg_prnt)

    
    # central_insert_bool_list is a list of booleans
    central_insert_bool_list = get_central_insert_bool_list(all_df, dbg_prnt=dbg_prnt)



    #if okControls:
    #    expsT0, exps_df = UseControlGroupsToGetExpsDfAndExpsT0(exps_df)
    #else:
        
    # expsT0 is a dict that stores Dates and Set Names of Time0 experiments to their
    #       related experiment names. ( expsT0 could be set to {} since it's updated
    # in the next function entirely anyways).
    expsT0 = createExpsT0(exps_df)
    if dbg_lvl>2:
        with open("tmp/py_expsT0.json", "w") as g:
            g.write(json.dumps(expsT0, indent=2))

    print(exps_df['t0set'])
    if not okControls:
        expsT0, exps_df = update_expsT0_and_exps_df_with_nont0sets(expsT0, 
                                    exps_df, okLane, okDay,
                                    okControls,
                                    print_bool=False,
                                    dbgp=False)


    t0tot = create_t0tot(expsT0, all_df, dbg_prnt=False)


    # All the locusIds from all_df which include central insertions (many repeats) 
    indexBy = all_df['locusId'][central_insert_bool_list]

    # t0_gN is the sums over the locusIds of the Time0s (control_groups)
    t0_gN = createt0gN(t0tot, central_insert_bool_list, indexBy, debug_print_bool=False) 


    # strainsUsed will be a list of booleans with length being
    # total number of strains (num rows of all.poolcount)
    strainsUsed_list = createStrainsUsed(t0tot, minT0Strain, central_insert_bool_list)

    # This int below might be the size of the resulting tables 't' and 'fitness'
    #nUniqueUsableLocusIds = getNumUniqueUsableLocusIds(all_df, strainsUsed_list)     


    if len(genesUsed_list) == 0:
        genesUsed_list = getGenesUsedList(t0_gN, strainsUsed_list, all_df, minT0Gene,
                                          genes_df, minGenesPerScaffold,
                                          minGenesAllowed,)
        nUniqueUsableLocusIds = len(genesUsed_list)



    print_info2(central_insert_bool_list, all_df, strainsUsed_list, genesUsed_list)

    # genesUsed_list12 is a list of locusIds that have t0tot sums with enough reads
    genesUsed_list12 = get_GenesUsed12(minT0Gene, strainsUsed_list, all_df,
                                  t0tot, genesUsed_list,
                                  minLengthGenesUsed12=minLengthGenesUsed12)

    logging.info(f"For cor12, using {len(genesUsed_list12)} genes. ");

    check_that_every_t0set_is_in_t0tot(exps_df, t0tot)

    # We update strainsUsed_list to only include strains that were inserted
    # in genes that are used (in genesUsed_list)
    all_df_locusIds = all_df['locusId']
    strainsUsed_list_new = []
    for i in range(len(strainsUsed_list)):
        strainsUsed_list_new.append(bool(strainsUsed_list[i] and \
                                        (all_df_locusIds.iloc[i] in genesUsed_list)))
    strainsUsed_list = strainsUsed_list_new

    strainsUsed_list12 = []
    for i in range(len(strainsUsed_list)):
        strainsUsed_list12.append(bool(strainsUsed_list[i] and \
                                        (all_df_locusIds.iloc[i] in genesUsed_list12)))

    # Important numerical variables:
    num_vars_d = {
            "nAllStrains": all_df.shape[0],
            "nAllStrainsCentral": central_insert_bool_list.count(True),
            "nAllStrainsCentralGoodGenes": strainsUsed_list.count(True),
            "nAllStrainsCentralGoodGenes12": strainsUsed_list12.count(True),
            "nStrainsUsed": strainsUsed_list.count(True),
            "nTotalGenes": genes_df.shape[0],
            "nGenesUsed": len(genesUsed_list),
            "nGenesUsed12": len(genesUsed_list12),
            "nExperiments": exps_df.shape[0],
            "nSetIndexToRun": len(all_df.columns[meta_ix:])
    }

    print(num_vars_d)


    return [[all_df, exps_df, genes_df, genesUsed_list], 
            [strainsUsed_list, genesUsed_list12, t0_gN, t0tot],
            [expsT0, num_vars_d]]



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
            1. We take all the columns of experiments in all_df (ignoring metadata columns), 
                and take the sum over each column. We check, for each experiment,
                that the sum is greater than the value 'minSampleReads'. In other words,
                we take the sum of all the reads over all the strains for each
                experiment and check if there were enough of them, enough of them
                meaning the number is greater than the number in minSampleReads.
            2. If the Drop column is True in exps_df then we ignore that column.
                For each row in the experiments file, we have a column called 
                'Drop', and if it is indicated to be True, then we ignore that
                row. Within the row, there is also the column 'name', which indicates
                the name of the experiment to ignore.
            3. For each experiment name we choose to ignore, we remove the 
                column from all_df (where they are column names) & the row from
                exps_df (where the name is under the column 'name')
            Note that we update BOTH exps_df and all_df here.

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
        We return a list the length of all_df (nAllStrains) 
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
        We create a dataframe out of exps_df which only holds experiments 
            that have their 'short' column as 'Time0', i.e. 'Control' Experiments.
        Then we take the 't0set' or 'control_group' name of those experiments and
        create a dict which maps the control_group -> list of experiments in that control
        group which are actually control experiments. Note, we do not include any
        non-control experiments in this dict.
        We return this dict which is called 'expsT0'
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



def update_expsT0_and_exps_df_with_nont0sets(expsT0, exps_df, 
                                            okLane, okDay, okControls,
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
        okControls: We get Time0 info from the Experiments dataframe (manually
                    written).
        print_bool: to print all the vars


        nont0sets: list of exps_df 't0set' values that don't have 'Time0' as their 'short',
                   

    Returns:
        exps_df: (Updated t0set col to just be date instead of date + setname)
        expsT0: (Updated keys to just be date instead of date + setname) 
            updated values to be pandas Series with indeces


    Description:
        This only occurs if we are not using okControls.
        First, we use get_nont0_sets to get the control group names ('t0set') for non controls.
        get_nont0_sets:
            Get all experiment's t0set (control group) strings that don't have 'Time0' as their short.
            In other words, get all control_group names for experiments that aren't controls. 
            Hopefully, every single one of these is associated with an existing control group,
            whose name is found in the dict expsT0.
            Gets a list of t0set values (date setname) which don't have 'Time0' as their short,
                and it iterates through them. We essentially get the control_group names
                for all the non-control experiments. For every experiment, we have to find
                an existing control_group to compare it to if it doesn't exist yet.
        Then we update the 't0set' column for exps_df with corrected t0set names 
        (control groups) if they aren't in good form. The idea is, 

        okControls hasn't been completed yet.

        Otherwise:
        If okDay is set to True, we choose a Time0 from the same SetName 
        but a different day. If okLane is set to True, we choose a Time0 
        from another lane but the same day.
        We set the exps_df['t0set'] value of that experiment to the 
            newly chosen Time0 date - which points to a list of 
            experiments that are associated with that Time0 in expsT0
        
    """

    if dbgp:
        print("A1 Original exps_df t0set:")
        print(exps_df['t0set'])
        print("A1 Original expsT0:")
        print(expsT0)

    # nont0sets is a list of str date + setname
    nont0sets = get_nont0_sets(exps_df, okControls, debug_print_bool=True)

    if print_bool:
        with open("tmp/py_nont0sets.json", "w") as g:
            g.write(json.dumps(nont0sets, indent=2))

    for t0setname in nont0sets:
        # Each t0setname is '{date} {setName}'
        if dbgp:
            print(f"Current t0setname: {t0setname}")

        # u is a list of bools that matches t0setnames to label where t0set is this one.
        u = exps_df['t0set'] == t0setname

        # This should be a list of length 1
        date_list = list(exps_df[u]['Date_pool_expt_started'].unique())
        if len(date_list) == 0:
            raise Exception(f"No date associated with nont0set date+setname value '{t0setname}'")
        else:
            associated_date = date_list[0]


        # unique set names over current t0setname 
        set_names_list = list(exps_df[u]['SetName'].unique())
        if len(set_names_list) > 0:
            associated_setname = set_names_list[0]
        else:
            raise Exception("No SetName associated with date setname value: {t0setname}")

        # Day
        t0_date_experiments = exps_df[exps_df['Date_pool_expt_started'] == associated_date][exps_df['short'].str.upper() == "TIME0"]
        # Lane (SetName)
        t0_setName_experiments = exps_df[exps_df['SetName'] == associated_setname][exps_df['short'].str.upper() == "TIME0"]

        if okLane and t0_date_experiments.shape[0] > 0:
            if t0setname in expsT0:
                del expsT0[t0setname]
            logging.info(f"Using Time0 from other lanes instead for {t0setname}")
            logging.info("Experiments affected:\n" + ", ".join(list(exps_df['name'][u])))
            #exps_df[u]['t0set'] = associated_date 
            for ix in range(len(u)):
                if u.iat[ix]:
                    exps_df['t0set'].iat[ix] = associated_date
            expsT0[associated_date] = list(exps_df['name'][exps_df['Date_pool_expt_started'] == \
                                           associated_date][exps_df['short'].str.upper() == "TIME0"])
        elif (okDay and t0_setName_experiments.shape[0] > 0 ):
            if t0setname in expsT0:
                del expsT0[t0setname]
            newt0sets = t0_setName_experiments['t0set']
            # Arbitrarily choosing the first one
            newt0set = newt0sets.iloc[0]
            logging.info(f"Note: Using Time0 from other days instead for {t0setname}.\n"
                          "Experiments affected:\n " + ", ".join(list(exps_df['name'][u])))

            #exps_df[u]['t0set'] = newt0set 
            for ix in range(len(u)):
                if u.iat[ix]:
                    exps_df['t0set'].iat[ix] = newt0set
        else:
            raise Exception(f"No Time0 for {t0setname}")


    if dbgp:
        print("A1 Final exps_df t0set:")
        print(exps_df['t0set'])
        print("A1 Final expsT0:")
        debug_print(expsT0, 'expsT0')

    return expsT0, exps_df


def get_nont0_sets(exps_df, okControls, debug_print_bool=False):
    """
    Args:
        exps_df (pandas DataFrame):
            contains columns: 
               t0set, 
               short
        okControls (bool): If True we get the non control groups by taking
                            control_bool == False
    Returns:
        unique_nont0sets list<str>: list of exps_df t0set values that don't have Time0 as their short,
    Description:
        Get all experiment's t0set (control group) strings that don't have 'Time0' as their short.
        In other words, get all control_group names for experiments that aren't controls. 
        Hopefully, every single one of these is associated with an existing control group,
        whose name is found in the dict expsT0.
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
        t0tot: a dataframe that contains the same number of rows as all.poolcount
        (nAllStrains), and the columns are one per each control_group (t0set).
        The values per row are the sums of the control experiments in that 
        control_group in that strain. Values are integers. For example
        suppose there are three control groups, each with two experiments
        that are associated with them. Then for each of those control groups,
        we create a column, and for each row in all_df, we take the sum
        of the two experiments associated with that group, and we are left
        with a dataframe (t0tot) with three columns and as many rows as in all_df,
        and in each row of t0tot, we have the 3 sums, each from the two control experiments
        associated with the control group which is the column name.
        
        
    Description:
        We create the data frame t0tot.
        First we take expsT0, which is a python dict
        which maps T0 group to list of experiment 
        names which belong to it (but only the controls,
        the true time0s, not any experiment to be
        compared to it). Then for each T0 group,
        we sum the experiments related to it 
        over all the reads. So we end up with a 
        dataframe that contains as many columns
        as T0 groups, and the number of rows in that
        column is as many as in all_df.

    """

    # We prepare to sum the values for all the pertinent setname-indexes for each datesetname
    # in expsT0.keys
    t0tot = {} #{date: pd_series([sum1, sum2, ...]) for date in expsT0.keys()}
    for date, exp_list in expsT0.items():
        print(date)
        print(exp_list)
        t0tot[date] = all_df[exp_list].sum(axis=1)

    # We recreate t0tot as a DataFrame
    t0tot = pd.DataFrame.from_dict(t0tot)

    if dbg_prnt:
        t0tot.to_csv("tmp/py_t0tot.tsv", sep= "\t")

    return t0tot


"""
def createIndexBy(all_df, central_insert_bool_list, print_bool=False):
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
    # All the locusIds which include insertions in the important regions
    indexBy = all_df['locusId'][central_insert_bool_list]
    if print_bool:
        debug_print(indexBy, 'indexBy')

    return indexBy
"""


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
        t0_gN:
            A dataframe with the same number of columns
            as t0tot + 1 (+1 for column 'locusIds'). Row number depends on the 
            number of unique locusIds in indexBy as well as 
            the genes with central insertions.
            It's length should be the same length as the number of 
            unique locusIds (num = nGenesUsed??, might be more because
            we aren't filtering for gene numbers that pass minT0GenesUsed)
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
    
    t0_gN = t0_gN.groupby(by="locusId", as_index=False).sum()

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
    nAllStrains = len(central_insert_bool_list)
    logging.info(f"Getting updated 'strainsUsed' boolean list. Num rows to parse: {nAllStrains}")
    strainsUsed = []
    for i in range(nAllStrains):
        if central_insert_bool_list[i] and t0tot.iloc[i,:].mean() >= minT0Strain:
            strainsUsed.append(True)
        else:
            strainsUsed.append(False)

    logging.info(f"Total number of strains used: {strainsUsed.count(True)}")

    return strainsUsed


def getNumUniqueUsableLocusIds(all_df, strainsUsed):
    """
    Description:
        We get the unique locus Ids where we can use the strain
    """

    logging.info("Getting number of unique usable locusIds.")
    unique_usable_locusIds = all_df['locusId'][strainsUsed].unique()
    num_unique_usable_locusIds = len(unique_usable_locusIds)
    if num_unique_usable_locusIds < 10:
        raise Exception("Less than ten usable locusIds, program designed to stop."
                        f" The number of usable genes is {num_unique_usable_locusIds}."
                        " The usable locusIds are: " + ", ".join(list(unique_usable_locusIds)))
    else:
        logging.info(f"Unique number of usable locusIds: {num_unique_usable_locusIds}")
    return num_unique_usable_locusIds



def getGenesUsedList(t0_gN, strainsUsed, all_df, minT0Gene, 
                     genes_df, minGenesPerScaffold,
                     minGenesAllowed,
                     debug_print_bool=False):
    """ We create the variable genesUsed_list
    Args:

        t0_gN:
            A dataframe with the same number of columns
            as t0tot + 1 (+1 for column 'locusIds'). Row number depends on the 
            number of unique locusIds in indexBy as well as 
            the genes with central insertions.
            It's length should be the same length as the number of 
            unique locusIds 
            Aggregate t0tot over locusIds 
        strainsUsed: list<bool> length of which is the same as all_df and t0tot
        all_df (pandas DataFrame): Uses col locusId
        minT0Gene: (int) 
        genesUsed_list (list): Inputted list of locusIds to be used (could be empty)
    Returns:
        genesUsed_list (list): list of unique locusIds such that their mean Time0 values
                    is greater than minT0Gene

    Description:
        We take t0_gN, which is the time0 totals summed over locusIds, 
        and we take the mean for each row over the Time0 reads.
        So now we have a series with row number = unique LocusIds,
        and values are the mean of the Time0 reads over that locusId.
        There are no longer as many columns as there are Time0 groups,
        (now there is only one column).
        Then we filter out the locusIds where the mean over the Time0
        reads is less than the integer threshold 'minT0Gene'.
        We store these initial locusIds (strings) as genesUsed list.
        Then we filter out genes that belong to scaffolds which
        have too few total genes on them. In other words, if a scaffold 
        has fewer genes on it than the integer 'minGenesPerScaffold', then we 
        won't use those genes in the analysis.
        Then we check that all the locusIds in the current genesUsed list
        are also in the genes_df (dataframe from genes.GC)
        We return this list.
    """

    # n0 is a pandas series with the mean for each row in t0_gN
    n0 = t0_gN.iloc[:,t0_gN.columns != 'locusId'].mean(axis=1)
    # Below we take the mean over the whole n0
    logging.info(f"Time0 reads per gene: mean {statistics.mean(n0)}"
                 f"median: {statistics.median(n0)} "
                 f" ratio: {statistics.mean(n0)/statistics.median(n0)}")

    # Below is boolean list of locations where the row mean passes minT0Gene
    # Each row is a locusId with the aggregated mean over all T0 (means)
    genesUsedpre = [(n0.iloc[i] >= minT0Gene) for i in range(n0.shape[0])]

    genesUsed_list = list(t0_gN['locusId'][genesUsedpre].unique())
    if debug_print_bool:
        with open("tmp/py_genesUsed.json","w") as f:
            f.write(json.dumps(genesUsed_list))

    removed_locusIds = list(all_locusIds_before_removal - set(genesUsed_list))

    logging.info(f"Number of LocusIds removed because the mean didn't pass {minT0Gene}:"
                f" {len(removed_locusIds)}. The locusIds: \n" + \
                 ", ".join(removed_locusIds))

    print(f"Initial number of remaining locusIds: {len(genesUsed_list)}")

    # HERE we refine the genesUsed list and remove genes which are in small scaffolds

    # genesPerScaffold is a dict {scaffoldId (str): num different locusIds in that scaffoldId}
    genesPerScaffold = getGenesPerScaffold(genes_df, genesUsed_list)

    # smallScaffold and smallLocusIds are both list<str>
    smallLocusIds = get_smallScaffoldLocusIds(genesPerScaffold, minGenesPerScaffold,
                                                     genes_df)

    # refining genesUsed_list - we remove the genes in small Scaffolds
    genesUsed_list = [x for x in genesUsed_list if x not in smallLocusIds]

    check_if_genes_not_in_genes_df(genesUsed_list, genes_df, minGenesAllowed=minGenesAllowed)


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

    # This is a dict with scaffoldId -> number of genes in that scaffold
    genesPerScaffold = py_table(list(genes_df['scaffoldId'][rows_with_locus_Ids_in_genesUsed_bool]
                                    ))

    return genesPerScaffold


def get_smallScaffoldLocusIds(genesPerScaffold, minGenesPerScaffold, genes_df, 
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
    for k, v in genesPerScaffold.items():
        logging.info(f"scaffold: {k}, num genes: {v}")
        if v < minGenesPerScaffold:
            smallScaffold.append(k)

    if debug_print_bool:
        debug_print(smallScaffold, 'smallScaffold')



    smallLocus_Ids = []
    if len(smallScaffold) > 0:
        logging.info("Ignoring genes on small scaffolds "
                     ", ".join(smallScaffold) + " " + \
                     "\ngenes left: " + str(len(genesUsed)) + "\n");
        for index, row in genes_df.iterrows():
            current_scaffold = row['scaffoldId']
            current_locus_id = row['locusId']
            if current_scaffold in smallScaffold:
                smallLocus_Ids.append(current_locus_id)

    return smallLocus_Ids


def check_if_genes_not_in_genes_df(genesUsed_list, genes_df, minGenesAllowed=100):
    """
    Args:
        genesUsed_list: list<locusId (str)>
        genes_df: Dataframe of genes.GC file (~12 columns)
        minGenesAllowed (int): If less than this number of genes, program stops.
    Returns:
        None
    Description:
        We go through each locusId in genesUsed_list and check if it is in
        the genes_df locusIds. If not, we raise an Error, the locusIds in
        all_df and the locusIds in genes.GC must be the same.
    """
    all_genes_locus_id = list(genes_df['locusId'])
    for x in genesUsed_list:
        if x not in all_genes_locus_id:
            raise Exception("LocusId {x} not in genes.GC file!")
    
    if len(genesUsed_list) < minGenesAllowed:
        raise Exception(f"Less than {minGenesAllowed} genes left; number left is {len(genesUsed_list)}"
                        ", exiting program.")
    

def print_info2(central_insert_bool_list, all_df, strainsUsed, genesUsed):
    """
    Args:
        central_insert_bool_list: list<bool>
        all_df: DataFrame of all.poolcount
        strainsUsed: list<bool>
        genesUsed: list<locusId (str)>
    Description:
        We print out logging info to the user.
    """
    
    # We count the number of Trues in central_insert_bool_list
    num_true_central_insert_bool_list = central_insert_bool_list.count(True)

    num_unique_locus_Ids = len(all_df['locusId'][central_insert_bool_list].unique())

    logging.info(f"Using {num_true_central_insert_bool_list} of { str(len(strainsUsed))} genic strains.")
    logging.info(f"Using {len(genesUsed)} of {num_unique_locus_Ids} genes with data.")

    return None

def get_GenesUsed12(minT0Gene, strainsUsed, all_df,
                    t0tot, genesUsed_list,
                    minLengthGenesUsed12=100):
    """
    Args:
        minT0Gene: int
        strainsUsed: list<bool> Length of all_df
        all_df: Dataframe needs col (f)
        t0tot: data frame where column names are 'date setname'
                and linked to a list of sums over the indexes that relate
                to that setname, with the list length being equal to the
                total number of strains (barcodes) in all.poolcount 
                (total number of rows is same as all.poolcount)
        genesUsed_list (list<str>): The original genesUsed
        minLengthGenesUsed12 (int): The minimum total number of genes
                            that have enough abundance of insertions on
                            both sides.
            
    Returns:
        genesUsed12: list of locusIds that have both high f (>0.5) and low f (<0.5)
                    insertions with enough abundance of insertions on both sides,
                    where the abundance is coming from the t0tot dataframe

    Description:
        We get the locusIds which have insertions both under 0.5 and over
        0.5 within the gene (percentage of length) and with values
        over minT0Gene/2. Then we also make sure all those genes are also
        in our original genesUsed_list, which have other thresholds, like
        belonging to large enough scaffolds.
        If the total number of remaining locusIds
        is less than minLengthGenesUsed12, then we raise an Exception. 

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
    
    print("Number of genesUsed12 before removing those not in genesUsed_list:"
          f" {len(genesUsed12)}")
    genesUsed12 = [x for x in genesUsed12 if (x in genesUsed_list)]
    print("Number of genesUsed12 after removing those not in genesUsed_list:"
          f" {len(genesUsed12)}")
        
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
                      'sum',
                      reset_index_bool=True)

    # Get all columns and rows besides locusId and take their minimum
    # Returns a pandas series with minimum of each row 
    crt_mins = crt.loc[:, crt.columns != 'locusId'].min(axis=1)
    #print(crt_mins)
    crt_row_min_bool = [bool(x >= minT0GeneSide) for x in list(crt_mins)]

    return crt, crt_row_min_bool


def check_that_every_t0set_is_in_t0tot(exps_df, t0tot):
    """
    Args:
        exps_df:
            Dataframe of FEBABarSeq.tsv
        t0tot: data frame where column names are 'date'
                and linked to a list of sums over the indexes that relate
                to that setname, with the list length being equal to the
                total number of strains (barcodes) in all.poolcount
        Description:
           We make sure every t0set value in the exps_df column 't0set'
           is also a column name in t0tot.
    """

    # We check if every t0set is in t0tot
    #{datesetname:[] for datesetname in expsT0.keys()}
    incorrect_sets = []
    t0sets = exps_df['t0set'].unique()
    for t0set in t0sets:
        if t0set not in t0tot.columns:
            incorrect_sets.append(t0set)

    if len(incorrect_sets) > 0:
        raise Exception("incorrect t0sets: \n" + ", ".join(incorrect_sets))


def UseControlGroupsToGetExpsDfAndExpsT0(exps_df):
    """
    Args:
        exps_df (pd DataFrame): Dataframe of experiments.
    Description:
        If the control groups are defined, then we use those to compute
        the controls and label the t0sets correctly.
    """

    if not "control_group" in exps_df.columns and "control_bool" in exps_df.columns:
        raise Exception("If using labelled controls (okControls) then you must include"
                        " column names 'control_group' and 'control_bool' in Experiments"
                        " file.")

    control_col = []
    for ix, val in exps_df["control_bool"].iteritems():
        if val.strip().upper() not in ["TRUE", "FALSE"]:
            raise Exception("Each control_bool value must be 'true' or 'false'."
                            f" In row {ix} value is {val}.")
        elif val.strip().upper() == "TRUE":
            control_col.append(True)
        elif val.strip().upper() == "FALSE":
            control_col.append(False)
        else:
            raise Exception(f"Error: cannot recognize control_bool value at row {ix}: {val}.")

    expsT0 = {}
    # Using enumerate on a python list to get index, value
    for ix, val in enumerate(control_col):
        if val:
            if exps_df['control_group'].iat[ix] in expsT0:
                expsT0[exps_df['control_group'].iat[ix]].append(exps_df['name'].iat[ix])
            else:
                expsT0[exps_df['control_group'].iat[ix]] = [exps_df['name'].iat[ix]]
            exps_df['short'].iat[ix] == "Time0"
        logging.debug(f"For row {ix}, control group is {exps_df['control_group'].iat[ix]}")
        exps_df['t0set'].iat[ix] == exps_df['control_group'].iat[ix]

    return expsT0, exps_df


