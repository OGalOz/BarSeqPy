
import re
import logging
import pandas as pd

def ShortSetNames(set_names_list):
    """ Using a table with rules, shorten the names of these sets
    Args:
        set_names_list: list<str> set Names 
    Returns:
        set_names_list: list<str> Edited set Names to be 
            in the format setX* or testX*
    """

    if not isinstance(set_names_list, list):
        raise Exception("Input to ShortSetNames must be a list")

    logging.debug("Original set names list: " + ", ".join(set_names_list))

    # returns a TRUE/FALSE vector indicating which 
    # elements of the character vector contain a match
    simple = [bool(re.search(r"(set|test)[0-9A-Z]+[0-9A-Z0-9]*$", x)) for x in set_names_list]
    logging.debug("simple: \n" + ",".join(list([str(x) for x in simple])))
   
    # We edit the values of set_names_list who are true for simple ^
    # by removing anything before 'set' or 'test'
    # We count the number of values that were false
    nleft = 0
    simple_set_names = []
    for i in range(len(simple)):
        if simple[i]:
            new_set_name = re.sub("^.*(set|test)", "\\1", set_names_list[i]) 
            set_names_list[i] = new_set_name
            simple_set_names.append(new_set_name)
        else:
            nleft += 1
    logging.debug("fixed set_names:\n" + ",".join(list(set_names_list)))

    candidates = ["set" + x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    logging.debug(candidates)

    # get the elements in candidates that are not in set_names_list[simple]
    candidates = [x for x in candidates if x not in simple_set_names]
    if (nleft > len(candidates)):
        raise Exception(f"Too many unexpected set names: {nleft}")

    # Get the non-simple values from set_names_list
    oldComplex = [x for x in set_names_list if x not in simple_set_names]
    logging.debug("oldComplex:\n" + ",".join(oldComplex))

    cnd_ix = 0 
    for i in range(len(simple)):
        if not simple[i]:
            logging.info(f"Set {set_names_list[i]} simplified to {candidates[cnd_ix]}")
            set_names_list[i] = candidates[cnd_ix]
            cnd_ix += 1
            
    if (len(set_names_list) != len(set(set_names_list))):
        raise Exception("Non-unique set names!:\n" + \
                        ", ".join(set_names_list))
    else:
        logging.info("Finished running short set names")
        logging.debug("Final set names list: " + ", ".join(set_names_list))

    return(set_names_list)



def applyRules(rules_df, desc_str_list):
    """
    Args:
        rules_df: data frame with rules (looks like):
        str(number), str [, str [str, etc...] replace_str

        desc_str_list: list<str>
    """
    for j in range(len(desc_str_list)):
        for i in range(0, len(rules_df.index)):
            desc_str_list[j] = desc_str_list[j].replace(rules_df.iloc[i]["V1"], 
                                                        rules_df.iloc[i]["V2"])
    return desc_str_list
        



def testShortSetNames():
    logging.basicConfig(level=logging.DEBUG)
    set_name_list1 = [
            "mytest3",
            "myset5",
            "superdooper",
            "foo",
            "footest",
            "simpletest",
            "setA"
            ]
    ShortSetNames(set_name_list1)


if __name__ == "__main__":
    testShortSetNames()

