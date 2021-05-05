
import pandas

def debug_print(myvar, varname):
    print(f"{varname}, type: {type(myvar)}")
    if type(myvar) == list:
        if len(myvar) == 0:
            print("list of length 0")
        for i in range(len(myvar)):
            print(str(myvar[i]))
    elif type(myvar) == int:
        print(str(myvar))
    elif type(myvar) == str:
        print(myvar)
    elif type(myvar) == dict:
        for k in myvar.keys():
            if type(myvar[k]) == str:
                print(f"{k}:{myvar[k]}")
            elif type(myvar[k]) == int:
                print(f"{k}:{str(myvar[k])}")
            elif type(myvar[k]) == pandas.core.series.Series:
                print(k)
                print(myvar[k])
            elif type(myvar[k]) == list:
                if len(myvar[k]) == 0:
                    print(k + ":")
                    print("\tlist of length 0")
                else:
                    print(k + ":")
                    for i in range(len(myvar[k])):
                        print("\t" + str(myvar[k][i]))
            else:
                print(f"value for key {k} not printable: {type(myvar[k])}")
    elif type(myvar) == pandas.core.series.Series:
        print(myvar)
    else:
        print("Unknown how to print")

