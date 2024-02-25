__author__ = "Neel Shah"
__visibility__ = "Course Runners, Students"

from Code.settings import *

def read_inflation():
    """
    Reads inflation.csv file
    Returns CPIAUCSL: Seasonally adjusted inflation from FRED
    Source:
    https://fred.stlouisfed.org/series/CPIAUCSL
    https://fred.stlouisfed.org/series/CPIAUCNS
    """

    data = pd.read_csv(os.path.join(data_path, 'inflation.csv'))
    data.rename(columns = {'DATE': 'Date'}, inplace = True)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'] + pd.offsets.MonthEnd(0)
    data = data.dropna()
    data = data.rename(columns={'Date': 'execution_date_monthend'})
    return data[['execution_date_monthend', 'CPIAUCSL']]

def get_resids(data, value_col, fe_cols, grouper = ['execution_date_monthend']):
    fe_cols = cList(fe_cols)
    fe_data = data[grouper + [value_col] + fe_cols].dropna()
    market_fe = pyhdfe.create(fe_data[fe_cols].astype(str))
    fe_data[value_col] = market_fe.residualize(fe_data[[value_col]])
    fe_data_raw = fe_data.groupby(grouper).mean()[value_col]
    # fe_data_ma = fe_data_raw.rolling(6).mean()
    return fe_data_raw

def get_resids2(data, value_col, fe_cols, grouper = ['execution_date_monthend']):
    fe_cols = cList(fe_cols)
    fe_data = data[grouper + [value_col, "Transaction SQFT"] + fe_cols].dropna() 
    market_fe = pyhdfe.create(fe_data[fe_cols].astype(str))             # create FE 
    fe_data[value_col] = market_fe.residualize(fe_data[[value_col]])    # get the residual of the value column (remove FE)
    
    # diff from get_resids() is here we weight by transaction SQFT
    fe_data["_r"] = fe_data[value_col] * fe_data['Transaction SQFT']    # numerator
    temp = fe_data[["_r", "Transaction SQFT"] + grouper].groupby(grouper).sum()     # sum of numerator and denominator
    
    temp[value_col] = temp["_r"] / temp["Transaction SQFT"]
    
    return temp[value_col]



def vprint(obj, verbose = True):
    """
    prints only if verbose, silent otherwise
    """
    if verbose: print(obj)

def writePickle(data, path, fileName):
    """
    Writes Pickle
    """
    with open(os.path.join(path, fileName), "wb") as output_file:
        cPickle.dump(data, output_file)
    return 1

def readPickle(path, fileName):
    """
    Reads Pickle
    """
    with open(os.path.join(path, fileName), "rb") as input_file:
        data = cPickle.load(input_file)
    return data

def cList(x):
    """
    :return: list depending upon input datatype
    """
    if type( x ) == list: return x
    if type( x ) == tuple: return list(x)
    if type( x ) in [str, int, float, np.float64]: return [ x ]
    if type( x ) in [pd.DataFrame]: return list( x.values )
    #[set, tuple, np.array, np.ndarray]
    else: return list(x)

def cListify_list(x):
    """
    x: any iterable
    converts every element of the iterable to a list
    """
    return [cList(i) for i in x]

# Flattens a list of lists
flatten = lambda x: [item for sublist in x for item in sublist]

def isOneToOne(df, col1, col2):
    """
    Checks if two columns of a dataframe are one-to-one
    df: pandas dataframe
    col1, col2: strings, names of columns to be compare
    return: boolean
    """
    first = df.groupby(col1)[col2].count().max()
    second = df.groupby(col2)[col1].count().max()
    return first + second == 2

def get_perc_av(df, col):
    """
    df: pandas dataframe
    col1: strings, names of columns to be compare
    return: % of non-null elements in a pandas column
    """
    print(len(df[df[col].notna()])/len(df))

def get_col_name_from_substr(data, substr):
    """
    data: pandas dataframe
    substr: string
    returns colun names that contain the substring; used for debgging
    """
    print([i for i in list(data) if substr in i.lower()])


def flip(items, ncol):
    """
    for graph plotting. flips list elements
    """
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

def winsorize(df, col_names, perc):
    """
    df: pandas dataframe
    col_names: list of columns to be winsorized
    perc: winsorize percentage (Eg. 95, 99, 99.9)
    return: dataframe df with winsorized columns
    """
    df = copy.deepcopy(df)
    for col in col_names:
        bottom, top = np.percentile(df[col], 100 - perc), np.percentile(df[col], perc)
        df.loc[df[col] < bottom, col] = bottom
        df.loc[df[col] > top, col] = top
    return df

def get_dummy(df, starting_year, ending_year):
    """generate independent tenary variable for repeated rent model"""
    df = df.copy()
    df = df[(df['old_com_year'] >= starting_year)]
    df = df[(df['new_com_year'] <= ending_year)]
    years = range(starting_year, ending_year + 1)
    qtrs = [1, 2, 3, 4]
    for year in years:
        for qtr in qtrs:
            df['{}Q{}'.format(year, qtr)] = 0.0

    df.reset_index(drop=True, inplace=True)
    for i in range(df.__len__()):
        df.loc[i, '{}Q{}'.format(df['old_com_year'][i], df['old_com_qtr'][i])] = -1.0
        df.loc[i, '{}Q{}'.format(df['new_com_year'][i], df['new_com_qtr'][i])] = 1.0

    return df, ['{}Q{}'.format(year, qtr) for year in years for qtr in qtrs]

def run_OLS_reg(data, y, x, intercept=True, verbose=True):
    data = data.dropna(subset=(cList(x) + cList(y)))
    X, Y = data[x], data[y]
    if intercept:
        X = sm.add_constant(X)
    X = X.astype(float)
    model = sm.OLS(Y, X)
    results = model.fit()
    if verbose:
        print(results.summary())
    return (results, data)
