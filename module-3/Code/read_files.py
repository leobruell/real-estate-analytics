__author__ = "Neel Shah"
__visibility__ = "Course Runners, Students"

from Code.settings import *
import Code.Util as util

#######################################################################################
# Fetching and reading files
#######################################################################################

def get_file_paths(states, areas):
    """
    areas: string; one of ['Main', 'Others', 'All']
    states: list; should be a subset of states in keys of dictionry state_area_name in settings.py
    return: list of file paths for the states and areas specified
    """

    assert set(states).issubset(set(state_area_name.keys())), "Unknown State added"
    assert areas in ['Main', 'Others', 'All']

    if areas in ['Main', 'Others']:
        prefixes = util.flatten([state_area_name[state][areas] for state in states])
    else:
        prefixes = util.flatten([state_area_name[state]['Others'] for state in states]) + \
                   util.flatten([state_area_name[state]['Main'] for state in states])

    files = os.listdir(save_dir)
    files_list = [file for file in files if list(filter(file.startswith, prefixes)) != []]
    files_list = [os.path.join(save_dir, file) for file in files_list]
    return files_list

def clean_rents_raw(df, rent_column_stub, verbose):
    """
    df: pandas dataframe; leasing data
    rent_column_stub: string; either monthly or annual
    verbose: boolena; prints stuff if True
    return: leasing data df, with rent column annualized based on the column name
    Why was this required? legacy mess. Some files were downloaded with monthly rents and some with annual rents,
    the only distinguishing factor was the column names which has the stub annual or monthly in it. this function
    returns a cleaned dataframe with annualized rents
    """

    rent_column_annual = rent_column_stub + ' (per year)'
    rent_column_monthly = rent_column_stub + ' (per month)'

    col_names = list(df)
    # rent_value_type is a string that holds whether the rents are monthly or annual
    if (rent_column_annual in col_names) & (rent_column_monthly in col_names):
        if verbose: print("Both annual and monthly present for: {}".format(rent_column_stub))
        df[rent_column_monthly] = df[rent_column_monthly] * 12
        if "Rent_value_type" not in df.keys():
            df.loc[df[rent_column_monthly].apply(lambda x: not (x > -0.001)), "Rent_value_type"] = "Annual"
            df["Rent_value_type"] = df["Rent_value_type"].fillna("Monthly")
        df[rent_column_annual] = df[rent_column_annual].fillna(df[rent_column_monthly])
        df = df.drop(columns = rent_column_monthly)
        rent_value_type = 'Both'
    elif (rent_column_annual in col_names):
        if verbose: print("Only annual present for: {}".format(rent_column_stub))
        rent_value_type = 'Annual'
    elif (rent_column_monthly in col_names):
        if verbose: print("Only monthly present for: {}".format(rent_column_stub))
        df[rent_column_annual] = df[rent_column_monthly] * 12
        df = df.drop(columns = rent_column_monthly)
        rent_value_type = 'Monthly'
    else:
        rent_value_type = np.nan
    return df, rent_value_type

def read_one_file(path, clean = False, verbose = False, debug = False):
    """
    reads a single excel sheet for one city/state stored here: share_realestate_analytics/CompStak/CompStak_combined/CompStak_11_11_2021
    path: path to the file
    verbose: boolena; prints stuff if True
    debug: boolean
    clean: boolean
    return: tuple: (uncleaned file, types of rent values, i.e. an nual/monthly/both)
    Note: in this function, cleaning refers to annualizing rents only and
          not the actual cleaning which is done later. the dataframe returned iss till raw
    """

    df = pd.read_excel(path)

    # column_1 = list(df)[0]
    # if df[column_1].iloc[-1].startswith('This document is provided solely for'):
    #     df = df.iloc[:-1]
    # else:
    #     print("No Tail Information for: {}".format(os.path.split(path)[-1]))
    # if df[column_1].iloc[-1].startswith('Data provided by CompStak: http://CompStak.com'):
    #     df = df.iloc[:-1]
    # else:
    #     print("No Tail Information for: {}".format(os.path.split(path)[-1]))
    #
    # df = df.dropna(how = 'all').drop_duplicates()

    file_name = os.path.split(path)[-1]
    state = file_name.split('_')[1]
    assert state in list(state_area_name.keys()), "File name slicing error, Invalid state name"

    df['file_name'] = file_name.replace('.xlsx', '')
    # Fill Missing values in state
    if 'State' in list(df):
        df['State'] = df['State'].fillna(df['State'].mode().values[0])
    else:
        if state in list(state_area_name.keys()): df['State'] = state

    if clean or debug:
        # Clean rental Column at an individual level
        rent_columns_stubs = ['Effective Rent (USD)', 'Starting Rent (USD)',
                              'Blended Rent (USD)', 'Office Portion Rent (USD)',
                              'Asking Rent (USD)']

        rent_value_types = []
        for rent_column_stub in rent_columns_stubs:
            df, rent_value_type = clean_rents_raw(df, rent_column_stub, verbose)
            rent_value_types.append(rent_value_type)

        present_main = len(list(filter(file_name.startswith, state_area_name[state]['Main']))) > 0
        present_sub = len(list(filter(file_name.startswith, state_area_name[state]['Others']))) > 0

        # Note that the order should give preference to main (because of NYC in Main and NY in others)
        if not present_main: assert present_sub, "{}_file".format(path)  # It should be present in either of those

        if present_main:
            # Do the city filling now. The idea is to fill in missing cities with their modal values only for major
            # cities because there, the likelihood is ~1. In other files, the city could be anything.
            # df['City'] = df['City'].fillna(df['City'].mode().values[0])
            pass

        # File can have either annual rents or Monthly rents
        assert not (('Annual' in rent_value_types) and ('Monthly' in rent_value_types))
        if 'Monthly' in rent_value_types: df['Rent_value_type'] = 'Monthly'
        elif 'Annual' in rent_value_types: df['Rent_value_type'] = 'Annual'
        # else: df['Rent_value_type'] = 'Unknown'

    else:
        rent_value_types = []

    return df, set(rent_value_types)

def get_full_df(states, areas, types = ['Office'], clean = True, verbose = True, debug = False,
                start_date = dt.date(2000, 1, 1), cutoff = None):
    """
    states: list of states
    areas: areas: string; one of ['Main', 'Others', 'All']
    types: list; for this code it is generally ['Office']
    verbose: boolena; prints stuff if True
    debug: boolean
    start_date: data start date. generally Jan 1 2000 for most analysis, but can be earlier for
    calculating some variables and for matching leases for repeat rent model
    cutoff: data until which data was downloaded.
    return: tuple: (data_dump, cleaned_data_dump) i.e. raw data, cleaned data
    Note: Here cleaning refers to complete cleaning
    """
    paths = get_file_paths(states, areas)
    data_dump = [read_one_file(path, clean, verbose, debug)[0] for path in paths]

    data_dump = pd.concat(data_dump)
    data_dump = data_dump.dropna(axis = 1, how = 'all').dropna(how = 'all')

    check_cols = ['State', 'Street Address', 'Building Name', 'Effective Rent (USD) (per year)', 'Tenant Name',
                  'Commencement Date', 'Execution Date', 'Lease Term']
    data_dump = data_dump[(data_dump['Space Type'].isna()) | (data_dump['Space Type'].isin(types))]
    data_dump = data_dump[(data_dump['Property Type'].isna()) | (data_dump['Property Type'].isin(types))]

    data_dump = data_dump[~(data_dump['Execution Date'].dt.date > cutoff)]

    data_dump = data_dump.drop_duplicates(subset = check_cols)

    if verbose: print("Downloaded {} data for states: {}".format(areas, states))

    data_dump = data_dump.reset_index(drop=True)

    data_dump['lease_code'] = data_dump.apply(lambda x: x['State'] + '_' +
                                                        x['Transaction Quarter'].replace(' - ', '_') + '_' +
                                                        str(x.name).zfill(6), axis=1)

    if clean:
        cleaned_data_dump = clean_ll_data(data_dump, start_date, verbose)
        check_cols_clean = ['State', 'Street Address', 'Building Name', 'Effective Rent (USD) (per year)', 'Tenant Name',
                            'execution_date_monthend', 'commencement_date_monthend', 'Lease Term']

        cleaned_data_dump = cleaned_data_dump.drop_duplicates(subset = check_cols_clean)
    else:
        cleaned_data_dump = None
    return data_dump, cleaned_data_dump

#######################################################################################
# Cleaning loan term, rent schedule, effective rents, etc.
#######################################################################################

def lease_term_clean(term):
    """
    term: string
    cleans the lease term which is genrally specified as 'aybm', converted to lease term in months
    Eg. '15y4m' = 15*12 + 4 = 184 (months)
    """
    if type(term) == np.float and np.isnan(term): return np.nan
    term = term.split(',')
    m_months = 0
    for sub_part in term:
        sub_part = sub_part.split(' ')
        if sub_part[1] == 'years': m_months += np.float(sub_part[0])*12
        elif sub_part[1] == 'months': m_months += np.float(sub_part[0])
        else: raise ValueError
    return m_months

def get_monthly_payment(x):
    """
    x = 1 row of lease level dataframe
    converts a schedule of rents to a list of rents afte taking into account the work value and free months.
    Uses various algorithms and is subjective
    Eg. '0.0/3m,65.0/1y,74.0/3m' = [0, 0, 0, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 74, 74, 74]
    """
    schedule = x['Rent Schedule (USD)']

    lease_term = x['Lease Term']
    n_free_rent = x['Free Rent']

    starting_rent = x['Starting Rent (USD) (per year)']
    effective_rent = x['Effective Rent (USD) (per year)']

    Rent_value_type = x['Rent_value_type']
    tenant_improvement = x['Work Value (USD)']

    if pd.isnull(schedule):
        # Schedule is not available, construct it from starting rent and lease term
        if pd.notnull(starting_rent) and pd.notnull(lease_term):
            monthly_rents = [0.] * int(n_free_rent) + [starting_rent] * int(lease_term - n_free_rent)
        elif pd.notnull(lease_term) and pd.notnull(effective_rent):
            # construct starting rent from effective rent, lease term, free rent and TI
            starting_rent = (effective_rent * lease_term + tenant_improvement) / (lease_term - n_free_rent)
            monthly_rents = [0.] * int(n_free_rent) + [starting_rent] * int(lease_term - n_free_rent)
        else:
            monthly_rents = np.nan
    else:
        monthly_rents = []
        for factor in schedule.split(','):
            rent, period = factor.split('/')
            rent = np.float(rent)
            # Remove free rents explicitly
            if rent == 0.0: continue
            if Rent_value_type == 'Monthly':
                rent = rent * 12
            year_split = period.split('y')
            if year_split == ['']: continue
            # Case 1, just year
            if len(year_split) == 2 and year_split[1] == '':
                months = 12 * np.float(year_split[0])
            # Case 2, just month
            elif len(year_split) == 1:
                months = np.float(year_split[0].split('m')[0])
            elif len(year_split) == 2 and 'm' in year_split[1]:
                months = 12 * np.float(year_split[0])
                months += np.float(year_split[1].split('m')[0])

            monthly_rents.extend([rent] * int(months))
        # Now we have the raw rent schedule
        # Considerations: Does the rent schedule match the lease term? Is it impacted by free rents?
        if pd.notnull(lease_term):
            n_free_rent = int(n_free_rent)
            if int(lease_term) == len(monthly_rents):
                # # Add free rents randomly because we don't know where they will be
                # np.random.seed(0)
                # free_months = np.random.choice(len(monthly_rents), size=int(n_free_rent))
                # monthly_rents = [0 if i in free_months else rent for i, rent in enumerate(monthly_rents)]
                # Add free rents at the beginning
                monthly_rents = list(np.zeros(n_free_rent)) + monthly_rents[n_free_rent:]

            elif int(lease_term) > len(monthly_rents):
                balance = int(lease_term) - len(monthly_rents)

                # if the difference equals the free rent period
                if balance == n_free_rent:
                    # Insert zeros at n random places
                    # monthly_rents = list(np.insert(monthly_rents, np.random.choice(len(monthly_rents), size=balance), np.zeros(balance)))
                    # Insert zeros in the beginning
                    monthly_rents = list(np.zeros(balance)) + monthly_rents
                elif balance < n_free_rent:
                    other_balance = int(n_free_rent - balance)
                    # First make other balance number of months random
                    np.random.seed(1)
                    free_months = np.random.choice(len(monthly_rents), size=int(other_balance))
                    monthly_rents = [0 if i in free_months else rent for i, rent in enumerate(monthly_rents)]
                    # Now add balance number of free rents at the beginning
                    monthly_rents = [0.]*balance + monthly_rents
                elif balance > n_free_rent:
                    balance = int(n_free_rent)
                    monthly_rents = list(np.zeros(balance)) + monthly_rents
            elif int(lease_term) < len(monthly_rents):
                monthly_rents = monthly_rents[:int(lease_term)]

    # Final Check
    if type(monthly_rents) == np.float:
        assert np.isnan(monthly_rents)
    elif type(monthly_rents) == list:
        if np.sum(monthly_rents) == 0: monthly_rents = 'DROP'
    else:
        raise ValueError

    return monthly_rents

def get_starting_rent(x):
    """
    x = 1 row of lease level dataframe
    gets the starting rent for the lease from effective rent, leae term, number of free months, work values, etc.
    starting_rent = (effective_rent * lease_term + tenant_improvement) / (lease_term - n_free_rent)
    """
    schedule = x['monthly_rent_schedule']
    lease_term = x['Lease Term']
    effective_rent = x['Effective Rent (USD) (per year)']
    tenant_improvement = x['Work Value (USD)']
    n_free_rent = x['Free Rent']

    if pd.notnull(lease_term) and pd.notnull(effective_rent) and type(schedule) == list:
        # First Non-zero rent
        starting_rent = [i for i in schedule if i > 0.][0]
    elif pd.notnull(lease_term) and pd.notnull(effective_rent):
        starting_rent = (effective_rent * lease_term + tenant_improvement) / (lease_term - n_free_rent)
    else:
        starting_rent = effective_rent
    return starting_rent


def perc_in_nm(x, duration):
    """
    gives % of rent due in duration months, 
    x =  1 row of lease level dataframe
    duration: int
    if duration = 6, gives ratio: average rent in the first 6 months / average rent
    """
    monthly_rent_schedule = x['monthly_rent_schedule']
    starting_rent = x['Starting Rent (USD) (per year)']

    if (type(monthly_rent_schedule) == np.float) | (pd.isnull(starting_rent)):
        return np.nan

    frac = np.mean(monthly_rent_schedule[:duration]) / np.mean(monthly_rent_schedule)
    
    return frac


#######################################################################################
# Submarket cleanup
#######################################################################################

def clean_submarket(data):
    """
    cleans submarket by grouping neighbourhoods. NYC specific
    data: pandas dataframe; leasing data
    return: updated dataframe
    """
    replace_submarket = [('UN Plaza', 'Midtown Eastside'), 
                     ('Upper Eastside', 'North Manhattan'),
                     ('Upper Westside', 'North Manhattan'),
                     ('Hudson Yards', 'TS + Penn Station + Chelsea'),
                     ('Chelsea', 'TS + Penn Station + Chelsea'),
                     ('NoHo Greenwich Village', 'NoHo + SoHo + Hudson Square'),
                     ('SoHo', 'NoHo + SoHo + Hudson Square'),
                     ('Hudson Square', 'NoHo + SoHo + Hudson Square'),
                     ('Tribeca', 'NoHo + SoHo + Hudson Square'),             
                     ('City Hall Insurance', 'FiDi + City Hall + WTC'),
                     ('Financial District', 'FiDi + City Hall + WTC'),
                     ('World Trade Center', 'FiDi + City Hall + WTC'),                  
                     ('Times Square', 'TS + Penn Station + Chelsea'),
                     ('Times Square South', 'TS + Penn Station + Chelsea'),
                     ('Penn Station', 'TS + Penn Station + Chelsea'),
                     ('Sixth Avenue', 'Madison/Fifth/Sixth Avenue'),
                     ('Madison/Fifth Avenue', 'Madison/Fifth/Sixth Avenue'),
                     ('Gramercy Park Union Square', 'Gramercy Park Union Square + Murray Hill'),
                     ('Murray Hill', 'Gramercy Park Union Square + Murray Hill'), 
                     ('Gramercy Park/Union Square', 'Gramercy Park Union Square + Murray Hill'), 
                     ('Park Avenue', 'Park Avenue + Midtown Eastside'),
                     ('Midtown Eastside', 'Park Avenue + Midtown Eastside')
                    ]

    for old, new in replace_submarket:
        data.loc[(data['City'] == 'New York') & (data['Submarket'] == old),'Submarket'] = new
        
    return data

def clean_industry(data, method = 'M1'):
    """
    cleans industries by grouping them into similar industries together
    method: string; one of 'M1', 'M2'
    data: pandas dataframe; leasing data
    """
    replace_industry = [('Technology Hardware & Equipment', 'Technology, Software, Telecom'),
                        ('Telecommunication Services', 'Technology, Software, Telecom'),
                        ('Software & Information', 'Technology, Software, Telecom'),
                        ('Insurance', 'Financial Services'),
                        ('Banks', 'Financial Services'),
                        ('Financial Services', 'Financial Services'),
                        ('Pharmaceuticals, Biotechnology & Life Sciences', 'Pharmaceuticals/Healthcare'),
                        ('Health Care Equipment & Services', 'Pharmaceuticals/Healthcare'),
                        ('Utilities', 'Consumer Durables/Utilities'),
                        ('Consumer Durables', 'Consumer Durables/Utilities'),
                        ('Apparel', 'Apparel/Retail'),
                        ('Retail', 'Apparel/Retail')
                         ]

    if method == 'M2':
        replace_industry2 = [('Automobiles & Components', 'Automobiles, Capital Goods'),
                        ('Capital Goods', 'Automobiles, Capital Goods'),
                        ('Consumer Durables/Utilities', 'Consumer Durables/Utilities, Materials'),
                        ('Materials', 'Consumer Durables/Utilities, Materials'),
                        ('Legal Services', 'Commercial & Professional Service'),
                        ('Non-Profit', 'Non-Profit/Public Institutions/Education'),
                        ('Education', 'Non-Profit/Public Institutions/Education'),
                        ('Public Institutions', 'Non-Profit/Public Institutions/Education'),
                        ('Food & Beverage', 'Consumer Durables/Utilities, Materials'),
                        ]
        replace_industry.extend(replace_industry2)


    for old, new in replace_industry:
        if method == 'M1':
            data.loc[data['Tenant Industry'] == old, 'Tenant Industry'] = new
        if method == 'M2':
            data.loc[data['Tenant Industry'] == old, 'Tenant Industry Concise'] = new

    return data

def clean_city(data):
    """
    Cleans city by combining boroughs of NYC and Jersey, etc./nearby cities for orange county
    data: pandas dataframe; leasing data
    """

    Orange_county_cities = ['Aliso Viejo', 'Anaheim', 'Buena Park', 'Costa Mesa', 'Cypress', 'Dana Point',
                            'Fountain Valley', 'Fullerton', 'Garden Grove', 'Huntington Beach', 'La Habra',
                            'Laguna Hills', 'Los Alamitos', 'Mission Viejo',  'Newport Beach', 'Orange', 'Placentia',
                            'Rancho Santa Margarita', 'Santa Ana', 'Irvine', 'San Clemente', 'San Juan Capistrano',
                            'Tustin']
    North_Jersey_cities = ['Jersey City', 'Hoboken', 'Newark', 'Union City']

    replace_city = [(['Washington'], 'Washington DC', 'DC'),
                    (Orange_county_cities, 'Orange County', 'CA'),
                    (['Miami Beach'], 'Miami', 'FL'),
                    (['Cambridge'], 'Boston', 'MA'),
                    (['Nashville-Davidson'], 'Nashville', 'TN'),
                    (North_Jersey_cities, 'North Jersey', 'NJ'),
                    (['Brooklyn'], 'New York', 'NY'),
                    (['Bronx'], 'New York', 'NY'),
                    (['Manhattan'], 'New York', 'NY'),
                    (['Staten Island'], 'New York', 'NY'),
                    ]

    for old, new, state in replace_city:
        data.loc[(data['City'].isin(old)) & (data['State'] == state) , 'City'] = new

    return data

#######################################################################################
# Fill Missing dates
#######################################################################################

def fill_missing_dates(data):
    """
    cleans data based on dates.
    Removes observations where a combination of commencement date, Lease Term and lease End date does not make sense
    fills missing values
    data: pandas dataframe; leasing data
    return: cleaned data
    """

    # Backout missing commencement and end dates
    data['commencement_date_monthend'] = data['Commencement Date'] + pd.offsets.MonthEnd(0)
    data['execution_date_monthend'] = data['Execution Date'] + pd.offsets.MonthEnd(0)

    data['execution_to_commencement'] = (data['commencement_date_monthend'] - data['execution_date_monthend']).dt.days

    data['transaction_quarter_lowerbound_m'] = data['transaction_quarter'].map({'Q1': 1, 'Q2': 4, 'Q3': 7, 'Q4': 10})
    data['transaction_quarter_upperbound_m'] = data['transaction_quarter'].map({'Q1': 3, 'Q2': 6, 'Q3': 9, 'Q4': 12})

    # data['transaction_year'] = data['Transaction Quarter'].apply(lambda x: x[:4]).astype(int)
    # data['transaction_quarter'] = data['Transaction Quarter'].apply(lambda x: x[-2:])

    data['ones'] = 1
    data['transaction_quarter_lowerbound'] = pd.to_datetime( data[['transaction_year', 'transaction_quarter_lowerbound_m', 'ones']].\
        rename(columns={'transaction_year': 'year', 'transaction_quarter_lowerbound_m': 'month', 'ones': 'day'})) + pd.offsets.MonthEnd(0)

    data['transaction_quarter_upperbound'] = pd.to_datetime( data[['transaction_year', 'transaction_quarter_upperbound_m', 'ones']].\
        rename(columns={'transaction_year': 'year', 'transaction_quarter_upperbound_m': 'month', 'ones': 'day'})) + pd.offsets.MonthEnd(0)

    data['isvalid_execution_date'] = ((data['Transaction Quarter'].isna()) |
                                     (data['execution_date_monthend'].isna())) | \
                                     ((data['transaction_quarter_lowerbound'] <= data[  'execution_date_monthend']) &
                                     (data['execution_date_monthend'] <= data['transaction_quarter_upperbound']))

    data['isvalid_commencement_date'] = (data['commencement_date_monthend'].isna()) | \
                                        (data['execution_date_monthend'].isna()) | \
                                        ((data['execution_to_commencement'] >= -92) &
                                        (data['execution_to_commencement'] <= 370))

    data = data.drop(columns=['transaction_quarter_lowerbound_m', 'transaction_quarter_upperbound_m'])

    assert all(data['isvalid_execution_date']), 'all execution dates not valid'

    data['commencement_date_calculated'] = data.apply( lambda x: np.nan if (pd.isnull(x['Expiration_date_monthend']) |
                                                                            (pd.isnull(x['Lease Term'])))
                                                                        else x[ 'Expiration_date_monthend'] - relativedelta(months=x['Lease Term']), axis=1) + pd.offsets.MonthEnd(0)

    data['ismatch_commencement_date'] = (data['commencement_date_monthend'].isna()) | \
                                        ( data['commencement_date_calculated'].isna()) | \
                                        (np.abs((data['commencement_date_calculated'] - data['commencement_date_monthend'])).dt.days < 35)

    # If calculated commencement date does not match given commencement date,
    # change commencement date to calculated commencement date if it is within 30 days of execution date,
    # provided that the execution date itself is valid
    data.loc[(data['ismatch_commencement_date'] == False) &
             (data['isvalid_execution_date']) &
             (np.abs((data[ 'commencement_date_calculated'] - data[ 'execution_date_monthend']).dt.days) < 35),
             'commencement_date_monthend'] = data['commencement_date_calculated']

    # recalculate ismatch commencement
    data['ismatch_commencement_date'] = (data['commencement_date_monthend'].isna()) |\
                                        (data['commencement_date_calculated'].isna()) | \
                                        (np.abs((data['commencement_date_calculated'] - data['commencement_date_monthend'])).dt.days < 35)

    # Drop the rest of the cases where commencement date does. not match
    data = data[data['ismatch_commencement_date']]

    data['execution_to_commencement_calculated'] = (data['commencement_date_calculated'] - data['execution_date_monthend']).dt.days

    data['isvalid_commencement_date_calculated'] = (data['commencement_date_calculated'].isna()) | \
                                                   ( data['execution_date_monthend'].isna()) | \
                                                   ((data['execution_to_commencement_calculated'] >= -92) &
                                                    (data['execution_to_commencement_calculated'] <= 370))

    # Fill missing commencement dates
    data['commencement_date_monthend'] = data['commencement_date_monthend'].fillna(data['commencement_date_calculated'])

    # Execution date fill na based on quarter
    data['execution_date_monthend_estimated'] = np.nan
    data['execution_date_monthend_estimated'] = data[['transaction_quarter_upperbound', 'commencement_date_monthend']].min(axis=1)
    data['execution_date_monthend_estimated'] = data[['transaction_quarter_lowerbound', 'execution_date_monthend_estimated']].max(axis=1)

    # Fill missing execution dates
    data['execution_date_monthend'] = data['execution_date_monthend'].fillna(data['execution_date_monthend_estimated'])

    # re-calculate commencement date validity
    data['execution_to_commencement'] = (data['commencement_date_monthend'] - data['execution_date_monthend']).dt.days
    data['isvalid_commencement_date'] = (data['commencement_date_monthend'].isna()) | \
                                        ( data['execution_date_monthend'].isna()) |\
                                        ((data['execution_to_commencement'] >= -92) & (data['execution_to_commencement'] <= 370))

    date_check_cols = ['commencement_date_monthend', 'execution_date_monthend', 'Lease Term', 'isvalid_commencement_date',
                       'execution_to_commencement', 'transaction_quarter_lowerbound', 'transaction_quarter_upperbound']

    # Drop all elements where commencement date is still invalid
    data = data[data['isvalid_commencement_date']]
    data = data[data['isvalid_commencement_date_calculated']]
    data = data[data['isvalid_execution_date']]
    data = data[data['ismatch_commencement_date']]

    # Try to backout the very few missing expiration dates
    data['Expiration_date_calculated'] = data.apply( lambda x: np.nan if (pd.isnull(x['commencement_date_monthend']) |
                                                                          (pd.isnull(x['Lease Term'])))
                                                                      else x['commencement_date_monthend'] + relativedelta(months=x['Lease Term']), axis=1) + pd.offsets.MonthEnd(0)

    data['Expiration_date_monthend'] = data['Expiration_date_monthend'].fillna(data['Expiration_date_calculated'])

    data = data.drop(columns=['execution_date_monthend_estimated', 'Expiration_date_calculated',
                              'isvalid_commencement_date_calculated', 'commencement_date_calculated', 'isvalid_commencement_date',
                              'execution_to_commencement', 'ones', 'execution_to_commencement_calculated', 'ismatch_commencement_date',
                              'transaction_quarter_lowerbound', 'transaction_quarter_upperbound', 'isvalid_execution_date',
                              'transaction_quarter'])

    return data

#######################################################################################
# Sector, Dingel WFH Data
#######################################################################################

def fill_modal_generic(data, grouper, filler, verbose = False):
    """
    generic function to fill missing values with column modal values
    data: pandas dataframe; leasing data
    grouper: list of columns to group by
    filler: string, column whose missing values are to be filled
    verbose: boolen; prints stuff if True
    return: cleaned dataframe
    """

    try:
        modal_vals = data[util.cList(grouper) + [filler]].dropna().groupby(util.cList(grouper))[filler].agg(pd.Series.mode)
        modal_vals = modal_vals.apply(lambda x: x[0] if (type(x) == np.ndarray) and len(x) > 0 else x).reset_index()
    except:
        # Known pandas bug
        if verbose: print("Tried But failed")
        modal_vals = data[util.cList(grouper) + [filler]].dropna().groupby(util.cList(grouper))[filler].apply(pd.Series.mode).reset_index()
        if 'level_1' in list(modal_vals):
            modal_vals = modal_vals[modal_vals['level_1'] == 0].drop(columns=['level_1']).reset_index(drop=True)
        if 'level_2' in list(modal_vals):
            modal_vals = modal_vals[modal_vals['level_2'] == 0].drop(columns=['level_2']).reset_index(drop=True)
        if verbose: print("Success")

    modal_vals = modal_vals.rename(columns={filler: filler + ' Fill'})
    modal_vals[filler + ' Fill'] = modal_vals[filler + ' Fill'].astype(str).replace('[]', np.nan)

    if verbose: print("Filling: ", grouper, filler, len(modal_vals), len(data[data[filler].notna()]) / len(data))
    data = pd.merge(data, modal_vals, on= grouper, how='left')
    data[filler] = data[filler].fillna(data[filler + ' Fill'])
    data = data.drop(columns=[filler + ' Fill'])
    if verbose: print("     .Filled: ", grouper, filler, len(data[data[filler].notna()]) / len(data))
    return data

def add_sector(data):
    """
    adds SIC and NAICS codes, fills missing values in NAIC and industry names
    data: pandas dataframe; leasing data
    """

    # fill missing NAICS and SIC with those corresponding to modal values based on Industry Name to SIC/NAICS
    # industry_to_naics = data.groupby(['Tenant Industry'])['Tenant Naics'].agg(pd.Series.mode).\
    #     apply(lambda x: x[0] if type(x) == np.ndarray else x).reset_index().rename(columns={'Tenant Naics': 'Tenant Naics Fill'})
    # data = pd.merge(data, industry_to_naics, on='Tenant Industry', how='left')
    # data['Tenant Naics'] = data['Tenant Naics'].fillna(data['Tenant Naics Fill'])
    data = fill_modal_generic(data, 'Tenant Industry', 'Tenant Naics')

    # parent_to_naics = data.groupby(['Tenant Parent'])['Tenant Naics'].agg(pd.Series.mode).apply(
    #     lambda x: x[0] if (type(x) == np.ndarray) and len(x) > 0 else x).reset_index().rename(
    #     columns={'Tenant Naics': 'Tenant Naics Fill tenant'})
    # parent_to_naics = parent_to_naics[parent_to_naics['Tenant Naics Fill tenant'].apply(lambda x: len(x) if type(x) == list else 0) > 0]
    # data = pd.merge(data, parent_to_naics, on='Tenant Parent', how='left')
    # data['Tenant Naics'] = data['Tenant Naics'].fillna(data['Tenant Naics Fill tenant'])
    data = fill_modal_generic(data, 'Tenant Parent', 'Tenant Naics')

    data.loc[data['Tenant Naics'].notna(), 'Tenant Naics'] = data['Tenant Naics'].dropna().astype(float).astype(int).astype(str)

    data['NAICS2'] = data['Tenant Naics'].apply(lambda x: x[:2] if not pd.isnull(x) else np.nan)
    data['NAICS3'] = data['Tenant Naics'].apply(lambda x: x[:3] if not pd.isnull(x) else np.nan)

    # fill missing Industry names with those corresponding to modal values based on NAICS to Industry Name
    data = fill_modal_generic(data, 'NAICS3', 'Tenant Industry')

    assert data['NAICS3'].value_counts().sum() == data['NAICS2'].value_counts().sum() == \
           data['Tenant Naics'].value_counts().sum(), "Sector data mismatch"

    return data

def add_dingel_data(data):
    """
    adds column 'teleworkable_emp3_mean' which contains the average Dingel Nieman WFH Score for that indsutry.
    this is done progressivly first using NAICS 3, then NAICS 2 and finally on industry names
    data: pandas dataframe; leasing data
    """

    # Dingel data
    dingel_2 = pd.read_excel(os.path.join(data_path, 'Dingel_naics.xlsx'), 'NAICS2')
    dingel_2.columns = [i + '2' for i in dingel_2.columns]
    dingel_2['NAICS2'] = dingel_2['NAICS2'].astype(str).str.zfill(2)
    dingel_3 = pd.read_excel(os.path.join(data_path, 'Dingel_naics.xlsx'), 'NAICS3')
    dingel_3.columns = [i + '3' for i in dingel_3.columns]
    dingel_3['NAICS3'] = dingel_3['NAICS3'].astype(str).str.zfill(3)

    data = pd.merge(data, dingel_3[['NAICS3', 'teleworkable_emp3']], on='NAICS3', how='left')
    data = pd.merge(data, dingel_2[['NAICS2', 'teleworkable_emp2']], on='NAICS2', how='left')

    data['teleworkable_emp3'] = data['teleworkable_emp3'].fillna(data['teleworkable_emp2'])

    data = data.drop(columns='teleworkable_emp2')

    # This somewhat calculates a weighted average of NAICS2-Tenant industry scores
    data['teleworkable_emp3_mean'] = data.groupby('Tenant Industry')['teleworkable_emp3'].transform('mean')
    data['teleworkable_emp3'] = data['teleworkable_emp3'].fillna(data['teleworkable_emp3_mean'])

    data = data.drop(columns=['teleworkable_emp3_mean']) # 'Tenant SIC Fill',
    return data

#######################################################################################
# Transaction types
#######################################################################################

def modify_transaction_types(data):
    """
    Bunches different kinds of transaction types into a larger bucket
    data: pandas dataframe; leasing data
    """

    # Transaction type
    data['Transaction Type'] = data['Transaction Type'].replace('Extension/Expansion', 'Expansion').\
                                                        replace('Renewal/Expansion', 'Expansion').\
                                                        replace('Relet', 'Renewal').\
                                                        replace('Renewal/Contraction', 'Contraction')

    data['Transaction Type Generic'] = data['Transaction Type']
    data['Transaction Type Generic'] = data['Transaction Type Generic'].replace('Expansion', 'Renewal_Modif').\
                                                                        replace('Contraction', 'Renewal_Modif').\
                                                                        replace('Restructure', 'Renewal_Modif')

    data['Transaction Type Generic2'] = data['Transaction Type Generic']
    data['Transaction Type Generic2'] = data['Transaction Type Generic2'].replace('Renewal_Modif', 'Renewal').\
                                                                          replace('Extension', 'Renewal').\
                                                                          replace('Assignment', 'New Lease')
    data['Transaction Type Generic3'] = data['Transaction Type']

    data.loc[~data['Transaction Type Generic3'].isin(['Expansion', 'Renewal', 'New Lease']), 'Transaction Type Generic3'] = 'Others'
    return data

#######################################################################################
# Floors occupancy
#######################################################################################

# Floors occupies
def clean_floors(x):
    """
    x = 1 row of lease level data
    cleans floors occupied (string cleaning) removes stuff like partial, full, basement etc.
    Essentially ignores partial occupancy
    """
    if pd.isnull(x): return np.nan
    x = str(x)
    x = x.replace('Partial', '').replace('Entire', '').replace('Mezzanine', '')
    x = x.replace('Ground', '1').replace('Concourse', '1').replace('Lobby', '1')
    x = x.replace('Lower Level', '-1').replace('Lower', '-1').replace('Basement', '-1').replace('Subbasement', '-1')

    x = x.strip().split(',')

    def put_range(floor_list):
        """
        Strictly internal function; function internal to clean_floors
        floor_list: list
        return: list
        """
        floor_list = floor_list.split('-')
        floor_list = [i for i in floor_list if i != '']
        if len(floor_list) > 1:
            floor_list = list(range(int(floor_list[0]), int(floor_list[1])))
        return [str(i) for i in floor_list]

    x = util.flatten([put_range(i.strip()) for i in x])
    x = [int(np.float(i)) for i in x]
    if len(x) == 1:
        x = x[0]
    else:
        x = sorted(set(x))
    return x

def get_size_quartiles(df, col):
    """
    adds column quartiles based on columns col
    df: pandas dataframe; leasing data
    col: string; column name whose values are to be divided into quartiles
    """

    df['quartiles'] = np.nan
    Q1, Q2, Q3 = np.percentile(df[col].dropna(), 25), np.percentile(df[col].dropna(), 50), np.percentile(df[col].dropna(), 75)
    df.loc[df[col] <= Q1, 'quartiles'] = 'Q1'
    df.loc[(df[col] <= Q2) & (df[col] > Q1), 'quartiles'] = 'Q2'
    df.loc[(df[col] <= Q3) & (df[col] > Q2), 'quartiles'] = 'Q3'
    df.loc[df[col] > Q3, 'quartiles'] = 'Q4'
    return df['quartiles']

def get_size_quintiles(df, col):
    """
    adds column quintiles based on columns col
    df: pandas dataframe; leasing data
    col: string; column name whose values are to be divided into quintiles
    """

    df['quintiles'] = np.nan
    Q1, Q2, Q3, Q4 = np.percentile(df[col].dropna(), 20), np.percentile(df[col].dropna(), 40), \
                     np.percentile(df[col].dropna(), 60), np.percentile(df[col].dropna(), 80)
    df.loc[df[col] <= Q1, 'quintiles'] = 'Q1'
    df.loc[(df[col] <= Q2) & (df[col] > Q1), 'quintiles'] = 'Q2'
    df.loc[(df[col] <= Q3) & (df[col] > Q2), 'quintiles'] = 'Q3'
    df.loc[(df[col] <= Q4) & (df[col] > Q3), 'quintiles'] = 'Q4'
    df.loc[df[col] > Q4, 'quintiles'] = 'Q5'
    return df['quintiles']

def add_period_bools(data):
    """
    adds column 'time_period' to dataframe; divides leases into periods like pre_financial_crises,
                 post_financial_crises, pre_covid, etc. As far as I can remember, this is no longer used.
    data: pandas dataframe; leasing data
    """

    # Pandemic Boolean
    data['Pre_pandemic'], data['financial_crisis'] = 0, 0
    data.loc[data['execution_date_monthend'].dt.date < dt.date(2020, 2, 28), 'Pre_pandemic'] = 1
    data.loc[(data['execution_date_monthend'].dt.date > dt.date(2007, 6, 30)) &
             (data['execution_date_monthend'].dt.date <= dt.date(2009, 3, 31)), 'financial_crisis'] = 1

    data['time_period'] = np.nan
    data.loc[
        data['execution_date_monthend'].dt.date <= dt.date(2007, 6, 30), 'time_period'] = 'Pre_Financial_Crisis'
    data.loc[(data['execution_date_monthend'].dt.date > dt.date(2007, 6, 30)) &
             (data['execution_date_monthend'].dt.date <= dt.date(2009, 3, 31)), 'time_period'] = 'Financial_Crisis'
    data.loc[(data['execution_date_monthend'].dt.date > dt.date(2009, 3, 31)) &
             (data['execution_date_monthend'].dt.date <= dt.date(2015, 12,
                                                                 31)), 'time_period'] = 'Post_Financial_Crisis'
    data.loc[(data['execution_date_monthend'].dt.date > dt.date(2015, 12, 31)) &
             (data['execution_date_monthend'].dt.date < dt.date(2020, 2, 28)), 'time_period'] = 'Pre_Covid_19'
    data.loc[data['execution_date_monthend'].dt.date >= dt.date(2020, 2, 28), 'time_period'] = 'Covid_19'
    return data

def add_lease_term_buckets(data):
    """
    data: pandas dataframe; leasing data
    bunches lease term into 'Lease Term Bucket' like 0-3 yrs, 3-5 years, etc.
    """

    data['Lease Term Bucket'] = np.nan
    data.loc[(data['Lease Term'] <= 36), 'Lease Term Bucket'] = '0-3Yrs'
    data.loc[(data['Lease Term'] > 36) & (data['Lease Term'] <= 60), 'Lease Term Bucket'] = '3-5Yrs'
    data.loc[(data['Lease Term'] > 60) & (data['Lease Term'] <= 120), 'Lease Term Bucket'] = '5-10Yrs'
    data.loc[(data['Lease Term'] > 120), 'Lease Term Bucket'] = '10Yrs+'
    return data

#######################################################################################
# Get zip code from geo points
#######################################################################################

def get_zip(data, clean = True, use_pickle = 'geo_to_zip_3_digit_full.pkl', return_mapping_only = False):
    """
    data: pandas dataframe; leasing data
    clean: boolean
    use_pickle: None or string. If None, then will recalculate the mapping between geo codes and zip codes. (takes ~8 hours)
                if a string is specified, it will use that string. I've stored 'geo_to_zip_3_digit_full' file
                which contains geo code to zip code mapping for leases till November 2011.
                I do not expect many new additions of geo codes in the future, but good to check.
                regenerate this pickle whenever you update the data (leave it running overnight)
    return_mapping_only: if True, will not append the mapping to the original dataframe, will just return the mapping data
    return: populated dataframe. Adds zip code, cities and states based on geo code and US Post mapping
    """

    # Getting zip from geo point is time consuming.
    if clean:
        data['Geo Point'] = data['Geo Point'].apply(lambda x: [round(eval(x)[0], 3), round(eval(x)[1], 3)]).astype(str)

    def clean_calc_zip(x):
        """
        Strictly internal function; function internal to get_zip
        x = one row
        this function cleans the zip code string obtained from US Post
        """
        x = x.split('-')[0].split(':')[0]
        if len(x) < 4: x = x.zfill(5)
        if len(x) > 5:
            return np.nan
        else:
            return x

    if use_pickle is not None:
        geo_pts = util.readPickle(data_path, use_pickle)
    else:
        geolocator = geopy.Nominatim(user_agent='zip_code_getter')
        geo_pts = data['Geo Point'].to_frame()
        geo_pts = geo_pts.drop_duplicates()
        geo_pts = geo_pts[~(geo_pts['Geo Point'] == '[nan, nan]')]

        def get_zipcode(x):
            x = eval(x)
            location = geolocator.reverse((x[0], x[1]))
            try:
                address = location.raw['address']
            except:
                address = np.nan
                print("No Zip Code found")
            return address

        geo_pts['address_calculated'] = geo_pts['Geo Point'].apply(get_zipcode)
        geo_pts['Zipcode_calculated'] = geo_pts['address_calculated'].apply(lambda x: x['postcode'] if 'postcode' in x else np.nan)
        geo_pts['State_calculated'] = geo_pts['address_calculated'].apply(lambda x: x['state'] if 'state' in x else np.nan)
        geo_pts['City_calculated'] = geo_pts['address_calculated'].apply(lambda x: x['city'] if 'city' in x else np.nan)
        geo_pts['Town_calculated'] = geo_pts['address_calculated'].apply(lambda x: x['town'] if 'town' in x else np.nan)

        geo_pts['Area_Type_Geo'] = 'Town'
        geo_pts.loc[geo_pts['City_calculated'].notna(), 'Area_Type_Geo'] = 'City'
        geo_pts['City_calculated'] = geo_pts['City_calculated'].fillna(geo_pts['Town_calculated'])
        geo_pts = geo_pts[~(geo_pts['State_calculated'] == '山东省')]

        geo_pts['State_calculated'] = geo_pts['State_calculated'].apply(lambda x: us.states.lookup(x).abbr if not pd.isnull(x) else np.nan)
        
        geo_pts = geo_pts.dropna(subset = ['Zipcode_calculated'])

        print("Geo Points calculated")

        geo_pts = geo_pts.dropna(subset = ['Zipcode_calculated', 'State_calculated']).drop_duplicates(subset = ['Geo Point'])
        geo_pts['Zipcode_calculated'] = geo_pts['Zipcode_calculated'].apply(clean_calc_zip)

        if return_mapping_only: return geo_pts

    data = pd.merge(data, geo_pts[['Geo Point', 'Zipcode_calculated', 'State_calculated',
                                  'City_calculated', 'Area_Type_Geo']], on='Geo Point', how='left')

    if 'Zip Code' in list(data):
        data['Zip Code'] = data['Zip Code'].apply(lambda x: np.nan if pd.isnull(x) else clean_calc_zip(str(x)))

    return data


#######################################################################################
# clean data function
#######################################################################################


def clean_ll_data(data, start_date, verbose = False):
    """
    The main data cleaning function. Does a bunch of cleaning.
    data: pandas dataframe; leasing data
    start_date: date; used for getting the scaled CPI levels for deflated rents
    verbose: boolena; prints stuff if True
    """
    
    if verbose: print("Dropping unnecessary data")

    drop_col_list = ['Tenant Phone Number', 'Operating Expenses (USD)', 'Tenant Websites',  'Operating Expenses Notes',
                    'Suite', 'Operating Expenses Type',  'Annual Taxes', 'Amps',
                    'Street Frontage', 'Vented Space', 'Volts', 'Sprinkler', 'Additional Rent Free',
                    'Break Option Dates',  'Blended Rent (USD) (per year)', 'Break Option Dates',
                    'Clear Height', 'Elevators', 'Free Rent Type', 'Pro Rata Percent', 'Percentage Rent',
                    'Parking Ratio', 'Parking Notes',  'Parking Lot Type', 'Office Portion', 'Occupancy %',
                    'Lot Size', 'Load Factor', 'Loading Docks', 'Selling Basement', 'Retail Anchor',
                    'Renewal Options', 'Rent Bump Year', 'Total Transaction Size',
                    'Rail', 'Asking Rent (Gross Annual) (USD)', 'Days On Market', 'Corner Unit',
                    'Retail Notes', 'Cross Streets', 'Starting Rent (Gross Annual) (USD)', 'Doors',
                    'Last Updated', 'Date Created', 'Work Type']

    drop_col_list = list(set(drop_col_list) & set(list(data)))
    data = data.drop(columns = drop_col_list)

    # Cleaning transaction years
    data['transaction_year'] = data['Transaction Quarter'].apply(lambda x: x[:4]).astype(int)
    data['transaction_quarter'] = data['Transaction Quarter'].apply(lambda x: x[-2:])

    if start_date is not None:
        data = data[data['transaction_year'] >= start_date.year]  # Only data post 2000

    # Some Initial Fillings
    data = get_zip(data, use_pickle = 'geo_to_zip_3_digit_full.pkl')

    # Manual Corrections
    data.loc[(data['State_calculated'] == 'NJ') & (data['Zipcode_calculated'] == '10014'), 'State_calculated'] = 'NY'
    data.loc[(data['State_calculated'] == 'NJ') & (data['Zipcode_calculated'] == '10014'), 'City_calculated'] = 'New York'

    # Fill values
    data['City_calculated'] = data['City_calculated'].fillna(data['City'])
    data['State_calculated'] = data['State_calculated'].fillna(data['State'])
    data['Zipcode_calculated'] = data['Zipcode_calculated'].fillna(data['Zip Code'])

    data['City'] = data['City_calculated'].copy(deep = True)
    data['State'] = data['State_calculated'].copy(deep = True)
    data['Zip Code'] = data['Zipcode_calculated'].copy(deep = True)

    data = data.drop(columns=['City_calculated', 'State_calculated', 'Zipcode_calculated'])

    data = clean_city(data)
    data['Area_Type2'] = 'Nonmajor_market'
    data.loc[data['City'].isin(util.flatten(state_major_city_map.values())), 'Area_Type2'] = 'Major_Market'

    # Coworking spaces
    data.loc[data['Tenant Name'].isin(['Regus', 'WeWork', 'Convene']),'Tenant Industry'] = 'Real Estate'

    for filler in ['Building Size', 'Building Floors', 'Year Built', 'Building Name']:
        data = fill_modal_generic(data, ['Zip Code', 'Street Address'], filler)
    for filler in ['Building Size', 'Building Floors', 'Year Built']:
        data = fill_modal_generic(data, ['Zip Code', 'Building Name'], filler)
    for filler in ['Building Class']:
        data = fill_modal_generic(data, ['transaction_year', 'Zip Code', 'Building Name'], filler)

    for filler in [ 'Tenant Tickers', 'Tenant Ownership', 'Tenant Naics', 'Tenant Industry']:
            data = fill_modal_generic(data, 'Tenant Name', filler)
    for filler in ['Tenant Employees']:
        data = fill_modal_generic(data, ['transaction_year', 'Tenant Name'], filler)
        
    # Replace zero effective rents with Null
    data['Effective Rent (USD) (per year)'] = data['Effective Rent (USD) (per year)'].replace(0, np.nan)

    # Drop pre-leases and early renewals
    data = data[~data['Transaction Type'].isin(['Pre-lease', 'Early Renewal'])]

    # Drop where Lease term is not avaiable, by design, this will also drop missing expiration date leases
    data = data[data['Lease Term'].notna()]

    if verbose: print("Cleaning effective rents")
    # Effective Rent related cleaning
    data['Lease Term'] = data['Lease Term'].apply(lease_term_clean)
    data['Free Rent'] = data['Free Rent'].apply(lease_term_clean)
    data['Free Rent'] = data['Free Rent'].fillna(0)
    data['Work Value (USD)'] = data['Work Value (USD)'].fillna(0)

    # First pass for getting monthly payment schedule
    data['monthly_rent_schedule'] = data.apply(get_monthly_payment, axis=1)
    data = data[~(data['monthly_rent_schedule'] == 'DROP')]

    data['Starting Rent (USD) (per year)_calc'] = data.apply(get_starting_rent, axis=1)
    data['Starting Rent (USD) (per year)'] = data['Starting Rent (USD) (per year)'].fillna(data['Starting Rent (USD) (per year)_calc'])
    data = data.drop(columns = ['Starting Rent (USD) (per year)_calc'])

    # Fraction of avg(rent due in next n months) / overall effective rents
    data['frac_6'] = data.apply(lambda x: perc_in_nm(x, 6), axis=1)
    data['frac_12'] = data.apply(lambda x: perc_in_nm(x, 12), axis=1)
    data['frac_24'] = data.apply(lambda x: perc_in_nm(x, 24), axis=1)
    data['perc_free'] = data['Free Rent'] / data['Lease Term']

    data['effective_to_starting'] = data['Effective Rent (USD) (per year)'] / data['Starting Rent (USD) (per year)']

    data['Expiration_date_monthend'] = data['Expiration Date'] + pd.offsets.MonthEnd(0)

    # Age and Square footage
    data['log_sqft'] = data['Transaction SQFT'].apply(lambda x: np.log(x) if not pd.isnull(x) else np.nan)


    data.loc[data['Year Built'].notna(), 'Year Built'] = data['Year Built'].dropna().astype(np.float).astype(int)
    data.loc[data['Year Renovated'].notna(), 'Year Renovated'] = data['Year Renovated'].dropna().astype(np.float).astype(int)
    data['Year Renovated_filled'] = data['Year Renovated'].fillna(data['Year Built'])

    data['Age Building'] = data['transaction_year'] - data['Year Built']
    data['Renovation Age'] = data['transaction_year'] - data['Year Renovated_filled']

    data.loc[data['Age Building'] < 1, 'Age Building'] = np.nan
    data.loc[data['Renovation Age'] < 1, 'Renovation Age'] = np.nan

    data['Age Building_Dec2020'] = 2020 - data['Year Built']
    data.loc[data['Age Building_Dec2020'] < 1, 'Age Building_Dec2020'] = np.nan

    # Backing out commencement date, execution date and expiration date from existing data
    # Also test the validity of these dates
    # Drop bad leases while preserving most of the data
    if verbose: print("Filling Missing dates")
    data = fill_missing_dates(data)

    if verbose: print("Adding Dingel and sector data")
    data = add_dingel_data(add_sector(data))

    # Relocation flag
    data['relocation'] = 0

    data.loc[data['Comments'].apply(lambda x: 'reloc' in str(x).lower()), 'relocation'] = 1
    data.loc[data['Comments'].apply(lambda x: 'move ' in str(x).lower()), 'relocation'] = 1
    data.loc[data['Comments'].apply(lambda x: 'moving from' in str(x).lower()), 'relocation'] = 1


    if verbose: print("Tenant Ownership Cleaning")
    data = fill_modal_generic( data, 'Tenant Name', 'Tenant Ownership')

    if verbose: print("Modifying transaction types")
    data = modify_transaction_types(data)

    data = add_period_bools(data)

    if verbose: print("Cleaning floors occupied")
    data['Floors Occupied'] = data['Floors Occupied'].apply(clean_floors)
    data['Avg Floor Occupied'] = data['Floors Occupied'].apply(lambda x: np.mean(util.cList(x)))

    # Fill markets using zip codes
    data['Zip Code'] = data['Zip Code'].apply(lambda x: x if type(x) == np.float else x.split('-')[0])
    data.loc[data['Zip Code'].apply(lambda x: not x.isnumeric()), 'Zip Code'] = np.nan

    data.loc[data['Zip Code'].notna(), 'Zip Code'] = data['Zip Code'].dropna().astype(float).astype(int).astype(str)
    data = fill_modal_generic(data, 'Zip Code', 'Submarket')

    data = clean_submarket(data)

    data = data.drop(columns=['Execution Date', 'Commencement Date', 'Expiration Date'])

    # Add other columns
    data['Dollar_val_annualized'] = data['Transaction SQFT'] * data['Effective Rent (USD) (per year)']
    data['Log_Dollar_val_annualized'] = np.log(data['Dollar_val_annualized'])
    data['Log_Dollar_val_annualized'] = data['Log_Dollar_val_annualized'].replace([-np.inf, np.inf], 0)

    data['Tenant Industry Raw'] = data['Tenant Industry'].copy(deep = True)

    data = clean_industry(data, 'M1')
    data = clean_industry(data, 'M2')

    data['Tenant Industry filled'] = copy.deepcopy(data['Tenant Industry'])
    data['Tenant Industry filled'] = data['Tenant Industry filled'].fillna('Other')

    data['Tenant Industry Concise filled'] = copy.deepcopy(data['Tenant Industry Concise'])
    data['Tenant Industry Concise filled'] = data['Tenant Industry Concise filled'].fillna('Other')

    data['Listed'] = 'Unlisted'
    data.loc[data['Tenant Tickers'].notna(), 'Listed'] = 'Listed'

    data['Tenant Revenue (USD)'] = data['Tenant Revenue (USD)'].replace(0, 1.)
    data['Tenant Employees'] = data['Tenant Employees'].replace(0, 1.)

    #Add Employee and tenant size
    data['Tenant Employees'] = data['Tenant Employees'].astype(np.float)
    data['Tenant Revenue (USD)'] = data['Tenant Revenue (USD)'].astype(np.float)
    data['Employee_size'] = get_size_quartiles(data, 'Tenant Employees')
    data['Revenue_size'] = get_size_quartiles(data, 'Tenant Revenue (USD)')

    data['Log_tenant_revenue'], data['Log_tenant_employees'] = np.log(data['Tenant Revenue (USD)']), np.log(data['Tenant Employees'])

    # Add size of lease and size of buildings
    data['Building Size'] = data['Building Size'].astype(np.float)
    data['lease_size'] = get_size_quartiles(data, 'Transaction SQFT')
    data['building_size'] = get_size_quartiles(data, 'Building Size')

    data['log_building_size'] = np.log(data['Building Size'])

    data['Employee_size_quin'] = get_size_quintiles(data, 'Tenant Employees')
    data['Revenue_size_quin'] = get_size_quintiles(data, 'Tenant Revenue (USD)')
    data['lease_size_quin'] = get_size_quintiles(data, 'Transaction SQFT')
    data['building_size_quin'] = get_size_quintiles(data, 'Building Size')

    data['Age Building'] = data['Age Building'].astype(np.float)

    # Add Inflation
    inflation = read_inflation()
    inflation = inflation[inflation['execution_date_monthend'] >= data['execution_date_monthend'].min()]
    inflation['CPIAUCSL'] = inflation['CPIAUCSL'] / inflation[inflation['execution_date_monthend'] ==
                                                               data['execution_date_monthend'].min()]['CPIAUCSL'].iloc[0]
    data = pd.merge(data, inflation[['execution_date_monthend', 'CPIAUCSL']], on='execution_date_monthend', how='left')
    data['Effective Rent deflated (USD) (per year)'] = data['Effective Rent (USD) (per year)'] / data['CPIAUCSL']
    data['Starting Rent deflated (USD) (per year)'] = data['Starting Rent (USD) (per year)'] / data['CPIAUCSL']

    data = add_lease_term_buckets(data)

    # State Filling
    data['State_Filled'] = data['State'].copy(deep=True)
    data.loc[~(data['State'].isin(['CA', 'TX', 'NY', 'IL', 'VA', 'MA', 'DC', 'CO', 'GA', 'NJ', 'PA', 'MD', 'FL', 'NC',
                                   'CT', 'OH', 'TN', 'MN'])), 'State_Filled'] = 'Other'

    # Submarket Filling
    data['Submarket_Filled'] = data['Submarket'].copy(deep=True)
    data.loc[(~(data['Submarket'].isin(['Gramercy Park Union Square + Murray Hill', 'Park Avenue + Midtown Eastside',
                                        'Madison/Fifth/Sixth Avenue', 'Grand Central', 'FiDi + City Hall + WTC',
                                        'TS + Penn Station + Chelsea', 'Columbus Circle', 'NoHo + SoHo + Hudson Square',
                                        'North Manhattan'])) & (data['City'] == 'New York')), 'Submarket_Filled'] = 'Other'

    data['Building Class Enhanced'] = data['Building Class']
    data.loc[(data['Building Class'] == 'B') | (data['Building Class'] == 'C'), 'Building Class Enhanced'] = 'B + C'

    # Filling Transaction Types
    data['Transaction Type filled'] = data['Transaction Type'].copy(deep=True)
    data.loc[data['Transaction Type'].isin(['Contraction', 'Restructure', 'Assignment']), 'Transaction Type filled'] = 'Other'

    data['Ones'] = 1

    return data

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
    data['Date'] = pd.to_datetime(data['Date'], format = '%d/%m/%y')
    data['Date'] = data['Date'] + pd.offsets.MonthEnd(0)
    data = data.dropna()
    data = data.rename(columns={'Date': 'execution_date_monthend'})
    return data[['execution_date_monthend', 'CPIAUCSL']]

