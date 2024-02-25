__author__ = "Neel Shah"
__visibility__ = "Course Runners, Students"

import pandas as pd
import numpy as np
import socket
import _pickle as cPickle
import os
import copy
from dateutil.relativedelta import relativedelta
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, f1_score
from statsmodels.genmod.generalized_linear_model import GLM
from patsy import dmatrices
import sklearn
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
import datetime as dt
import matplotlib.ticker as mticker
import geopy
import json
import plotly.express as px
import config
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyhdfe
import itertools
from uszipcode import SearchEngine
import us
from tqdm import tqdm
import importlib
import swifter

import subprocess  
import sys
try:
    import ray
except ImportError:
    # robustly install packages in the current environment
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ray"])
    print('We just installed Ray for parallelizing groupby apply operations. Now you need to restart the kernel')


# =============================================================================
# Paths and Dates
# =============================================================================
# module_path = os.path.dirname(os.path.dirname(os.getcwd()))
module_path = os.path.dirname(os.path.dirname(os.getcwd()))
data_path = os.path.join(module_path, 'Data')
archive_path = os.path.join(data_path, 'Archive')
# Date until which data was downloaded
last_data_download_date = dt.date(2023, 12, 4)
last_data_download_date_str = last_data_download_date.strftime('%Y%m%d')
# Latest quarter for which complete data is available
last_data_download_quarter = dt.date(2023, 12, 31)


# =============================================================================
# Selections
# =============================================================================
state_area_name = {'NY': {'Main': ['NYC'],
                          'Others' : ['LongIsland', 'New York', 'Buffalo', 'Westchester']
                          },
                   'PA': {'Main': ['Pittsburgh', 'Philadelphia'],
                          'Others' : ['Pennsylvania']
                          },
                   'VA': {'Main': ['Richmond'],
                          'Others' : ['Virginia', 'Norfolk']
                          },
                   'MA': {'Main': ['Boston'],
                          'Others' : ['MAOther']
                          },
                   'TX': {'Main': ['San', 'Houston', 'Dallas', 'Austin'],
                          'Others' : ['Texas', 'El Paso']
                         },
                   'TN': {'Main': ['Nashville', 'Memphis', 'Knoxville'],
                          'Others' : [ 'Tennessee']
                          },
                   'OH': {'Main': ['Columbus', 'Cleveland', 'Dayton'],
                          'Others' : ['Ohio' ]
                         },
                   'NJ': {'Main': ['New Jersey - North and Central'],
                          'Others' : ['New Jersey - Other']
                         },
                   'NC': {'Main': ['Raleigh', 'Charlotte'],
                          'Others' : ['North Carolina']
                         },
                   'MN': {'Main': ['Minneapolis'],
                          'Others' : ['Minnesota']
                          },
                   'MD': {'Main': ['Baltimore'],
                          'Others' : ['Maryland']
                         },
                   'IL': {'Main': ['Chicago', 'St. Louis'],
                          'Others' : ['Illinois']
                         },
                   'GA': {'Main': ['Atlanta'],
                          'Others' : ['Georgia']
                         },
                   'FL': {'Main': ['Tampa', 'Miami', 'Orlando', 'Jacksonville'],
                          'Others' : ['Southwest', 'Panhandle']
                          },
                   'DC': {'Main': ['Washington DC'],
                          'Others' : []
                         },
                   'CO': {'Main': ['Denver'],
                          'Others' : ['Colorado']
                         },
                   'CA': {'Main': ['San Diego', 'Sacramento', 'Los Angeles', 'Bay Area', 'San Francisco'],
                          'Others' : ['California']
                         },
                   }

state_major_city_map = {'NY': ['New York'],
                        'CA': ['San Francisco', 'Los Angeles', 'San Diego', 'San Jose', 'Orange County', 'Seattle'],
                        'TX': ['Dallas', 'Houston', 'Austin', 'Phoenix'],
                        'DC': ['Washington DC'],
                        'IL': ['Chicago'],
                        'GA': ['Atlanta'],
                        'MA': ['Boston'],
                        'CO': ['Denver'],
                        'MD': [],
                        'MO': [],
                        'CT': [],
                        'DE': [],
                        'IN': [],
                        'NM': [],
                        'OR': ['Portland'],
                        'PA': ['Philadelphia', 'Pittsburgh'],
                        'FL': ['Miami'],
                        'NC': ['Charlotte'],
                        'MN': ['Minneapolis'],
                        'NJ': ['North Jersey'],
                        'TE': ['Nashville'],
                        'VA': []
                       }


# Some Hard Codes
lease_term_buckets = ['0-3Yrs', '3-5Yrs', '5-10Yrs', '10Yrs+']
major_states = ['NY', 'CA', 'TX', 'IL', 'MA', 'IL', 'DC', 'VA']
major_industries = ['Financial Services', 'Pharmaceuticals/Healthcare', 'Legal Services', 'Education', 'Non-Profit',
                    'Technology, Software, Telecom', 'Commercial & Professional Service', 'Media', 'Apparel/Retail']
# Subset of all major cities for plotting
major_cities = ['New York', 'San Francisco', 'Los Angeles', 'San Diego', 'Chicago', 'Dallas',
                 'Washington DC', 'Boston', 'Philadelphia',  'Orange County']




# =============================================================================
# Below are not used in the lecture notebook
# =============================================================================
# This line *MUST* be run on grid
# assert socket.gethostname().startswith('research'), "Please run this code on the grid"
# all_data_folder_path = '/shared/share_ztrax/'
# base = os.path.join(all_data_folder_path, 'CompStak')
# base_path = os.path.join(base, 'CompStak_combined')
# archive_path = os.path.join(base, 'Raw_compstack_data_archive')
# nyc_visualization_path = None
# # This is the folder containing the latest compstak data
# folders = 'CompStak_06_22_2022'

# #This folder contains the combined excel files.
# save_dir = os.path.join(base_path, folders)