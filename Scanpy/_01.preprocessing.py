## Temporarily,
## (in cm03) conda activate wc300_ver2
## ipython --profile=wc300scrna

from anndata import AnnData
import anndata
from scipy import sparse, io
import scipy
import pandas as pd
import scipy.io
import os
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors
matplotlib.use('TkAgg')
import numpy as np
import seaborn as sns
import math
import scanpy.external as sce
#import scrublet as scr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
sns.set(font="Arial", font_scale=1, style='ticks')
sc.settings.verbosity = 3
plt.rcParams['figure.figsize'] = (6,6)
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'Arial'
plt.rc("axes.spines", top=False, right=False)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#104e8b", "#ffdab9", "#8b0a50"])
batch_palette=['#689aff', '#fdbf6f', '#b15928']
%matplotlib
%autoindent

clinic_info = pd.read_table("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/sample.tsv", index_col=0)

S21_0043584N = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0043584N.h5")
S21_0043584T = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0043584T.h5")
S21_0046021T = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0046021T.h5")
S21_0047298T = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0047298T.h5")
S21_0048032N = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0048032N.h5")
S21_0048032T = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0048032T.h5")
S21_0048036T = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0048036T.h5")
S21_0049142T = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0049142T.h5")

S21_0043584N.var_names_make_unique()
S21_0043584T.var_names_make_unique()
S21_0046021T.var_names_make_unique()
S21_0047298T.var_names_make_unique()
S21_0048032N.var_names_make_unique()
S21_0048032T.var_names_make_unique()
S21_0048036T.var_names_make_unique()
S21_0049142T.var_names_make_unique()

mito_genes = S21_0043584N.var_names.str.startswith('MT-')
m01.obs['percent_mito'] = np.ravel(np.sum(m01[:, mito_genes].X, axis=1)) / np.ravel(np.sum(m01.X, axis=1))
m10.obs['percent_mito'] = np.ravel(np.sum(m10[:, mito_genes].X, axis=1)) / np.ravel(np.sum(m10.X, axis=1))
m20.obs['percent_mito'] = np.ravel(np.sum(m20[:, mito_genes].X, axis=1)) / np.ravel(np.sum(m20.X, axis=1))

for sample in [m01, m10, m20]:
    sce.pp.scrublet(sample, adata_sim=None, sim_doublet_ratio=2.0, expected_doublet_rate=0.05, stdev_doublet_rate=0.02, synthetic_doublet_umi_subsampling=1.0, knn_dist_metric='euclidean', n_prin_comps=30, verbose=True)
