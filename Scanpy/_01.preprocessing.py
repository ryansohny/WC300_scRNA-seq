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

clinic_info = pd.read_table("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/clinic_info.tsv", index_col=0)

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
S21_0043584N.obs['percent_mito'] = np.ravel(np.sum(S21_0043584N[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0043584N.X, axis=1))
S21_0043584T.obs['percent_mito'] = np.ravel(np.sum(S21_0043584T[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0043584T.X, axis=1))
S21_0046021T.obs['percent_mito'] = np.ravel(np.sum(S21_0046021T[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0046021T.X, axis=1))
S21_0047298T.obs['percent_mito'] = np.ravel(np.sum(S21_0047298T[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0047298T.X, axis=1))
S21_0048032N.obs['percent_mito'] = np.ravel(np.sum(S21_0048032N[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0048032N.X, axis=1))
S21_0048032T.obs['percent_mito'] = np.ravel(np.sum(S21_0048032T[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0048032T.X, axis=1))
S21_0048036T.obs['percent_mito'] = np.ravel(np.sum(S21_0048036T[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0048036T.X, axis=1))
S21_0049142T.obs['percent_mito'] = np.ravel(np.sum(S21_0049142T[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0049142T.X, axis=1))

'''
for sample in [m01, m10, m20]:
    sce.pp.scrublet(sample, adata_sim=None, sim_doublet_ratio=2.0, expected_doublet_rate=0.05, stdev_doublet_rate=0.02, synthetic_doublet_umi_subsampling=1.0, knn_dist_metric='euclidean', n_prin_comps=30, verbose=True)
'''

# First Integrate and then execute QC
integrated = AnnData.concatenate(S21_0043584N, S21_0043584T, join='outer', batch_key='Sample', batch_categories = ['S21_0043584N', 'S21_0043584T'], index_unique = '-')
sc.pp.calculate_qc_metrics(integrated, inplace=True)

sc.pl.violin(integrated, 'total_counts', groupby='Sample', log=True, size=2, cut=0)
sc.pl.violin(integrated, 'percent_mito', groupby='Sample', size=2, cut=0)
sc.pl.scatter(integrated, 'total_counts', 'n_genes_by_counts', color='Sample', size=10)

sns.histplot(data=integrated[integrated.obs['total_counts'] < 25000].obs, x='total_counts', kde=True, hue='Sample')

sc.pp.filter_cells(m01, min_counts=2000)
sc.pp.filter_cells(m01, min_genes=1500)

sc.pp.filter_cells(m10, min_counts=3000)
sc.pp.filter_cells(m10, min_genes=1500)

sc.pp.filter_cells(m20, min_counts=3000)
sc.pp.filter_cells(m20, min_genes=1500)

m01 = m01[m01.obs['percent_mito'] < 0.2]
m10 = m10[m10.obs['percent_mito'] < 0.2]
m20 = m20[m20.obs['percent_mito'] < 0.2]

integrated = AnnData.concatenate(m01, m10, m20, join='outer', batch_categories = ['m01', 'm10', 'm20'], index_unique = '-')
integrated.obs['Doublet'] = integrated.obs['predicted_doublet'].astype(str).astype('category')
integrated.obs[['Doublet', 'batch']].value_counts()
del integrated.obs['predicted_doublet']

sc.pp.filter_genes(integrated, min_cells=5) # 'n_cells' added in integrated.var 
integrated.layers["counts"] = integrated.X.copy()
integrated.raw = integrated