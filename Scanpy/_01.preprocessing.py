## Temporarily,
## (in cm03) conda activate 
## ipython --profile=wc300_scrna

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
#matplotlib.use('TkAgg')
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
%autoindent
%matplotlib

clinic_info = pd.read_table("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/clinic_info.tsv", index_col=0)

S21_0043584N_MSS = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0043584N.h5")
S21_0043584T_MSS = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0043584T.h5")
S21_0046021T_MSI = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0046021T.h5")
S21_0047298T_MSS = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0047298T.h5")
S21_0048032N_MSS = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0048032N.h5")
S21_0048032T_MSS = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0048032T.h5")
S21_0048036T_MSI = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0048036T.h5")
S21_0049142T_MSI = sc.read_10x_h5("/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/Barcode_Files/S21-0049142T.h5")

S21_0043584N_MSS.var_names_make_unique()
S21_0043584T_MSS.var_names_make_unique()
S21_0046021T_MSI.var_names_make_unique()
S21_0047298T_MSS.var_names_make_unique()
S21_0048032N_MSS.var_names_make_unique()
S21_0048032T_MSS.var_names_make_unique()
S21_0048036T_MSI.var_names_make_unique()
S21_0049142T_MSI.var_names_make_unique()

mito_genes = S21_0043584N_MSS.var_names.str.startswith('MT-')
S21_0043584N_MSS.obs['percent_mito'] = np.ravel(np.sum(S21_0043584N_MSS[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0043584N_MSS.X, axis=1))
S21_0043584T_MSS.obs['percent_mito'] = np.ravel(np.sum(S21_0043584T_MSS[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0043584T_MSS.X, axis=1))
S21_0046021T_MSI.obs['percent_mito'] = np.ravel(np.sum(S21_0046021T_MSI[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0046021T_MSI.X, axis=1))
S21_0047298T_MSS.obs['percent_mito'] = np.ravel(np.sum(S21_0047298T_MSS[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0047298T_MSS.X, axis=1))
S21_0048032N_MSS.obs['percent_mito'] = np.ravel(np.sum(S21_0048032N_MSS[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0048032N_MSS.X, axis=1))
S21_0048032T_MSS.obs['percent_mito'] = np.ravel(np.sum(S21_0048032T_MSS[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0048032T_MSS.X, axis=1))
S21_0048036T_MSI.obs['percent_mito'] = np.ravel(np.sum(S21_0048036T_MSI[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0048036T_MSI.X, axis=1))
S21_0049142T_MSI.obs['percent_mito'] = np.ravel(np.sum(S21_0049142T_MSI[:, mito_genes].X, axis=1)) / np.ravel(np.sum(S21_0049142T_MSI.X, axis=1))

'''
for sample in [m01, m10, m20]:
    sce.pp.scrublet(sample, adata_sim=None, sim_doublet_ratio=2.0, expected_doublet_rate=0.05, stdev_doublet_rate=0.02, synthetic_doublet_umi_subsampling=1.0, knn_dist_metric='euclidean', n_prin_comps=30, verbose=True)
'''

# First Integrate and then execute QC
integrated = AnnData.concatenate(S21_0043584N_MSS, S21_0043584T_MSS, join='outer', batch_key='Sample', batch_categories = ['S21_0043584N_MSS', 'S21_0043584T_MSS'], index_unique = '-')
sc.pp.calculate_qc_metrics(integrated, inplace=True)

sc.pl.violin(integrated, 'total_counts', groupby='Sample', log=True, size=2, cut=0)
sc.pl.violin(integrated, 'percent_mito', groupby='Sample', size=2, cut=0)
sc.pl.scatter(integrated, 'total_counts', 'n_genes_by_counts', color='Sample', size=10)
sc.pl.scatter(integrated, 'total_counts', 'n_genes_by_counts', color='percent_mito', size=10)

sns.histplot(data=integrated[integrated.obs['total_counts'] < 25000].obs, x='total_counts', kde=True, hue='Sample')

'''
우선, Trans/Epi팀 분석대로, 이건 나중에 고쳐야 함.
'''
sc.pp.filter_genes(integrated, min_cells=3)
sc.pp.filter_cells(integrated, min_genes=200)
sc.pp.filter_cells(integrated, max_genes=8000)
integrated = integrated[integrated.obs['percent_mito'] < 0.25] # 20,145 cells (only S21_0043584N_MSS & S21_0043584T_MSS, 2023-03-23)
integrated.layers["counts"] = integrated.X.copy()
sc.pp.normalize_total(integrated, target_sum=1e4)
sc.pp.log1p(integrated)
sc.pp.highly_variable_genes(integrated, min_mean=0.0125, max_mean=3, min_disp=0.5)
integrated.raw = integrated
integrated.var['highly_variable'].value_counts() # 3,783 (only S21_0043584N_MSS & S21_0043584T_MSS, 2023-03-23)
sc.pp.scale(integrated, max_value=10)
sc.tl.pca(integrated, n_comps=50, use_highly_variable=True, svd_solver='arpack')
sc.pl.pca(integrated, color=['Sample'], legend_loc='right margin', size=8, add_outline=False, color_map='CMRmap', annotate_var_explained=True, components=['1,2'])
'''
sc.pp.neighbors(integrated, n_neighbors=10, n_pcs=15) # batch correction 능력 확인
sc.tl.umap(integrated, min_dist=0.5, spread=1.0, n_components=2, alpha=1.0, gamma=1.0, init_pos='spectral', method='umap')
sc.pl.umap(integrated[rand_is, :], color=['Sample'], add_outline=False, legend_loc='right margin', size=20)
'''
sce.pp.bbknn(integrated, batch_key='Sample', n_pcs=15, neighbors_within_batch=5, trim=None)
sc.tl.umap(integrated, min_dist=0.5, spread=1.0, n_components=2, alpha=1.0, gamma=1.0, init_pos='spectral', method='umap')
np.random.seed(42)
rand_is = np.random.permutation(list(range(integrated.shape[0])))
sc.pl.umap(integrated[rand_is, :], color=['Sample'], add_outline=False, legend_loc='right margin', size=20)

fig, axes = plt.subplots(1,2)
sc.pl.umap(integrated[rand_is, :], color=['Sample'], add_outline=False, legend_loc=None, size=20, groups=['S21_0043584N_MSS'], title='S21_0043584N_MSS', ax=axes[0])
sc.pl.umap(integrated[rand_is, :], color=['Sample'], add_outline=False, legend_loc=None, size=20, groups=['S21_0043584T_MSS'], title='S21_0043584T_MSS', ax=axes[1])

fig, axes = plt.subplots(1,5)
sc.pl.umap(integrated[rand_is, :], color=['CD8A'], add_outline=False, legend_loc=None, size=20, title='CD8+ T lymphocyte', ax=axes[0])
sc.pl.umap(integrated[rand_is, :], color=['IL7R'], add_outline=False, legend_loc=None, size=20, title='CD4+ T lymphocyte', ax=axes[1])
sc.pl.umap(integrated[rand_is, :], color=['MS4A1'], add_outline=False, legend_loc=None, size=20, title='B lymphocyte', ax=axes[2])
sc.pl.umap(integrated[rand_is, :], color=['EPCAM'], add_outline=False, legend_loc=None, size=20, title='Epithelial Cell', ax=axes[3])
sc.pl.umap(integrated[rand_is, :], color=['PECAM1'], add_outline=False, legend_loc=None, size=20, title='Endothelial Cell', ax=axes[4])

sc.tl.leiden(integrated, resolution=0.5, key_added='leiden_r05') #### 0 ~  ==> 2023-03-23
sc.tl.leiden(integrated, resolution=1.0, key_added='leiden_r10')
sc.pl.umap(test3, color=['batch', 'leiden_r05', 'leiden_r10'], add_outline=False, legend_loc='right margin', size=20)



'''
sc.pp.filter_cells(S21_0043584N_MSS, min_counts=2000)
sc.pp.filter_cells(S21_0043584N_MSS, min_genes=1500)

sc.pp.filter_cells(S21_0043584T_MSS, min_counts=3000)
sc.pp.filter_cells(S21_0043584T_MSS, min_genes=1500)

S21_0043584N_MSS = S21_0043584N_MSS[S21_0043584N_MSS.obs['percent_mito'] < 0.2]
S21_0043584T_MSS = S21_0043584T_MSS[S21_0043584T_MSS.obs['percent_mito'] < 0.2]

integrated = AnnData.concatenate(S21_0043584N_MSS, S21_0043584T_MSS, join='outer', batch_key='Sample', batch_categories = ['S21_0043584N_MSS', 'S21_0043584T_MSS'], index_unique = '-')
#integrated.obs['Doublet'] = integrated.obs['predicted_doublet'].astype(str).astype('category')
#integrated.obs[['Doublet', 'batch']].value_counts()
#del integrated.obs['predicted_doublet']

sc.pp.filter_genes(integrated, min_cells=5) # 'n_cells' added in integrated.var 
integrated.layers["counts"] = integrated.X.copy()
integrated.raw = integrated
'''