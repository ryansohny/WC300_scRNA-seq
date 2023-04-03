## Temporarily,
## (in cm03) conda activate 
## ipython --profile=wc300_scrna

from anndata import AnnData
import anndata
from scipy import sparse, io
import scipy
import pandas as pd
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
import scrublet as scr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings(action='ignore')
# Back to warning: warnings.filterwarnings(action='default')
sns.set(font="Arial", font_scale=1, style='ticks')
sc.settings.verbosity = 3
plt.rcParams['figure.figsize'] = (6,6)
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'Arial'
plt.rc("axes.spines", top=False, right=False)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#104e8b", "#ffdab9", "#8b0a50"])
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

# Doublet Detection
for sample in [S21_0043584T_MSS, S21_0047298T_MSS, S21_0048032T_MSS, S21_0046021T_MSI, S21_0048036T_MSI, S21_0049142T_MSI]:
    sce.pp.scrublet(sample, adata_sim=None, sim_doublet_ratio=2.0, expected_doublet_rate=0.08, stdev_doublet_rate=0.02, synthetic_doublet_umi_subsampling=1.0, knn_dist_metric='euclidean', n_prin_comps=30, verbose=True) # n_neighbors = None => automatically set to np.round(0.5 * np.sqrt(n_obs))


# First Integrate and then execute QC (Tumor Only)
integrated = AnnData.concatenate(S21_0043584T_MSS, S21_0047298T_MSS, S21_0048032T_MSS, S21_0046021T_MSI, S21_0048036T_MSI, S21_0049142T_MSI, join='outer', batch_key='Sample', batch_categories = ['S21_0043584T_MSS', 'S21_0047298T_MSS', 'S21_0048032T_MSS', 'S21_0046021T_MSI', 'S21_0048036T_MSI', 'S21_0049142T_MSI'], index_unique = '-')
integrated.obs['predicted_doublet_cat'] = integrated.obs['predicted_doublet'].astype(str).astype('category')

test = integrated.obs[['Sample']].copy()
result = test.merge(clinic_info, left_on = "Sample", right_index = True)
integrated.obs[list(result.columns)] = result
del test, result

sc.pp.calculate_qc_metrics(integrated, inplace=True)

sc.pl.violin(integrated, 'total_counts', groupby='Sample', log=True, size=2, cut=0)
sc.pl.violin(integrated, 'percent_mito', groupby='Sample', size=2, cut=0)
sc.pl.scatter(integrated, 'total_counts', 'n_genes_by_counts', color='Sample', size=20)
sc.pl.scatter(integrated, 'total_counts', 'n_genes_by_counts', color='percent_mito', size=20)

sns.histplot(data=integrated[integrated.obs['total_counts'] < 25000].obs, x='total_counts', kde=True, hue='Sample')

'''
우선, Trans/Epi팀 분석대로, 이건 나중에 고쳐야 함.
'''
sc.pp.filter_genes(integrated, min_cells=3)
sc.pp.filter_cells(integrated, min_genes=200)
sc.pp.filter_cells(integrated, max_genes=8000)
integrated = integrated[integrated.obs['percent_mito'] < 0.25] # 20,145 cells (only S21_0043584N_MSS & S21_0043584T_MSS, 2023-03-23)
integrated.layers["counts"] = integrated.X.copy()

# Fork.1
sc.pp.normalize_total(integrated, target_sum=1e4)

# Fork.1_rest
sc.pp.log1p(integrated)
sc.pp.highly_variable_genes(integrated, min_mean=0.0125, max_mean=3, min_disp=0.5)
integrated.raw = integrated
integrated.var['highly_variable'].value_counts() # 3,783 (only S21_0043584N_MSS & S21_0043584T_MSS, 2023-03-23)
sc.pp.scale(integrated, max_value=10)
sc.tl.pca(integrated, n_comps=100, use_highly_variable=True, svd_solver='arpack')
sc.pl.pca(integrated, color=['Sample'], legend_loc='right margin', size=8, add_outline=False, color_map='CMRmap', annotate_var_explained=True, components=['1,2'])
sc.pl.pca_variance_ratio(integrated, n_pcs=100, log=False)

'''
sc.pp.neighbors(integrated, n_neighbors=10, n_pcs=15) # batch correction 능력 확인
sc.tl.umap(integrated, min_dist=0.5, spread=1.0, n_components=2, alpha=1.0, gamma=1.0, init_pos='spectral', method='umap')
sc.pl.umap(integrated[rand_is, :], color=['Sample'], add_outline=False, legend_loc='right margin', size=20)
'''

cell_cycle_genes=[x.strip()[0] + x.strip()[1:].upper() for x in open("/data/Projects/phenomata/01.Projects/11.Vascular_Aging/Database/regev_lab_cell_cycle_genes.txt")]
s_genes= cell_cycle_genes[:43]
g2m_genes= cell_cycle_genes[43:]
cell_cycle_genes = [x for x in cell_cycle_genes if x in integrated.var_names]
sc.tl.score_genes_cell_cycle(integrated, s_genes=s_genes, g2m_genes=g2m_genes)

sce.pp.bbknn(integrated, batch_key='Sample', n_pcs=15, neighbors_within_batch=5, trim=None)
sc.tl.umap(integrated, min_dist=0.5, spread=1.0, n_components=2, alpha=1.0, gamma=1.0, init_pos='spectral', method='umap')

'''
from pylab import *

colormap = cm.get_cmap('Set3', 6)

for i in range(colormap.N):
    rgba = colormap(i)
    # rgb2hex accepts rgb or rgba
    print(matplotlib.colors.rgb2hex(rgba))
'''

integrated.uns['Sample_colors'] = ['#8dd3c7', '#bebada', '#80b1d3', '#fccde5', '#bc80bd', '#ffed6f']

np.random.seed(42)
rand_is = np.random.permutation(list(range(integrated.shape[0])))
sc.pl.umap(integrated[rand_is, :], color=['Sample'], add_outline=False, legend_loc='right margin', size=15)

'''
fig, axes = plt.subplots(1,2)
sc.pl.umap(integrated[rand_is, :], color=['Sample'], add_outline=False, legend_loc=None, size=20, groups=['S21_0046021T_MSI'], title='S21_0046021T_MSI', palette='Set3', ax=axes[0])
sc.pl.umap(integrated[rand_is, :], color=['Sample'], add_outline=False, legend_loc=None, size=20, groups=['S21_0043584T_MSS'], title='S21_0043584T_MSS', palette='Set3', ax=axes[1])
'''

fig, axes = plt.subplots(2,4, figsize=(20,10))
sc.pl.umap(integrated[rand_is, :], color=['CD8A'], add_outline=False, legend_loc=None, size=20, title='CD8+ T lymphocyte\nCD8A' , ax=axes[0][0])
sc.pl.umap(integrated[rand_is, :], color=['IL7R'], add_outline=False, legend_loc=None, size=20, title='CD4+ T lymphocyte\nIL7R', ax=axes[0][1])
sc.pl.umap(integrated[rand_is, :], color=['MS4A1'], add_outline=False, legend_loc=None, size=20, title='B lymphocyte\nMS4A1', ax=axes[0][2])
sc.pl.umap(integrated[rand_is, :], color=['FCGR3A'], add_outline=False, legend_loc=None, size=20, title='Natural Killer Cell\nFCGR3A', ax=axes[0][3])
sc.pl.umap(integrated[rand_is, :], color=['CD14'], add_outline=False, legend_loc=None, size=20, title='Monocyte\nCD14', ax=axes[1][0])
sc.pl.umap(integrated[rand_is, :], color=['FCER1A'], add_outline=False, legend_loc=None, size=20, title='Dendritic cell\nFCER1A', ax=axes[1][1])
sc.pl.umap(integrated[rand_is, :], color=['EPCAM'], add_outline=False, legend_loc=None, size=20, title='Epithelial Cell\nEPCAM', ax=axes[1][2])
sc.pl.umap(integrated[rand_is, :], color=['PECAM1'], add_outline=False, legend_loc=None, size=20, title='Endothelial Cell\nPECAM1', ax=axes[1][3])

sc.tl.leiden(integrated, resolution=0.5, key_added='leiden_r05') #### 0 ~  ==> 2023-03-23
sc.tl.leiden(integrated, resolution=1.0, key_added='leiden_r10')
sc.pl.umap(integrated[rand_is, :], color=['leiden_r05', 'leiden_r10'], add_outline=False, legend_loc='on data', size=20)

ax = sc.pl.correlation_matrix(integrated, groupby='leiden_r05', show_correlation_numbers=True, dendrogram=True, ax=None, vmin=-1, vmax=1)
ax = sc.pl.correlation_matrix(integrated, groupby='leiden_r10', show_correlation_numbers=True, dendrogram=True, ax=None, vmin=-1, vmax=1)

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


'''
Scran pooling normalization 이용
'''
# Fork.2
adata_pp = integrated.copy()

'''
여기에서는 integrated에 cell/gene filtering and doublet removal이외에 어떤 변형도 가해지지 않은 상황에서이다.
우선 2023-03-28에서는 normalize와 scaling이 들어간것이 adata_pp.X에 들어가 있으므로, 아래와 같이 수행한 다음 진행한다. 
adata_pp.X = adata_pp.layers['counts'].copy()
'''

sc.pp.normalize_per_cell(adata_pp, counts_per_cell_after=1e6) ## Default
sc.pp.normalize_per_cell(adata_pp, counts_per_cell_after=1e4) ## 일단은 이렇게 (2023-03-28)
sc.pp.log1p(adata_pp)
sc.tl.pca(adata_pp, n_comps=15) ## 여기서 이 n_component의 숫자를 늘리면 size_factors를 estimation하는 데 도움이 될까?
sc.pp.neighbors(adata_pp)
sc.tl.leiden(adata_pp, key_added='groups', resolution=0.5)

# Scran Normalization
test = anndata.AnnData(X=adata_pp.layers['counts'], obs=adata_pp.obs[['groups']], var=adata_pp.var[['gene_ids', 'feature_types', 'genome']])
test.write(filename="/mnt/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/test.h5ad") # Move this to bdcm03 and source activate scanpy_1.9.1

'''From bdcm03 (scanpy_1.9.1 conda environment)'''
from scipy import io
import scanpy as sc
# bdcm03에서 %paste로 하면 이 ERROR 뜸 => TclError: no display name and no $DISPLAY environment variable
#io.mmwrite('/mnt/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/test.mtx', adata_pp.layers['counts'])
test = sc.read_h5ad("test.h5ad")
io.mmwrite('/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/test.mtx', test.X)
cell_meta = test.obs[['groups']].copy()
cell_meta['Barcode'] = cell_meta.index
#cell_meta['UMAP1'] = adata_pp.obsm['X_umap'][:,0]
#cell_meta['UMAP2'] = adata_pp.obsm['X_umap'][:,1]
gene_meta = test.var.copy()
gene_meta['GeneName'] = gene_meta.index
cell_meta.to_csv('/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/counts_cellMeta.csv',index=None)
gene_meta.to_csv('/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/counts_geneMeta.csv',index=None)

'''From bdcm03 (scRNA_R conda environment) for calculating size factor for each cell using scran normalization
https://medium.com/@daimin0514/how-to-convert-singlecellexperiment-to-anndata-8ec678b3a99e
library(Seurat)
library(Matrix)
counts <- readMM('/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/test.mtx')
#dim(counts)
cellMeta<-read.csv('/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/counts_cellMeta.csv')
geneMeta<-read.csv('/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/counts_geneMeta.csv')
rownames(counts) <- cellMeta$Barcode
colnames(counts) <- geneMeta$GeneName
seurat_obj <- CreateSeuratObject(counts = t(counts), min.cells = 0, min.features = 0, names.field = 1, names.delim = "_")
seurat_obj@meta.data <- cbind(cellMeta, seurat_obj@meta.data)
sce_obj <- as.SingleCellExperiment(seurat_obj)

library(scran)
library(BiocParallel)
sce_obj <- computeSumFactors(sce_obj, clusters = sce_obj$groups, min.mean=0.1, assay.type = "counts", BPPARAM = MulticoreParam(50))
size_factors_df <- data.frame("size_factors" = sizeFactors(sce_obj), row.names = colnames(sce_obj))
write.csv(size_factors_df, "/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/test_size_factors.csv")
'''

'''Now Back to cm03 (scanpy_1.9.3 conda environment)'''
size_factors = pd.read_csv('/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/test_size_factors.csv', index_col=0)
del adata_pp
integrated.obs['size_factors'] = size_factors

integrated.X /= integrated.obs['size_factors'].values[:, None]
integrated.X = scipy.sparse.csr_matrix(integrated.X) #왜 이게 새로 들어가야될까????? # 아니면 ERRROR 남 (highly_variable_genes에서)

integrated.layers['scran'] = integrated.X

# Fork.2_rest
sc.pp.log1p(integrated) # works on anndata.X
integrated.layers['scran_log1p'] = integrated.X
integrated.raw = integrated

sc.pp.highly_variable_genes(integrated)
integrated.var['highly_variable'].value_counts() # 3,867 (2023-03-28 18:00)

sc.pp.scale(integrated, max_value=10) # tabula muris senis default (2021-08-10) # mean and std on adata.var
#sc.pp.scale(test3, zero_center=True, max_value=10, copy=False, layer=None, obsm=None)

'''
cell_cycle_genes=[x.strip()[0] + x.strip()[1:].upper() for x in open("/data/Projects/phenomata/01.Projects/11.Vascular_Aging/Database/regev_lab_cell_cycle_genes.txt")]
s_genes= cell_cycle_genes[:43]
g2m_genes= cell_cycle_genes[43:]
cell_cycle_genes = [x for x in cell_cycle_genes if x in test3.var_names]
sc.tl.score_genes_cell_cycle(test3, s_genes=s_genes, g2m_genes=g2m_genes)
'''

sc.tl.pca(integrated, n_comps=100, use_highly_variable=True, svd_solver='arpack')
sc.pl.pca(integrated, color=['Sample'], legend_loc='right margin', size=8, add_outline=False, color_map='CMRmap', annotate_var_explained=True, components=['1,2'])
sc.pl.pca_variance_ratio(integrated, n_pcs=100, log=False)

sce.pp.bbknn(integrated, batch_key='Sample', n_pcs=15, neighbors_within_batch=5, trim=None)
sc.tl.umap(integrated, min_dist=0.5, spread=1.0, n_components=2, alpha=1.0, gamma=1.0, init_pos='spectral', method='umap')

'''
from pylab import *

colormap = cm.get_cmap('Set3', 6)

for i in range(colormap.N):
    rgba = colormap(i)
    # rgb2hex accepts rgb or rgba
    print(matplotlib.colors.rgb2hex(rgba))
'''

integrated.uns['Sample_colors'] = ['#8dd3c7', '#bebada', '#80b1d3', '#fccde5', '#bc80bd', '#ffed6f']

np.random.seed(42)
rand_is = np.random.permutation(list(range(integrated.shape[0])))
sc.pl.umap(integrated[rand_is, :], color=['Sample'], add_outline=False, legend_loc='right margin', size=15)

fig, axes = plt.subplots(3,3, figsize=(18, 10))
sc.pl.umap(integrated[rand_is, :], color=['Sample'], add_outline=False, legend_loc=None, size=20, groups=['S21_0046021T_MSI'], title='S21_0046021T_MSI', palette='Set3', ax=axes[0])
sc.pl.umap(integrated[rand_is, :], color=['Sample'], add_outline=False, legend_loc=None, size=20, groups=['S21_0043584T_MSS'], title='S21_0043584T_MSS', palette='Set3', ax=axes[1])

'''
fig, axes = plt.subplots(1,2)
sc.pl.umap(integrated[rand_is, :], color=['Sample'], add_outline=False, legend_loc=None, size=20, groups=['S21_0046021T_MSI'], title='S21_0046021T_MSI', palette='Set3', ax=axes[0])
sc.pl.umap(integrated[rand_is, :], color=['Sample'], add_outline=False, legend_loc=None, size=20, groups=['S21_0043584T_MSS'], title='S21_0043584T_MSS', palette='Set3', ax=axes[1])
'''

fig, axes = plt.subplots(3,4, figsize=(20,15))
sc.pl.umap(integrated[rand_is, :], color=['CD8A'], add_outline=False, legend_loc=None, size=20, title='CD8+ T lymphocyte\nCD8A' , ax=axes[0][0])
sc.pl.umap(integrated[rand_is, :], color=['IL7R'], add_outline=False, legend_loc=None, size=20, title='CD4+ T lymphocyte\nIL7R', ax=axes[0][1])
sc.pl.umap(integrated[rand_is, :], color=['MS4A1'], add_outline=False, legend_loc=None, size=20, title='B lymphocyte\nMS4A1', ax=axes[0][2])
sc.pl.umap(integrated[rand_is, :], color=['FCGR3A'], add_outline=False, legend_loc=None, size=20, title='Natural Killer Cell\nFCGR3A', ax=axes[0][3])
sc.pl.umap(integrated[rand_is, :], color=['CD14'], add_outline=False, legend_loc=None, size=20, title='Monocyte\nCD14', ax=axes[1][0])
sc.pl.umap(integrated[rand_is, :], color=['FCER1A'], add_outline=False, legend_loc=None, size=20, title='Dendritic cell\nFCER1A', ax=axes[1][1])
sc.pl.umap(integrated[rand_is, :], color=['EPCAM'], add_outline=False, legend_loc=None, size=20, title='Epithelial Cell\nEPCAM', ax=axes[1][2])
sc.pl.umap(integrated[rand_is, :], color=['PECAM1'], add_outline=False, legend_loc=None, size=20, title='Endothelial Cell\nPECAM1', ax=axes[1][3])
sc.pl.umap(integrated[rand_is, :], color=['COL1A1'], add_outline=False, legend_loc=None, size=20, title='Fibroblast\nCOL1A1', ax=axes[2][0])

sc.tl.leiden(integrated, resolution=0.5, key_added='leiden_r05') #### 0 ~  ==> 2023-03-23
sc.tl.leiden(integrated, resolution=1.0, key_added='leiden_r10')
sc.pl.umap(integrated[rand_is, :], color=['leiden_r05', 'leiden_r10'], add_outline=False, legend_loc='on data', size=20)

ax = sc.pl.correlation_matrix(integrated, groupby='leiden_r05', show_correlation_numbers=True, dendrogram=True, ax=None, vmin=-1, vmax=1)
ax = sc.pl.correlation_matrix(integrated, groupby='leiden_r10', show_correlation_numbers=True, dendrogram=True, ax=None, vmin=-1, vmax=1)

# SingleR cell-type annotation
# 위에 scanpy에서 size_factors 가지고 나눴던 것 다시 해야 할 것 같음.
''' From bdcm03 (scRNA_R2 conda environment)
suppressMessages(library(celldex))
suppressMessages(library(SingleR))
suppressMessages(library(scran))
suppressMessages(library(Seurat))
suppressMessages(library(Matrix))
suppressMessages(library(BiocParallel))

ref.data <- celldex::HumanPrimaryCellAtlasData(ensembl=FALSE) # Human Primary Cell Atlas Data

# ref.data.ensembl <- celldex::HumanPrimaryCellAtlasData(ensembl=TRUE) # Human Primary Cell Atlas Data
# bpe.ensembl <- celldex::BlueprintEncodeData(ensembl=TRUE) # Blueprint ENCODE
# bpe <- celldex::BlueprintEncodeData(ensembl=FALSE) # Blueprint ENCODE

# https://medium.com/@daimin0514/how-to-convert-singlecellexperiment-to-anndata-8ec678b3a99e

counts <- readMM('/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/test.mtx')
#dim(counts)
cellMeta<-read.csv('/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/counts_cellMeta.csv')
geneMeta<-read.csv('/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/counts_geneMeta.csv')
rownames(counts) <- cellMeta$Barcode
colnames(counts) <- geneMeta$GeneName
seurat_obj <- CreateSeuratObject(counts = t(counts), min.cells = 0, min.features = 0, names.field = 1, names.delim = "_")
seurat_obj@meta.data <- cbind(cellMeta, seurat_obj@meta.data)
sce_obj <- as.SingleCellExperiment(seurat_obj)
sce_obj <- computeSumFactors(sce_obj, clusters = sce_obj$groups, min.mean=0.1, assay.type = "counts", BPPARAM = MulticoreParam(50))
#size_factors_df <- data.frame("size_factors" = sizeFactors(sce_obj), row.names = colnames(sce_obj))

sce_obj <- logNormCounts(sce_obj, log=FALSE) # adds normcounts in assays
temp_norm <- log(assays(sce_obj)$normcounts +1)
temp_norm <- as(temp_norm, "dgCMatrix")
assays(sce_obj)$lognormcounts <- temp_norm

# Counts file
writeMM(assays(sce_obj)$normcounts, '/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/test_scran.mtx')
writeMM(assays(sce_obj)$lognormcounts, '/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/test_scranlog.mtx')

barcodes <- data.frame(colnames(sce_obj))
colnames(barcodes) <- 'Barcode'
#write.csv(barcodes, '/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/test_scran_barcodes.csv',
          quote = FALSE, row.names = FALSE) # Not necessary (duplicates)
write.csv(barcodes, '/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/test_scranlog_barcodes.csv',
          quote = FALSE, row.names = FALSE)
genes <- data.frame(rownames(sce_obj))
colnames(genes) <- 'Gene'
#write.csv(genes, '/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/test_scran_genes.csv',
          quote = FALSE,row.names = FALSE) # Not necessary (duplicates)
write.csv(genes, '/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/test_scranlog_genes.csv',
          quote = FALSE,row.names = FALSE)
size_factors_df <- data.frame("size_factors" = sizeFactors(sce_obj), row.names = colnames(sce_obj))
write.csv(size_factors_df, "/mnt/mone/Project/WC300/07.scRNA-seq/Scran_normalization/test_size_factors.csv")

# For metadata information (Omitted this time)
#cellMeta <- seo@meta.data
#write.csv(cellMeta, '/home/daimin/data/single_cell/mouse_brain_vmh/SMART-seq_VMH_cpm_cellMeta.csv',
#          quote = FALSE,row.names = TRUE)
'''

'''Back in scanpy_1.9.3 conda environment (cm03)'''
# integrated의 layers (또는 anndata.X)에 scran을 집어넣는다.
from scipy import sparse, io
counts = io.mmread('/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/test_scranlog.mtx') # 얘는 SCE file convention을 따른다.
counts = sparse.csr_matrix(counts)
barcodes = pd.read_csv('/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/test_scranlog_barcodes.csv')
genes = pd.read_csv('/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/test_scranlog_genes.csv')
adata = sc.AnnData(counts.T) # Therefore, we need to transpose the count matrix
adata.obs_names=barcodes['Barcode'].values
adata.var_names=genes['Gene'].values

integrated.layers['scran_log1p'] = adata.X
integrated.X = integrated.layers['scran_log1p']
# integrated.uns['log1p']["base"] = None # scran normalized counts와 log transforming을 R에서 했으므로, sc.pp.log1p가 안들어가서 이걸 set하는게 필요할 수 있다.
del adata, counts, barcodes, genes

integrated.raw = integrated

sc.pp.highly_variable_genes(integrated)
integrated.var['highly_variable'].value_counts() # 3,867 (2023-03-28 18:00)

sc.pp.scale(integrated, max_value=10) # tabula muris senis default (2021-08-10) # mean and std on adata.var
#sc.pp.scale(test3, zero_center=True, max_value=10, copy=False, layer=None, obsm=None)

cell_cycle_genes=[x.strip()[0] + x.strip()[1:].upper() for x in open("/data/Projects/phenomata/01.Projects/11.Vascular_Aging/Database/regev_lab_cell_cycle_genes.txt")]
s_genes= cell_cycle_genes[:43]
g2m_genes= cell_cycle_genes[43:]
cell_cycle_genes = [x for x in cell_cycle_genes if x in integrated.var_names]
sc.tl.score_genes_cell_cycle(integrated, s_genes=s_genes, g2m_genes=g2m_genes)


sc.tl.pca(integrated, n_comps=100, use_highly_variable=True, svd_solver='arpack')
sc.pl.pca(integrated, color=['Sample'], legend_loc='right margin', size=8, add_outline=False, color_map='CMRmap', annotate_var_explained=True, components=['1,2'])
sc.pl.pca_variance_ratio(integrated, n_pcs=100, log=False)

sce.pp.bbknn(integrated, batch_key='Sample', n_pcs=15, neighbors_within_batch=5, trim=None)
sc.tl.umap(integrated, min_dist=0.5, spread=1.0, n_components=2, alpha=1.0, gamma=1.0, init_pos='spectral', method='umap')
sc.tl.draw_graph(integrated, layout='fa')

integrated.uns['Sample_colors'] = ['#8dd3c7', '#bebada', '#80b1d3', '#fccde5', '#bc80bd', '#ffed6f']

np.random.seed(42)
rand_is = np.random.permutation(list(range(integrated.shape[0])))
sc.pl.umap(integrated[rand_is, :], add_outline=False, legend_loc='right margin', size=15, color=['Sample'])
sc.pl.draw_graph(integrated[rand_is, :], add_outline=False, legend_loc='right margin', size=15, color=['Sample'])

# Cell cycle phase (Checking for cell cycle effect)
fig, axes = plt.subplots(1,3, figsize=(18, 5.5))
sc.pl.umap(integrated[rand_is, :], color=['phase'], add_outline=False, legend_loc=None, size=20, groups=['G1'], title='G1', ax=axes[0])
sc.pl.umap(integrated[rand_is, :], color=['phase'], add_outline=False, legend_loc=None, size=20, groups=['G2M'], title='G2M', ax=axes[1])
sc.pl.umap(integrated[rand_is, :], color=['phase'], add_outline=False, legend_loc=None, size=20, groups=['S'], title='S', ax=axes[2])

sc.tl.leiden(integrated, resolution=0.5, key_added='leiden_r05') #### 0 ~  ==> 2023-03-23
sc.tl.leiden(integrated, resolution=1.0, key_added='leiden_r10')
sc.pl.umap(integrated[rand_is, :], color=['leiden_r05', 'leiden_r10'], add_outline=False, legend_loc='on data', size=20)

ax = sc.pl.correlation_matrix(integrated, groupby='leiden_r05', show_correlation_numbers=True, dendrogram=True, ax=None, vmin=-1, vmax=1)
ax = sc.pl.correlation_matrix(integrated, groupby='leiden_r10', show_correlation_numbers=True, dendrogram=True, ax=None, vmin=-1, vmax=1)

# UMAP projection with Microsatellite instability
fig, axes = plt.subplots(1,2, figsize=(12, 5.5))
sc.pl.umap(integrated[rand_is, :], color=['phase'], add_outline=False, legend_loc=None, size=20, groups=['G1'], title='G1', ax=axes[0])
sc.pl.umap(integrated[rand_is, :], color=['phase'], add_outline=False, legend_loc=None, size=20, groups=['G2M'], title='G2M', ax=axes[1])

# UMAP projection with Fraction of mtRNA expression
sc.pl.umap(integrated[rand_is, :], color=['percent_mito'], add_outline=False, legend_loc=None, size=20, color_map=cmap, title='Fraction of mtRNA expression')

cell_meta = integrated.obs[['leiden_r05']].copy()
cell_meta['Barcode'] = cell_meta.index
cell_meta.to_csv('/mnt/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/integrated_leiden_r05_cellMeta.csv',index=None)

# UMAP projection with MSI status
fig, axes = plt.subplots(2,1, figsize=(5,10.5))
sc.pl.umap(integrated[rand_is, :], color=['MSI_type'], groups = 'MSI', add_outline=False, legend_loc=None, size=10, title='MSI type', palette = {'MSI':'#5F4B8BFF', 'MSS': '#E69A8DFF'}, ax=axes[0])
sc.pl.umap(integrated[rand_is, :], color=['MSI_type'], groups = 'MSS', add_outline=False, legend_loc=None, size=10, title='MSS type', palette = {'MSI':'#5F4B8BFF', 'MSS': '#E69A8DFF'}, ax=axes[1])

'''
### SingleR execution
# Adding Leiden clustering information obtained from Scanpy into sce_obj
leiden_df <- read.csv("/mnt/mone/Project/WC300/07.scRNA-seq/SingleR/integrated_leiden_r05_cellMeta.csv")
rownames(leiden_df) <- leiden_df$Barcode
leiden_df$Barcode <- NULL
colData(sce_obj) <- cbind(colData(sce_obj), leiden_df)
sce_obj$leiden_r05 <- as.character(sce_obj$leiden_r05) # Maybe not necessary (2023-03-31 17:06)


# rm(counts)
# rm(cellMeta)
# rm(geneMeta)
# rm(seurat_obj)
# rm(temp_norm)


# test 1.
#predictions.lognormcounts.ref.data <- SingleR(test = sce_obj, assay.type.test = 'lognormcounts', ref = ref.data, labels = ref.data$label.fine)
#predictions.lognormcounts.bpe <- SingleR(test = sce_obj, assay.type.test = 'lognormcounts', ref = bpe, labels = bpe$label.fine)

# test 2.
predictions.lognormcounts.clusters.ref.data <- SingleR(test = sce_obj, assay.type.test = 'lognormcounts', ref = ref.data, labels = ref.data$label.fine, clusters = sce_obj$leiden_r05)
predictions.lognormcounts.clusters.bpe <- SingleR(test = sce_obj, assay.type.test = 'lognormcounts', ref = bpe, labels = bpe$label.fine, clusters = sce_obj$leiden_r05)


plotScoreHeatmap(predictions.lognormcounts.clusters.ref.data, labels.use = predictions.lognormcounts.clusters.ref.data$pruned.labels)
dev.off()


'''


'''SlingShot execution'''
cell_meta = integrated.obs[['phase']].copy()
cell_meta['Barcode'] = cell_meta.index
cell_meta['UMAP1'] = integrated.obsm['X_umap'][:,0]
cell_meta['UMAP2'] = integrated.obsm['X_umap'][:,1]
cell_meta.to_csv('/mnt/data/Projects/phenomata/01.Projects/97.Others/WC300_scRNA/cellMeta_for_SlingShot.csv', index = None)

'''Slingshot in R
cellMeta_slingshot <- read.csv('/mnt/mone/Project/WC300/07.scRNA-seq/SlingShot/cellMeta_for_SlingShot.csv')
rownames(cellMeta_slingshot) <- cellMeta_slingshot$Barcode
cellMeta_slingshot$Barcode <- NULL

reducedDim(sce_obj, "UMAP") <- cellMeta_slingshot[, c('UMAP1', 'UMAP2')]
sce_obj <- slingshot::slingshot(sce_obj, clusterLabels = 'leiden_r05', reducedDim = 'UMAP')
# PseudotimeOrdering Class Added to colData(sce_obj)$slingshot
# colData(sce_obj)$slingshot
# class: PseudotimeOrdering 
# dim: 51387 5 
# metadata(4): lineages mst slingParams curves
# pathStats(2): pseudotime weights
# cellnames(51387): AAACCCAAGGCCATAG-1-S21_0043584T_MSS
#   AAACCCAAGTTACGAA-1-S21_0043584T_MSS ...
#   TTTGTTGTCTCCTACG-1-S21_0049142T_MSI
#   TTTGTTGTCTGAGAAA-1-S21_0049142T_MSI
# cellData names(2): reducedDim clusterLabels
# pathnames(5): Lineage1 Lineage2 Lineage3 Lineage4 Lineage5
# pathData names(0):
plot(reducedDim(sce_obj), pch = 16, cex = 0.5)
lines(slingshot::SlingshotDataSet(sce_obj), lwd = 2, type = 'lineages', col = 'gray')
dev.off()

'''