import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from utils import *
import re


#load .csv data and return .h5ad files
def csv2h5ad(data_path,pattern,split=False):
    for file in os.listdir(data_path):
        name, ext = os.path.splitext(file)
        
        if re.match(pattern,file):
            split_file = name.split('_')
            data_temp = pd.read_csv(os.path.join(data_path, file),delimiter=',')
            feat_cols = data_temp.columns[3:]
            data_temp = data_temp[data_temp[feat_cols].sum(axis=1)!=0]
            nodefeats = data_temp[feat_cols]
            nodefeats = np.array(nodefeats)
            data_temp['obsindex'] = data_temp['x'].astype(str)+"x"+data_temp['y'].astype(str)
            xs=data_temp["x"]
            ys=data_temp["y"]
            xs=np.array(xs)
            ys=np.array(ys)
            pos=np.stack((xs,ys), axis=-1)
            adata=sc.AnnData(X=nodefeats,dtype='float32')
            obs=data_temp.set_index('obsindex')[['region']]
            adata.obs=obs
            adata.obsm['spatial'] = pos
            varindex = pd.DataFrame(index=data_temp[feat_cols].columns)
            adata.var = varindex
            adata.raw = adata
            
            adata.write(os.path.join(os.path.join(data_path,'h5ad'),f'{split_file[0]}_{split_file[1]}.h5ad'))

            if split==True:
                unique_id=data_temp['region'].unique().tolist()
                for i in range(len(unique_id)):
                    adata_sub=adata[adata.obs['region']==unique_id[i]].copy()
                    adata_sub.write(os.path.join(os.path.join(data_path,'h5ad'),f'{unique_id[i]}_{split_file[1]}.h5ad'))


                    
#merge omics data by coordinates and return .h5ad file
def pooldata(data_path,pattern,split=False):
    adata_dict={}
    for file in os.listdir(data_path):
        name, ext = os.path.splitext(file)
        
        if re.match(pattern,file):
            split_file = name.split('_')
            print(split_file)
            data_temp = pd.read_csv(os.path.join(data_path, file),delimiter=',')
            feat_cols = data_temp.columns[3:]
            data_temp = data_temp[data_temp[feat_cols].sum(axis=1)!=0]
            nodefeats = data_temp[feat_cols]
            nodefeats = np.array(nodefeats)
            data_temp['obsindex'] = data_temp['x'].astype(str)+"x"+data_temp['y'].astype(str)

            adata_temp=sc.AnnData(X=nodefeats,dtype='float32')
            obsindex = pd.DataFrame(index=data_temp['obsindex'])
            adata_temp.obs = obsindex
            varindex = pd.DataFrame(index=data_temp[feat_cols].columns)
            adata_temp.var = varindex
            adata_temp.raw = adata_temp

            adata_dict[split_file[1]]=adata_temp

    #remove duplicate columns between metabolomics and lipidomics from metabolomics
    common_elements = list(set(adata_dict['metabolomics'].var.index.to_list()).intersection(adata_dict['lipidomics'].var.index.to_list()))
    adata_dict['metabolomics']=adata_dict['metabolomics'][:,~adata_dict['metabolomics'].var.index.isin(common_elements)]

    #merge dataset
    adata=ad.concat(adata_dict,join='inner',axis=1,label='omics')

    #reset annotation
    obs=data_temp.set_index('obsindex')[['region']]
    obs=obs[obs.index.isin(adata.obs.index)]
    adata.obs=obs
    adata.obsm['spatial']=adata.obs.reset_index()['obsindex'].str.split('x',expand=True).astype(np.float64).to_numpy()
    
    if split==True:
        unique_id=data_temp['region'].unique().tolist()
        for i in range(len(unique_id)):
            adata_sub=adata[adata.obs['region']==unique_id[i]].copy()
            adata_sub.write(os.path.join(os.path.join(data_path,"h5ad"),f'{unique_id[i]}_pool.h5ad'))
        
    adata.write(os.path.join(os.path.join(data_path,"h5ad"),f'{split_file[0]}_pool.h5ad'))
    


