#prediction_bias.py
from statistic_functions import *
import math
import scipy as sc
from scipy import stats
import pandas as pd
    
def object_columns(df):
    return filter(lambda x: df[x].dtype == object    , df.columns )
    
def continuous_columns(df):
    
     return filter(lambda x: df[x].dtype in (int,float), df.columns )
    
    



def test_catgorical_fisher_asociation(df,significant_pval=0.8):
    
  
    def get_pval_for_single_value(val,categorical_col):
    
        tab_ocur=pd.crosstab(df["pred_state"],df[categorical_col]==val)
        if tab_ocur.shape != (2,2):
          return 1
        
        return stats.fisher_exact(tab_ocur,alternative="greater")[1]
    
    return map(lambda col: filter(lambda test_result: test_result[0]<significant_pval,  map( lambda value: (get_pval_for_single_value(value,col),col,value),  df[col].unique())),object_columns(df))
    
    
def multicategorical_significance(df,significant_correlation=0.1):
    
    return filter(lambda y :y[0]>=significant_correlation , map(lambda x: (cramers_corrected_stat(df[x],df["pred_state"]),x),object_columns(df)))

def forward_multicategorical_significance(df,significant_correlation=0.1):
    
    return filter(lambda y :y[0]>=significant_correlation , map(lambda x: (theils_u(df[x],df["pred_state"]),x),object_columns(df)))
    
def backward_multicategorical_significance(df,significant_correlation=0.1):
    
    return filter(lambda y :y[0]>=significant_correlation , map(lambda x: (theils_u(df["pred_state"],df[x]),x),object_columns(df)))
    
def continuous_significance(df, significant_pval=0.01):
   
    df=df.copy()
    df=df.replace([np.inf, -np.inf], np.nan)
    df=df.dropna(subset=list(continuous_columns(df)))
    return filter(lambda y :y[0]<=significant_pval, map(lambda x: (stats.pointbiserialr(df["pred_state"],df[x])[1],x),continuous_columns(df)))         
        
    
    


class stat_summarizer:
    def __init__(self,evaluator , legend):
        self.evaluator=evaluator
        self.legend=legend
    def __call__(self,df,**kwargs):
        print(f"On evaluating {self.evaluator.__name__}")
        self.legend(self.evaluator(df,**kwargs))
        
        pass
      
def make_per_value_summary(filter_result):
    for col_result in filter_result:
            [print("column {0} was found asociated to the target at value {2} with a significance of {1} ".format(result[1],result[0],result[2])) for result in  col_result] 
    pass
def make_column_summary(filter_result):
    
    [print("column {0} was found asociated to the target with a significance of {1} ".format(result[1],result[0])) for result in  filter_result]
    pass





def test_grouped_catgorical_fisher_asociation(df,groupby_col="question_id",significant_pval=0.8):
    
  
    def get_pval_for_single_value(val,categorical_col,groupby_col=groupby_col,df=df):
    
        df=df.copy()
        df[categorical_col]= df[categorical_col]==val
        df=df[["pred_state",categorical_col]].groupby(groupby_col).max(axis=1)
        tab_ocur=pd.crosstab(df["pred_state"],df[categorical_col])

        
        
        if tab_ocur.shape != (2,2):
              return 1
        
        return stats.fisher_exact(tab_ocur,alternative="greater")[1]
    
    return map(lambda col: filter(lambda test_result: test_result[0]<significant_pval,  map( lambda value: (get_pval_for_single_value(value,col),col,value),  df[col].unique())),object_columns(df))
    
    
    