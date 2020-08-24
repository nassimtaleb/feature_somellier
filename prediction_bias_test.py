import prediction_bias as pb
import unittest
import pandas as pd
import numpy as np
import random
np.random.seed(0)

#despues del refactor para hacer los mismos tests en los q no son sobre el target hay q agregar un filtro de este estilo



#items=[item for item in result_filter if item[1] != "pred_state"]


df_test=pd.DataFrame({"pred_state":np.random.choice([0,1],size=100),"second":np.random.normal(size=100),"third":np.random.normal(size=100),"f4th":np.random.choice(["a","b","2","3"],size=100)})

df_test["cont"]=np.random.normal(size=100)

df_test["cont2"]=np.random.normal(loc=1,size=100)

class columns_selector_test(unittest.TestCase):
   def test(self):
    self.assertTrue(set(pb.continuous_columns(df_test))==set(('pred_state', 'cont', 'cont2','second', 'third')))
    
    self.assertTrue(set(pb.object_columns(df_test))==set( ['f4th']))

  
  

class test_fisher_variable_with_itself(unittest.TestCase):
  def test(self):
    df=pd.DataFrame({"pred_state":df_test["pred_state"],"pred_state_copy":df_test["pred_state"]})
    print(df)
    result=np.array(list(pb.test_catgorical_fisher_asociation(df,"pred_state_copy")))
    print(result)
    self.assertTrue(all(result<0.01))


class test_siginificance(unittest.TestCase):
  def test(self):
    
    #self.assertTrue(pb.significance(pd.DataFrame({0:df["pred_state"],1:df["pred_state"]})))
    result=list(pb.multicategorical_significance(df_test))
    self.assertTrue(len(result)==1)           
    self.assertTrue(result[0]== (0.11110775035606506, 'f4th'))       
      
class test_continuous_significance(unittest.TestCase):
  def test(self):
   
    self.assertTrue(list(pb.continuous_significance(df_test,["cont2"],1))[0][1]=="cont2")
    
    

import io
import unittest
import unittest.mock



class TestInfoPrint(unittest.TestCase):

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def assert_stdout(self, func,df, expected_output, mock_stdout):
        func(df)
        self.assertEqual(mock_stdout.getvalue(), expected_output)

    def test_only_numbers(self):
        categorical_significance=pb.stat_summarizer(pb.multicategorical_significance, pb.make_column_summary)

      
        self.assert_stdout(categorical_significance,df_test, '\n'.join(["On evaluating multicategorical_significance" ,"column f4th was found asociated to the target with a significance of 0.11110775035606506 \n"]))import prediction_bias as pb