def test_load():
  return 'loaded'


def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]


def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + 0.01


def cond_probs_product(table, evidence_row, target, target_value):
  #your function body below
  table_columns = up_list_column_names(table)
  evidence_columns = table_columns[0:-1]
  evidence_complete = up_zip_lists(evidence_columns, evidence_row)
  cond_prob_list = [cond_prob(table, i, j, target, target_value) for i, j in evidence_complete]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator


def prior_prob(table, target, target_value):
  t_list = up_get_column(table, target)
  p_a = sum([1 if v==target_value else 0 for v in t_list])/len(t_list)
  return p_a


def naive_bayes(table, evidence_row, target):
  #compute P(Flu=0|...) by collecting cond_probs in a list, take the product of the list, finally multiply by P(Flu=0)
  target_value = 0
  num_0 = cond_probs_product(table, evidence_row, target, target_value) * prior_prob(table, target, target_value)
  #do same for P(Flu=1|...)
  target_value = 1
  num_1 = cond_probs_product(table, evidence_row, target, target_value) * prior_prob(table, target, target_value)
  #Use compute_probs to get 2 probabilities
  neg, pos = compute_probs(num_0, num_1)
  #return your 2 results in a list
  return [neg, pos]


def metrics(zipped_list):
  #asserts here
  assert isinstance(zipped_list, list), f"zipped_list is not a list"
  assert all(isinstance(a_list, list) for a_list in zipped_list), f"not a list of lists"
  assert all(len(a_list) == 2 for a_list in zipped_list), f"not a list of pairs"
  for a,b in zipped_list:
   assert isinstance(a,(int,float)) and isinstance(b,(int,float)), f'zipped_list contains a non-int or non-float pair: {[a,b]}'
  for a,b in zipped_list:
   assert float(a) in [0.0,1.0] and float(b) in [0.0,1.0], f'zipped_list contains a non-binary pair: {[a,b]}'
  # assert all(isinstance(item, int) and item >= 0 for a_list in zipped_list for item in a_list), f'item in pair in zipped_list is not an integer, or it is not >= 0 '
  #body of function below
  predictions, labels = zip(*zipped_list)
  all_cases = len(predictions)

  #first compute the sum of all 4 cases. See code above
  tp = sum([p == 1 and l == 1 for p, l in zipped_list])
  fn = sum([p == 0 and l == 1 for p, l in zipped_list])
  fp = sum([p == 1 and l == 0 for p, l in zipped_list])
  tn = sum([p == 0 and l == 0 for p, l in zipped_list])

  #now can compute precicision, recall, f1, accuracy. Watch for divide by 0.
  precision = tp / (tp + fp) if tp + fp != 0 else 0
  recall = tp / (tp + fn) if tp + fn != 0 else 0
  f1 = 2 * precision * recall /  (precision + recall) if precision + recall != 0 else 0
  accuracy = (tp + tn) / all_cases

  #now build dictionary with the 4 measures
  dict_metrics = {'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy}
  #finally, return the dictionary
  return dict_metrics

from sklearn.ensemble import RandomForestClassifier  #make sure this makes it into your library

def run_random_forest(train, test, target, n):
  #target is target column name
  #n is number of trees to use
  X = up_drop_column(train, target)
  y = up_get_column(train, target)

  k_feature_table = up_drop_column(test, target)
  k_actuals = up_get_column(test, target)

  clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0)

  clf.fit(X, y)  #builds the trees as specified above
  probs = clf.predict_proba(k_feature_table)
  pos_probs = [p for l,p in probs]  #probs is list of [neg,pos] like we are used to seeing.

  all_mets = []
  for t in thresholds:
    all_predictions = [1 if pos>t else 0 for pos in pos_probs]
    pred_act_list = up_zip_lists(all_predictions, k_actuals)
    mets = metrics(pred_act_list)
    mets['Threshold'] = t
    all_mets = all_mets + [mets]

  metrics_table = up_metrics_table(all_mets)



  return metrics_table

def try_archs(full_table, target, architectures, thresholds):
  #target is target column name
  #split full_table
  train_table, test_table = up_train_test_split(full_table, target, .4)
  #now loop through architectures
  for arch in architectures:
    all_results = up_neural_net(train_table, test_table, arch, target)
    #loop through thresholds
    all_mets = []
    #loop through thresholds
    for t in thresholds:
      all_predictions = [1 if pos>=t else 0 for neg,pos in all_results]
      pred_act_list = up_zip_lists(all_predictions, up_get_column(test_table, target))
      mets = metrics(pred_act_list)
      mets['Threshold'] = t
      all_mets = all_mets + [mets]

    print(f'Architecture: {arch}')
    print(up_metrics_table(all_mets))

  return None  #main use is to print out threshold tables, not return anything useful.
