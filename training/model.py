from catboost import CatBoostClassifier, Pool
import pandas as pd
import argparse

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(
                    prog = 'CatboostTrainer',
                    description = 'Runs a catboost model training job')
    parser.add_argument('--trees',
                        default=20,
                        type=int, 
                        help='number of trees to build')
    parser.add_argument('--depth',
                        default=2,
                        type=int, 
                        help='tree depth')
    parser.add_argument('--learn_rate',
                        default=.03,
                        type=float, 
                        help='learning rate')
    args=parser.parse_args()
    
    # #read local copy of data
    # df = pd.read_csv('../gcp-ml-play/data/train.csv')
    
    #read local copy of data
    df = pd.read_csv('train.csv')
    
    print('data read in ...')
    
    #TODO randomize this 
    train = df.loc[df.split_seg < 8]
    test = df.loc[df.split_seg >= 8]
    
    target = 'label'

    c_features = ['os',
                  'country',
                  'campaign',
                  'traffic_source',
                  'medium',
                  'device_category',
                  'engagement_type']

    n_features = ['pageviews',
                  'visits',
                  'hits',
                  'bounces',
                  'is_mobile']

    train_X = train[n_features+c_features]
    train_data = Pool(train_X,
                      label=train[[target]],
                      cat_features=c_features,
                      feature_names=list(train_X.columns),
                      has_header=True)

    test_X = test[n_features+c_features]
    test_data = Pool(test_X,
                      label=test[[target]],
                      cat_features=c_features,
                      feature_names=list(test_X.columns),
                      has_header=True)
    print('data pools created ...')
    model = CatBoostClassifier(iterations=args.trees,
                               depth=args.depth,
                               learning_rate=args.learn_rate,
                               loss_function='Logloss',
                               verbose=True,
                               train_dir='cb_default')

    model.fit(train_data,
              eval_set=test_data,
              early_stopping_rounds=10)
    print('model trained!')