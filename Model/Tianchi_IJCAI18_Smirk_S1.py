# -*-coding:utf-8-*-
# Project: tianchi-IJCAI-18  
# Filename: Tianchi_IJCAI18_Smirk_S1
# Author: Smirk <smirk dot cao at gmail dot com>

# Model
from data_process import *
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import utils
import lightgbm as lgb
import warnings
import os

warnings.filterwarnings("ignore")


def load_dataset():
    # load train dataset
    logger.info('Load train dataset to all_data.')
    all_data = pd.read_csv('../Input/round1_ijcai_18_train_20180301/round1_ijcai_18_train_20180301.txt', sep=" ")
    logger.info('all_data dataset shape : %s', all_data.shape)
    logger.info("all_data data have %i columns\n %s" % (len(all_data.columns.tolist()), all_data.columns.tolist()))

    # load test_a dataset
    logger.info('load test_a dataset to test_a')
    test_a = pd.read_csv('../Input/round1_ijcai_18_test_a_20180301/round1_ijcai_18_test_a_20180301.txt', sep=" ")
    logger.info("test_a data shape : %s", test_a.shape)

    # load test_b dataset
    logger.info('load test_b dataset to test_b')
    test_b = pd.read_csv('../Input/round1_ijcai_18_test_b_20180418/round1_ijcai_18_test_b_20180418.txt', sep=" ")
    logger.info("test_b data shape : %s", test_b.shape)

    # combine test_a and test_b to test data
    logger.info('concat test_a and test_b dataset to test')
    test = test_a
    test = test.append(test_b).reset_index(drop=True)
    logger.info("test data shape : %s", test.shape)
    return all_data, test_a, test_b, test


def search_param(X_train, y_train, X_val, y_val):
    gbm = lgb.LGBMClassifier(objective='binary',
                             num_leaves=35,
                             max_depth=8,
                             learning_rate=0.01,
                             n_estimators=20000,
                             colsample_bytree=0.8,
                             subsample=0.9,
                             seed=0
                             )
    gbm_model = gbm.fit(X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=150)
    best_iter = gbm_model.best_iteration_
    logger.info("...Lightgbm modeling end...")
    logger.info(gbm.get_params)
    logger.info('Start predicting...')
    # predict
    pred = gbm_model.predict_proba(X_val)[:, 1]
    xx_analyse = pd.DataFrame()
    xx_analyse['pred'] = pred
    xx_analyse['index'] = range(len(y_val))
    xx_analyse['truth'] = y_val

    filename = pd.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    filename = outputFolder + "xx_" + filename + ".csv"
    xx_analyse.to_csv(filename, index=False)

    loss_gbm = log_loss(xx_analyse['truth'], xx_analyse["pred"])
    logger.info("log_loss : %s" % loss_gbm)
    predictors = [i for i in X_train.columns]
    feat_imp = pd.Series(gbm.feature_importances_, predictors).sort_values(ascending=False)
    logger.info("\nfeature count:%s\nfeature importance \n%s " % (feat_imp.shape[0], feat_imp))
    return best_iter, loss_gbm


def sub(X_train, y_train, X_test, best_iter):
    gbm = lgb.LGBMClassifier(objective='binary',
                             num_leaves=35,
                             max_depth=8,
                             learning_rate=0.01,
                             n_estimators=best_iter,
                             colsample_bytree=0.8,
                             subsample=0.9,
                             seed=0
                             )
    gbm_model = gbm.fit(X_train, y_train)
    logger.info("...Lightgbm modeling end...")
    logger.info(gbm.get_params)
    logger.info('Start predicting...')
    # predict
    pred = gbm_model.predict_proba(X_test)[:, 1]
    return pred


if __name__ == "__main__":
    outputFolder = ""
    logger, outputFolder = utils.work_pre(outputFolder)
    # print("Output Folder : %s" % outputFolder)
    logger.info("Start to run code...")

    # Load Dataset

    all_data, test_a, test_b, test = load_dataset()

    # Data Process
    # save to file
    logger.info("...Generate statistic maps start...")
    stat_maps = dict()
    logger.info("hour_trade_rate")
    stat_maps["hour_trade_rate"] = trade_rate_map(all_data, days=ALL_DAY)
    logger.info("gender_rate")
    stat_maps["gender_rate"] = gender_rate_map(all_data)
    logger.info("item_collected_level_count")
    stat_maps["item_collected_level_count"] = collected_level_count_map(all_data, days=ALL_DAY)
    logger.info("item_price_level_rate")
    stat_maps["item_price_level_rate"] = item_price_level_map(all_data)
    logger.info("shop_star_rate")
    stat_maps["shop_star_rate"] = shop_star_level_map(all_data)
    logger.info("...Generate statistic maps done...")

    # preprocess train data
    logger.info('...Preprocess train dataset start...')
    # drop duplicates
    all_data = all_data.drop_duplicates("instance_id")
    # data clean
    all_data = clean_data(all_data)

    train = pre_process(all_data, days=TRAIN_DAY, stat_maps=stat_maps)
    logger.info('...Preprocess train dataset done... \n train dataset shape : %s' % str(train.shape))
    logger.info("train dataset features : \n %s" % train.columns.tolist())
    logger.info("max train datetime : %s", train['time_'].max())
    logger.info("min train datetime : %s", train['time_'].min())
    #  preprocess test_a data
    logger.info('...Preprocess test_a dataset start...')
    test_a = pre_process(test_a, TEST_DAY, stat_maps=stat_maps)
    logger.info("...Preprocess test_a dataset done... \n test_a data shape : %s", str(test_a.shape))
    # preprocess test_b data
    logger.info('...Preprocess test_b dataset start...')
    test_b = pre_process(test_b, TEST_DAY, stat_maps=stat_maps)
    logger.info("...Preprocess test_b dataset done... \n test_b data shape : %s", str(test_b.shape))
    # preprocess val data
    logger.info('...Preprocess val dataset start...')
    val = pre_process(all_data, days=VAL_DAY, stat_maps=stat_maps)
    logger.info("...Preprocess val dataset done... \n val dataset shape : %s", str(val.shape))
    # preprocess test data
    logger.info('...Preprocess test dataset start...')
    test = pre_process(test, days=TEST_DAY, stat_maps=stat_maps)
    logger.info('...Preprocess test dataset done... \n test dataset shape : %s' % str(test.shape))
    logger.info("test dataset features : \n %s" % test.columns.tolist())
    logger.info("max test datetime : %s", test['time_'].max())
    logger.info("min test datetime : %s", test['time_'].min())
    logger.info("Concat the test to all_data")
    all_data = all_data.append(test).reset_index(drop=True)
    # Feature Engineering

    # drop some fea
    drop_fea_names = ["hour", "day", "min"]
    logger.debug("[Del]Drop features : %s" % drop_fea_names)
    for fea in drop_fea_names:
        train.pop(fea)
        test.pop(fea)
        val.pop(fea)

    # continual value
    ext_fea_names = [
        'shop_review_num_level', 'shop_review_positive_rate', 'shop_score_service', "shop_star_rate",
        "item_count", 'item_sales_level', 'item_pv_level', 'len_item_category', 'len_item_property',
        'len_predict_category_property', "collected_level_count", "item_price_rate",
        "user_count", "user_star_level_", "user_shop_count", "user_shop_count_count", "gender_trade_rate",
        "dow", "hour_count"]
    logger.info("Concat extera feature to train, val, test")
    for i, feat in enumerate(ext_fea_names):
        if i == 0:
            ext_train = pd.DataFrame(train.pop(feat))
            ext_test = pd.DataFrame(test.pop(feat))
            ext_val = pd.DataFrame(val.pop(feat))
        else:
            ext_train[feat] = train.pop(feat)
            ext_test[feat] = test.pop(feat)
            ext_val[feat] = val.pop(feat)
    logger.info("...Scalling data start...")

    min_max_scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    extra_train_sc = min_max_scaler.fit_transform(ext_train.values)
    extra_test_sc = min_max_scaler.fit_transform(ext_test.values)
    extra_val_sc = min_max_scaler.fit_transform(ext_val.values)

    extra_train_sc = pd.DataFrame(extra_train_sc, columns=ext_train.columns)
    extra_test_sc = pd.DataFrame(extra_test_sc, columns=ext_test.columns)
    extra_val_sc = pd.DataFrame(extra_val_sc, columns=ext_val.columns)
    logger.info("...Scalling data done...")
    logger.info("%i features : %s" % (len(val.columns.tolist()), val.columns.tolist()))

    logger.info("[Del]is_trade, instance_id,time_")
    y_train = train.pop('is_trade')
    train_index = train.pop('instance_id')

    y_val = val.pop('is_trade')
    val_index = val.pop('instance_id')
    test_index = test.pop('instance_id')
    test_b_index = test_b.pop('instance_id')

    del train['time_']
    del val['time_']
    del test['time_']
    logger.info("Final there are %i features" % len(test.columns.tolist()))

    logger.info('...Training start ...')
    lb = LabelEncoder()
    feat_set = list(test.columns)
    ext_feas = list(ext_train.columns)
    logger.info("\n Categrical Features\n %s", feat_set)
    logger.info("\n Continual Values Features\n %s", ext_feas)
    logger.info("...Generate X_train, X_val, X_test_a start...")
    logger.info("...Add categrical data start...")
    X_train, X_test, X_val = train, test, val
    for i, feat in enumerate(feat_set):
        #   拼接，for label
        feat_degree = len(feat_set)
        X_train[feat] = X_train[feat].apply(str)
        X_test[feat] = X_test[feat].apply(str)
        X_val[feat] = X_val[feat].apply(str)

        lb.fit((list(X_train[feat]) + list(X_val[feat]) + list(X_test[feat])))
        X_train[feat] = lb.transform(X_train[feat])
        X_test[feat] = lb.transform(X_test[feat])
        X_val[feat] = lb.transform(X_val[feat])

        logger.info("feat : X_train, X_val, X_test ->  %s : %s, %s, %s", feat, X_train.shape, X_val.shape, X_test.shape)

    logger.info("...Add categrical data end...")
    logger.info("...Add continual data start...")
    # 添加数值特征
    for i, feat in enumerate(ext_feas):
        logger.info("feat : X_train, X_val, X_test ->  %s : %s, %s, %s", feat, X_train.shape, X_val.shape, X_test.shape)
        X_train[feat], X_test[feat], X_val[feat] = extra_train_sc[feat], extra_test_sc[feat], extra_val_sc[feat]
    logger.info("...Add continual data end...")

    # Modeling
    logger.info("...Lightgbm modeling start...")
    # 线下
    best_iter, loss_gbm = search_param(X_train, y_train, X_val, y_val)
    # Result Output
    # 线上
    pred_sub = sub(X_train.append(X_val), y_train.append(y_val), X_test, best_iter)

    result_test = pd.DataFrame()
    result_test['instance_id'] = list(test_index)
    result_test['predicted_score'] = list(pred_sub)

    result_test_b = pd.DataFrame()
    result_test_b['instance_id'] = list(test_b_index)
    result_test_b = pd.merge(result_test_b, result_test, on=["instance_id"], how="left")

    filename = pd.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    filename = outputFolder + "Result_" + filename + "_CRLF.txt"
    result_test_b.to_csv(filename, sep=" ", index=False)

    # Convert CRLF to LF
    finalFilename = filename.replace("_CRLF", "_{0:.5f}".format(loss_gbm) + "_LF")

    with open(filename, 'r', newline=None) as infile:
        with open(finalFilename, 'w', newline='\n') as outfile:
            outfile.writelines(infile.readlines())
    # Save the last filename

    _, tempfilename = os.path.split(finalFilename)
    print(tempfilename)

    # Version Description
    ModelVersion = '''
      本次结果:{0:.6f}
      ALL_DAY:{1}
      TRAIN_DAY:{2}
      VAL_DAY:{3}
      TEST_DAY:{4}

      ---
      添加了三个特征
      'len_item_category','len_item_property','len_predict_category_property'
      加了之后结果更差了
      ---
      去掉one-hot编码da'te
      去掉one_hot之后就36个特征了
      ---
      调整了一下参数，然后到0.081805感觉妥妥的过拟合了
      去掉了context_id，context相关的特征还要考察怎么用
     '''
    logger.info(ModelVersion.format(loss_gbm, ALL_DAY, TRAIN_DAY, VAL_DAY, TEST_DAY))
