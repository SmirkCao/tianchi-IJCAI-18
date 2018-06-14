import pandas as pd
import numpy as np
import time
import logging

# 参数配置
# 1,5,8 是一类，包含三次点击成交的样本
# 2，3，4，6，7是一类，只包含2次点击成交的样本

ALL_DAY = [1, 2, 3, 4, 5, 6, 7]
DROP_DAY = []
TRAIN_DAY = [1, 2, 3, 4, 5, 6]
VAL_DAY = list(set(ALL_DAY).difference(set(TRAIN_DAY).union(set(DROP_DAY))))
TEST_DAY = [8]

logger = logging.getLogger("CTR")
logger.info("DAYS : 18,19,20,21,22,23,24,25")
logger.info("ALL_DAY %s, TRAIN_DAY %s, VAL_DAY %s, TEST_DAY %s ", ALL_DAY, TRAIN_DAY, VAL_DAY, TEST_DAY)


def data_split_by_day(data, days=TRAIN_DAY):
    data["time_"] = data["context_timestamp"].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
    data["time_t"] = data["time_"].apply(pd.to_datetime)
    data["day"] = data["time_t"].dt.day - 17  # caution : hardcode
    data['hour'] = data['time_t'].dt.hour
    data['min'] = data['time_t'].dt.minute
    # 希望9-18是周一，按照日期偏差刚好  
    data["dow"] = data["time_t"].dt.dayofweek
    data["dow"] = data["dow"].apply(lambda x: 7 if x == 0 else x)
    data_split = []
    for i in days:
        data_temp = data[data["day"] == i]
        del data_temp["time_t"]
        data_split.append(data_temp)
    return data_split


# 数据抽样
# test_a的数据不全，要考虑抽样或者丰富test_a
# test_b的数据对test_a做了补充，基本上一样。暂时不抽样

# 统计特征
def user_count(data):
    logger.debug("user count")
    user_count_df = pd.DataFrame(data.groupby("user_id").size(), columns=["user_count"]).reset_index()
    data = pd.merge(data, user_count_df, on=["user_id"], how="left")
    return data


def item_count(data):
    logger.debug("item count")
    item_count_df = pd.DataFrame(data.groupby("item_id").size(), columns=["item_count"]).reset_index()
    data = pd.merge(data, item_count_df, on=["item_id"], how="left")
    return data


# 希望构建一个和时间相关的统计特征，每个小时的成交率，那么测试集怎么弄，按天统计，希望能给个参考。7天是一周的循环。
# 如果用这个的话，那应该把重复值处理掉。
def hour_count(data):
    #     logger.info("hour count")
    hour_count_df = pd.DataFrame(data.groupby(by=["hour", "is_trade"]).size(), columns=["hour_count"]).reset_index()
    count_is_trade_df = hour_count_df[hour_count_df["is_trade"] == 1].set_index("hour")
    count_no_trade_df = hour_count_df[hour_count_df["is_trade"] == 0].set_index("hour")
    hour_count = pd.DataFrame({"hour_count": count_is_trade_df["hour_count"]
                                             / count_no_trade_df["hour_count"]}).reset_index()
    # data = pd.merge(data,hour_count,on=["hour"],how="left")
    return hour_count


#
def trade_rate_map(data, days):
    # 按照日期生成map
    data_split = data_split_by_day(data, days=days)
    #     hour_count_map = get_hour_count_map(data,days)
    data_processed = []
    for (data_, day) in zip(data_split, days):
        hour_trade_rate = hour_count(data_)
        data_processed.append(hour_trade_rate)
    return data_processed


def gender_rate_map(data):
    gender_trade_rate = pd.DataFrame({"gender_trade_rate": data.groupby(by=["user_gender_id", "is_trade"]).size()})
    gender_trade_rate = gender_trade_rate.reset_index("is_trade")
    tmp_df = gender_trade_rate[gender_trade_rate["is_trade"] == 1] / (
                gender_trade_rate[gender_trade_rate["is_trade"] == 0] + gender_trade_rate[
            gender_trade_rate["is_trade"] == 1])
    del tmp_df["is_trade"]
    tmp_df = tmp_df.reset_index()
    return tmp_df


def count_collected_level(data):
    tmp_df = pd.DataFrame({"collected_level_count": data.groupby(by=["item_collected_level"]).size()}).reset_index()
    return tmp_df


def collected_level_count_map(data, days):
    processed_data = []
    for data_ in data_split_by_day(data, days):
        tmp_df = count_collected_level(data_)
        processed_data.append(tmp_df)
    return processed_data


def item_price_level_map(data):
    tmp_df = pd.DataFrame({"item_price_rate": data.groupby(by=["item_price_level", "is_trade"]).size()}).reset_index(
        "is_trade")
    tmp_df = (tmp_df[tmp_df["is_trade"] == 1] / (
                tmp_df[tmp_df["is_trade"] == 0] + tmp_df[tmp_df["is_trade"] == 1])).fillna(0).drop("is_trade", axis=1)
    tmp_df = tmp_df.reset_index()
    return tmp_df


def shop_star_level_map(data):
    tmp_df = pd.DataFrame({"shop_star_rate": data.groupby(by=["shop_star_level", "is_trade"]).size()}).reset_index(
        "is_trade")
    tmp_df = (tmp_df[tmp_df["is_trade"] == 1] / (
                tmp_df[tmp_df["is_trade"] == 0] + tmp_df[tmp_df["is_trade"] == 1])).fillna(0).drop("is_trade", axis=1)
    tmp_df = tmp_df.reset_index()
    return tmp_df


def user_shop_count(data):
    #     count + count_count，count 判节距，count_count判斜率
    #
    user_shop_df = data.groupby(by=["user_id", "shop_id"]).size()
    user_shop_df = pd.DataFrame({'user_shop_count': user_shop_df})
    user_shop_df.reset_index(inplace=True)
    data = pd.merge(data, user_shop_df, on=["user_id", "shop_id"], how="left")
    temp_df = data.groupby(by=["user_shop_count"]).size()
    temp_df = pd.DataFrame({'user_shop_count_count': temp_df})
    temp_df.reset_index(inplace=True)
    data = pd.merge(data, temp_df, on=["user_shop_count"], how="left")
    data["user_shop_count_count"] = data["user_shop_count_count"].apply(np.log)

    data["effe_th"] = data["day"].apply(lambda x: 2 if x in [2, 3, 4, 6, 7] else 3)
    data["user_shop_count"] = data["user_shop_count"][(data["user_shop_count"] - data["effe_th"]) > 0]
    data["user_shop_count"] = data["user_shop_count"].fillna(0)
    data["user_shop_count_count"][data["user_shop_count"] == 0] = 0
    return data


# outlier处理
# 之前滤掉了点击率 >14的，开始认为这里是离群点，实际上这部分算是采样的工作。但是这样采不是很合理。
def clean_user_count_outlier(data):
    # 清除点击过多的记录
    logger.info("Clean user count outlier")
    # user_count <= 14 ,B榜数据集补充之后，数据分布基本上一样了。不用drop了
    # data = data[data["user_count"] <= 14]
    return data


def clean_item_price_level_outlier(data):
    #     删除训练集中价格0，成交的那条记录
    data = data.drop(data[data["item_price_level"] == 0][data["is_trade"] == 1].index)
    data = data.drop(data[data["item_price_level"] == 16].index)
    data = data.drop(data[data["item_price_level"] == 17].index)
    #     删掉缺失值很多的
    data = data.drop(148963)
    data = data.drop(300028)
    data = data.drop(114423)
    data = data.drop(115243)
    data = data.drop(420848)
    data = data.drop(420847)
    data = data.drop(333200)
    data = data.drop(206298)
    data = data.drop(271385)
    data = data.drop(67300)
    data = data.drop(213280)
    data = data.drop(213279)
    return data


def clean_shop_star_level_outlier(data):
    # 5001的数据离群，删掉2个样本的店铺，该店铺在测试集中不存在
    data = data.drop(data[data["shop_id"] == 7164491533039214442].index)
    return data


# 转变成分类特征
# 用户的预测职业编号，Int类型
def process_user_occupation_id(x):
    if x == -1 or x == 2003:
        value = 1
    if x == 2004 or x == 2005:
        value = 2
    if x == 2002:
        value = 3
    if not value:
        print(x)
    return value


def process_user_star_level(x):
    # 用户的星级编号，Int类型；数值越大表示用户的星级越高

    if x == -1 or x == 3000:
        value = 1
    if 3001 <= x <= 3008:
        value = 2
    if x == 3009 or x == 3010:
        value = 3
    return value


def process_user_age_level(x):
    # 用户的预测年龄等级，Int类型；数值越大表示年龄越大

    if x == -1 or x == 1000 or x == 1001:
        value = 1
    if 1004 >= x >= 1002:
        value = x - 1000
    if x == 1006:
        value = 5
    if x == 1005 or x == 1007:
        value = 6
    return value


# 数据质量分析
#  缺失值处理，异常值处理
def clean_data(data):
    #   清理price_level train包含，但是测试数据不包含的
    #   清理在price_level = 0的情况下，训练数据里面成交的那条数据
    data = clean_item_price_level_outlier(data)
    # 清理5001
    data = clean_shop_star_level_outlier(data)
    # shop_review_positive 这个训练集中有7条空数据
    return data


# 数据预处理
def pre_process(data, days, stat_maps={}):
    logger.debug("Split data by %s days." % days)
    # 日期分桶统计特征
    data_split = data_split_by_day(data, days=days)
    data_processed = []
    for (data_, day) in zip(data_split, days):
        logger.debug("clean data of %s th day." % day)

        data_ = item_count(data_)
        # user
        data_ = user_count(data_)
        logger.debug('[Add]user_shop_count,user_count_count_log')
        data_ = user_shop_count(data_)
        # item
        hour_trade_rate = stat_maps["hour_trade_rate"][(day) % 7]
        data_ = pd.merge(data_, hour_trade_rate, on=["hour"], how="left")

        collected_level_count = stat_maps["item_collected_level_count"][(day) % 7]
        data_ = pd.merge(data_, collected_level_count, on=["item_collected_level"], how="left")
        data_ = pd.merge(data_, stat_maps["item_price_level_rate"], on=["item_price_level"], how="left")
        # shop
        data_ = pd.merge(data_, stat_maps["shop_star_rate"], on=["shop_star_level"], how="left")
        data_processed.append(data_)
    data = pd.concat(data_processed)
    # 全局统计特征
    gender_rate = stat_maps["gender_rate"]
    data = pd.merge(data, gender_rate, on=["user_gender_id"], how="left")

    data['len_item_category'] = data['item_category_list'].map(lambda x: len(str(x).split(';')))
    data['len_item_property'] = data['item_property_list'].map(lambda x: len(str(x).split(';')))

    logger.debug('[Proc]item_category_list->category_0,category_1,category_2')
    # 1 item_category_list 
    # 广告商品的的类目列表，String类型；
    # 从根类目（最粗略的一级类目）向叶子类目（最精细的类目）依次排列，
    # 数据拼接格式为 "category_0;category_1;category_2"，
    # 其中 category_1 是 category_0 的子类目，category_2 是 category_1 的子类目
    for i in range(3):
        data['category_%d' % (i)] = data['item_category_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else " "
        )
    logger.debug('[Del]item_category_list,category_0')
    del data['item_category_list']
    del data['category_0']  # category_0 都一样
    #    del data['category_2']
    # 2 item_property_list
    # 这个挺复杂的，很多内容，暂时分一下试试
    logger.debug('[Proc]item_property_list->property_0,property_1,property_2')
    #     from sklearn.feature_extraction.text import TfidfVectorizer
    #     count_vec = TfidfVectorizer()
    #     data_ip = count_vec.fit_transform(data['item_property_list'])
    #     print("here:",data_ip.shape,data.shape)
    #     print(data_ip[:4,:])
    for i in range(3):
        data['property_%d' % (i)] = data['item_property_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else " "
        )
    logger.debug('[Del]item_property_list')
    del data['item_property_list']
    data['len_predict_category_property'] = data['predict_category_property'].map(lambda x: len(str(x).split(';')))
    logger.debug('[Proc]predict_category_property->predict_category_0,predict_category_1,predict_category_2')
    for i in range(3):
        data['predict_category_%d' % (i)] = data['predict_category_property'].apply(
            lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else " "
        )
    logger.debug('[Del]predict_category_property,predict_category_0')
    del data['predict_category_property']
    del data['predict_category_1']
    del data['predict_category_2']
    # 5 user_occupation_id
    #     data['user_occupation_id_'] = data['user_occupation_id'].apply(process_user_occupation_id)
    #     del data['user_occupation_id']

    # 6 user_star_level_
    logger.debug('[Proc]user_star_level->user_star_level_')
    data['user_star_level_'] = data['user_star_level'].apply(process_user_star_level)
    logger.debug('[Del]user_star_level')
    del data['user_star_level']

    logger.debug('[Del]shop_score_delivery,shop_score_description')
    del data["shop_score_delivery"]
    del data["shop_score_description"]
    logger.debug('Drop empty shop review positive rate')
    # 去掉好评是空的，这部分数据比较少，训练数据里面有7条
    data = data[data["shop_review_positive_rate"].notnull()]
    #     del data["time_"]
    del data["user_gender_id"]
    del data["item_collected_level"]
    del data["item_price_level"]
    del data["context_timestamp"]
    del data["shop_star_level"]
    del data["context_id"]
    return data
