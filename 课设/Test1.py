import matplotlib.pyplot as plt
import seaborn as sns
import gc
import re
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.model_selection import StratifiedKFold
from dateutil.relativedelta import relativedelta
#读取数据
train_data = pd.read_csv(r'D:\大三上\机器学习\代码\raw_data\train_public.csv')
submit_example = pd.read_csv(r'D:\大三上\机器学习\代码\raw_data\submit_example.csv')
test_public = pd.read_csv(r'D:\大三上\机器学习\代码\raw_data\test_public.csv')
train_inte = pd.read_csv(r'D:\大三上\机器学习\代码\raw_data\train_internet.csv')

pd.options.display.max_rows=200
pd.options.display.max_columns=None
pd.options.display.float_format=lambda x:"%.3f" % x
#训练集训练
def train_model(data_, test_, y_, folds_):   #data_:各种数据包含train_data\train_inste\test_public  #folds:几折交叉验证
    oof_preds = np.zeros(data_.shape[0])     #返回一个给定形状和类型的用0填充的数组，此处是data_第一维度长度的全0数组
    sub_preds = np.zeros(test_.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in data_.columns if f not in ['loan_id', 'user_id', 'isDefault']]

    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):    #enumerate可以遍历集合并能得到元素的索引
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        #构建LGBM模型
        clf = LGBMClassifier(
            n_estimators=4000,      #核心参数
            learning_rate=0.08,       #核心参数       学习率
            num_leaves=2 ** 5,         #与决策树相关的参数   机器学习最大叶子数
            colsample_bytree=.65,
            subsample=.9,
            max_depth=5,             #与决策树相关的参数，机器学习的最大树深度
            reg_alpha=.3,     #L1正则化项
            reg_lambda=.3,    #L2正则化
            min_split_gain=.01,   #与决策树相关
            min_child_weight=2,
            silent=-1,        #是否打印每次运行结果
            verbose=-1,
        )
   #用LGBM训练拟合
        clf.fit(trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=40  # 30
                )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))

    test_['isDefault'] = sub_preds

    return oof_preds, test_[['loan_id', 'isDefault']], feature_importance_df


def display_importances(feature_importance_df_):
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature",
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')


def workYearDIc(x):
    if str(x) == 'nan':
        return -1
    x = x.replace('< 1', '0')
    return int(re.search('(\d+)', x).group())


def findDig(val):
    fd = re.search('(\d+-)', val)
    if fd is None:
        return '1-' + val
    return val + '-01'


class_dict = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
}
timeMax = pd.to_datetime('1-Dec-21')
#使用map函数对内部类型进行转化（因为内部数据类型有）
train_data['work_year'] = train_data['work_year'].map(workYearDIc)
test_public['work_year'] = test_public['work_year'].map(workYearDIc)
train_data['class'] = train_data['class'].map(class_dict)
test_public['class'] = test_public['class'].map(class_dict)
train_data['earlies_credit_mon'] = pd.to_datetime(train_data['earlies_credit_mon'].map(findDig))
test_public['earlies_credit_mon'] = pd.to_datetime(test_public['earlies_credit_mon'].map(findDig))

train_data.loc[train_data['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = train_data.loc[train_data['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(years=-100)
test_public.loc[test_public['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = test_public.loc[test_public['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(years=-100)
#贷款的月份按照年月日规整的写好  2016/1/4--->2016-01-04
train_data['issue_date'] = pd.to_datetime(train_data['issue_date'])
test_public['issue_date'] = pd.to_datetime(test_public['issue_date'])

# Internet数据处理
train_inte['work_year'] = train_inte['work_year'].map(workYearDIc)
train_inte['class'] = train_inte['class'].map(class_dict)
train_inte['earlies_credit_mon'] = pd.to_datetime(train_inte['earlies_credit_mon'])
train_inte['issue_date'] = pd.to_datetime(train_inte['issue_date'])
#测试集和训练集中贷款时间转换成月数和星期数
train_data['issue_date_month'] = train_data['issue_date'].dt.month
test_public['issue_date_month'] = test_public['issue_date'].dt.month
train_data['issue_date_dayofweek'] = train_data['issue_date'].dt.dayofweek
test_public['issue_date_dayofweek'] = test_public['issue_date'].dt.dayofweek
#
train_data['earliesCreditMon'] = train_data['earlies_credit_mon'].dt.month
test_public['earliesCreditMon'] = test_public['earlies_credit_mon'].dt.month
train_data['earliesCreditYear'] = train_data['earlies_credit_mon'].dt.year
test_public['earliesCreditYear'] = test_public['earlies_credit_mon'].dt.year

###internet数据
#对训练集（外部）的还款日期进行转换
train_inte['issue_date_month'] = train_inte['issue_date'].dt.month
train_inte['issue_date_dayofweek'] = train_inte['issue_date'].dt.dayofweek
train_inte['earliesCreditMon'] = train_inte['earlies_credit_mon'].dt.month
train_inte['earliesCreditYear'] = train_inte['earlies_credit_mon'].dt.year
##对这两个字段进行处理,将employer_type中相同的编程同一数字，有n种不同的类型则数字从0到n-1
cat_cols = ['employer_type', 'industry']
from sklearn.preprocessing import LabelEncoder
for col in cat_cols:
    lbl = LabelEncoder().fit(train_data[col])
    train_data[col] = lbl.transform(train_data[col])
    test_public[col] = lbl.transform(test_public[col])
    # Internet处理
    train_inte[col] = lbl.transform(train_inte[col])

#丢弃数据，将一些已经读取的数据丢弃（不影响）
col_to_drop = ['issue_date', 'earlies_credit_mon']
train_data = train_data.drop(col_to_drop, axis=1)
test_public = test_public.drop(col_to_drop, axis=1)
##internet处理
train_inte = train_inte.drop(col_to_drop, axis=1)

#列名的集合
tr_cols = set(train_data.columns)
same_col = list(tr_cols.intersection(set(train_inte.columns)))#这里是外部训练集和内部训练集的交集
#训练集
train_inteSame = train_inte[same_col].copy()  #dataframe

Inte_add_cos = list(tr_cols.difference(set(same_col)))   #差集tr_cols有的交集没有的
for col in Inte_add_cos:
    train_inteSame[col] = np.nan

#

y = train_data['isDefault']
folds = KFold(n_splits=5, shuffle=True, random_state=546789)#五折交叉验证
#首先训练原始样本（train_pubilc\train_inste）（train_data是去除了issue_data、earlies_credit_mon的内部训练集）
#tranin_inteSame是外部训练集中交集列的数据    y是训练集isDefalt列数据
oof_preds, IntePre, importances = train_model(train_data, train_inteSame, y, folds)

IntePre['isDef'] = train_inte['is_default']
from sklearn.metrics import roc_auc_score
roc_auc_score(IntePre['isDef'], IntePre.isDefault)    #预测
## 选择阈值0.08，从internet表中提取预测小于该概率的样本，并对不同来源的样本赋予来源值
InteId = IntePre.loc[IntePre.isDefault < 0.08, 'loan_id'].tolist()

#设置
train_data['dataSourse'] = 1
test_public['dataSourse'] = 1
train_inteSame['dataSourse'] = 0
train_inteSame['isDefault'] = train_inte['is_default']
use_te = train_inteSame[train_inteSame.loan_id.isin(InteId)].copy()
#扩充数据    concat（）此处为默认的纵向拼接  reset_index是索引重置
data = pd.concat([train_data, test_public, use_te]).reset_index(drop=True)


# IntePre.isDefault
plt.figure(figsize=(16, 6))
plt.title("Distribution of Default values IntePre")
sns.displot(IntePre['isDefault'], color="black", kde=True, bins=120, label='train_data')
# sns.distplot(train_inte[col],color="red", kde=True,bins=120, label='train_inte')
plt.legend();
plt.show()
train = data[data['isDefault'].notna()]  #isDefalut的缺省布尔列表
test = data[data['isDefault'].isna()]
# for col in ['sub_class', 'work_type']:
#     del train[col]
#     del test[col]


del data
del train_data, test_public

y = train['isDefault']
#再度训练以及预测
folds = KFold(n_splits=5, shuffle=True, random_state=546789) #五折交叉验证
oof_preds, test_preds, importances = train_model(train, test, y, folds)
#训练得到结果数据集
test_preds.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('baseline.csv', index=False)
