import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import csv

path = './'
teamname = 'huji hackers'
out_name = path + teamname + '_submission.csv'

usage_f = "train_usage_data.csv"
data_f = "train_users_data.csv"
test_usage_f = "test_usage_data.csv"
test_data_f = "test_users_data.csv"

train_res = None


# def fix_and_create_data(usage_f, data_f):


def arange_data(usage_f, data_f, test_flag=False):
    # read data
    df_usage = pd.read_csv(path + usage_f)
    train_meta_users = pd.read_csv(path + data_f, parse_dates=['installation_date', 'subscripiton_date'])
    df_usage['end_use_date'] = pd.to_datetime(df_usage['end_use_date'], infer_datetime_format=True)
    df_usage['active_time'] = df_usage['end_use_date'].dt.time

    train_meta_users['delta_date'] = (
    train_meta_users["subscripiton_date"] - train_meta_users["installation_date"]).apply(
        lambda x: x.days if x.seconds // 3600 < 12 else x.days + 1)
    train_meta_users['delta_date'] = train_meta_users['delta_date'].apply(lambda x: x // 30)
    df_users = train_meta_users.drop(columns='Unnamed: 0')

    joined_df = df_usage.join(df_users.set_index('id'), on='id')

    # Let's create a table with statistic summaries: rows correspond to users; columns to various statistics:
    users_usage_summaries = pd.pivot_table(df_usage[['id', 'feature_name']], index=['id'], columns=['feature_name'],
                                           aggfunc=len, fill_value=0)

    # Let's add the mean of 'accepted' for each user:
    accepted_rate = df_usage.groupby(['id'])['accepted'].mean().to_frame()
    if not test_flag:
        churned = joined_df.groupby(['id'])['churned'].mean().to_frame()
    train_meta_users_delta_date = train_meta_users.groupby(['id'])['delta_date'].sum().to_frame()

    usage_duration = df_usage.groupby(['id'])['usage_duration'].mean().to_frame()

    joined_df['week_day'] = (joined_df["end_use_date"] - joined_df["subscripiton_date"]).apply(
        lambda x: x.days if x.seconds // 3600 < 12 else x.days + 1)

    week_day = joined_df.groupby(["id", "week_day"]).count().max(level=0).to_frame()
    if not test_flag:
        data_res = users_usage_summaries.join(accepted_rate, how='left').join(churned, how='left').join(
            train_meta_users_delta_date, how='left').join(usage_duration, how='left').join(week_day, how='left')
    if test_flag:
        data_res = users_usage_summaries.join(accepted_rate, how='left').join(train_meta_users_delta_date,
                                              how='left').join(usage_duration, how='left').join(week_day, how='left')
    # if not test_flag:
    #     features = data_res.drop(["churned"], axis=1).columns
    return data_res


def train(data_res):
    X = data_res.iloc[:, data_res.columns != 'churned'].values
    y = data_res.loc[:, 'churned'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05)

    clf = RandomForestClassifier(n_estimators=100, max_depth=3)
    # df_train, df_test = train_test_split(data_res, test_size=0.25)
    # clf.fit(df_train[features], df_train["churned"])

    train = clf.fit(X_train, y_train)

    res = train.predict(X_val)
    cm = confusion_matrix(y_val, res)
    # print("sss: ", cm)
    print(classification_report(y_pred=res, y_true=y_val))
# print only f1 score for positive
# print("aaa: ", np.round(f1_score(y_pred=res,y_true=y_val),3))

    train = clf.fit(X_train, y_train)

    test_data_res = arange_data(test_usage_f, test_data_f, True)

    # score = clf.score(df_test[features], df_test["churned"])
    # print("Accuracy: ", score)

    # # Make predictions
    # predictions = clf.predict(df_test[features])
    # probs = clf.predict_proba(df_test[features])
    # display(predictions)
    #
    # # get_ipython().magic('matplotlib inline')
    # confusion_matrix = pd.DataFrame(
    #     confusion_matrix(df_test["churned"], predictions),
    #     columns=["Predicted False", "Predicted True"],
    #     index=["Actual False", "Actual True"]
    # )
    # display(confusion_matrix)
    #
    # # Calculate the fpr and tpr for all thresholds of the classification
    # fpr, tpr, threshold = roc_curve(df_test["churned"], probs[:, 1])
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()

    # submit result:
    # test your predictor:

    # 1.Prepare your test-set (in case you created new features/transformed the input data):

    # Let's merge the two:
    X_test = test_data_res.values
    pred = train.predict(X_test)
    df = pd.DataFrame(pred, index=test_data_res.index.astype(str), columns=['churned'], dtype=str)
    df.to_csv(out_name, header=True, quoting=csv.QUOTE_NONNUMERIC)
    return None



if __name__ == '__main__':
    train_data_res = arange_data(usage_f, data_f)
    train_res = train(train_data_res)
    # test_data_res = arange_data(test_usage_f, test_data_f, True)
    # final_res = train(test_data_res, True)
