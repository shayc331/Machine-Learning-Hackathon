
################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################

################################################
# Eran Turgeman
# Noam Issachar
# Shay Cohen
# Itay Chachy
################################################

################################################
#NOTE:
# This project was made during HUJI Machine Learining Hackathon
# This project was the first time we created actual learinging model without any instructions or directions
# This project was made in less than 35 hours
################################################

# imports
import pickle as serialize
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import collections
import ast
import matplotlib.pyplot as plt

# Constants
TRAINING_SIZE = 3500
TRAIN = 0
TEST = 1
top_2_languages, revenue_directors_sets, revenue_actors_sets, vote_directors_sets, vote_actors_sets = \
    None, None, None, None, None

TEST_FILE = 'test_data.csv' # change in case you want to check different test file

TO_STAY = ['id', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'original_language', 'original_title',
           'overview', 'vote_average', 'vote_count', 'production_companies', 'production_countries',
           'release_date', 'runtime', 'spoken_languages', 'status', 'tagline', 'title', 'keywords', 'cast', 'crew',
           'revenue']

############# helper function to visualize the data #############
def plot_feature(df: pd.DataFrame, y: pd.DataFrame, feature_name: str, y_name: str) -> None:
    """
    This function used to help us visualize the data
    Plots (scatter) a given feature with a given response vector
    :param df: data frame with all data
    :param y: response vector (column)
    :param feature_name: feature name
    :param y_name: y name
    :return: None
    """
    plt.scatter(df[[feature_name]], y)
    plt.xlabel(f'{feature_name}')
    plt.ylabel(y_name)
    plt.savefig(f"{feature_name} {y_name}")
    plt.show()
#_____________________________________________________________#

def divide_data(csv_file: str) -> None:
    """
    divides data into training samples and test samples, and saves them as csv files
    :param csv_file: original data csv file
    :return: None
    """
    df = pd.read_csv(csv_file)
    rand = np.arange(df.shape[0])
    np.random.shuffle(rand)
    df.loc[rand[TRAINING_SIZE:], :].to_csv("test_data.csv", index=False)
    df.loc[rand[:TRAINING_SIZE], :].to_csv("training_data.csv", index=False)
    return


def load_data(csv_file: str):
    """
    loads data from a given path
    :param csv_file: path to file
    :return: loaded data as a DataFrame
    """
    df = pd.read_csv(csv_file)
    return df


def predict(csv_file: str) -> tuple:
    """
    Given a csv data file, predicts revenues and vote_avg of movies.
    Note: Here you should also load your model since we are not going to run the training process.
    (In the task given to us we had to save the trained model and not re-train it every single run)
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    global top_2_languages, revenue_directors_sets, revenue_actors_sets, vote_directors_sets, vote_actors_sets

    #loads data and models, preparing data to work with
    randomForestRegressor_revenue_model = serialize.load(open("revenue_model.sav", 'rb'))
    randomForestRegressor_vote_average_model = serialize.load(open("vote_rate_model.sav", 'rb'))
    globals_array = serialize.load(open("globals_array.sav", 'rb'))
    top_2_languages, revenue_directors_sets, revenue_actors_sets, vote_directors_sets, vote_actors_sets = \
        globals_array[0], globals_array[1], globals_array[2], globals_array[3], globals_array[4]
    revenue_df = load_data(csv_file)
    vote_df = load_data(csv_file)
    #filtering any extra columns that are not in the original training data
    revenue_df = revenue_df[TO_STAY]
    vote_df = vote_df[TO_STAY]

    try:
        not_released = set()
        for i, k in enumerate(revenue_df["status"]):
            if k != "Released":
                not_released.add(i)
        revenue_df = preprocess_test_revenue(revenue_df)
        vote_df = preprocess_test_vote(vote_df)

        #filtering columns needed to be predicted
        vote_df.drop(labels=['vote_average', 'revenue'], axis=1, inplace=True)
        revenue_df.drop(labels=['vote_average', 'revenue'], axis=1, inplace=True)

        #training chosen models among the four we tested
        #NOTE: we trained: LinearRegression, RandomForest, Ridge, Lasso and out the four of them we got the best
        # result (RMSE error function) with the RandomForest. In this project we left only the chosen model and applied
        # Lasso on it to improve its results
        revenue_prediction = randomForestRegressor_revenue_model.predict(revenue_df)
        vote_prediction = randomForestRegressor_vote_average_model.predict(vote_df)
        for i in not_released:
            # movies that have not been released yet are not helping us learn in this case
            revenue_prediction[i] = 0
            vote_prediction[i] = 0

        return list(revenue_prediction), list(np.round(vote_prediction, decimals=1))

    except Exception:
        # there where no assumptions on the data tested out model' and no assumptions when it could have fail out code
        # so we calculated the mean predictions and in any failure case we predicted the mean values
        mean_vals = serialize.load(open("mean_values.sav", 'rb'))
        num_of_samples = len(revenue_df)
        return [mean_vals[0][0]] * num_of_samples, [np.round(mean_vals, decimals=1)[1][0]] * num_of_samples


######################### validating data frames worked with #########################
def validate_training_samples(data: pd.DataFrame) -> pd.DataFrame:
    """
    validates the training data samples: checking data set and filling missing values and fixing problems
     in data needed for the model, and cpuld fail the code
    :param data: training data set (DataFrame)
    :return: validated training data (DataFrame)
    """
    global top_2_languages
    data["budget"] = data["budget"].fillna(0)
    data["budget"] = data["budget"].map(lambda x: abs(x))

    data["vote_count"] = data["vote_count"].fillna(0)
    data["vote_count"] = data["vote_count"].map(lambda x: abs(x))

    data = data[data['status'].map(lambda x: x == "Released")]

    data["runtime"] = data["runtime"].fillna(0)
    data["runtime"] = data["runtime"].map(lambda x: abs(x))
    data = data[data["runtime"] > 0]
    data = data[data["revenue"] >= 0]

    data = data[data["vote_average"].between(0.0, 10.0)]

    top_2_languages = data["original_language"].value_counts()[:2].index.tolist()
    data[f'original_language_{top_2_languages[0]}'] = 1 * (data[["original_language"]] == top_2_languages[0])
    data[f'original_language_{top_2_languages[1]}'] = 1 * (data[["original_language"]] == top_2_languages[1])
    data["original_language_other"] = np.abs(
        (data[f'original_language_{top_2_languages[0]}'] + data[f'original_language_{top_2_languages[1]}']) - 1)

    data["homepage"] = data["homepage"].map(lambda x: 1 if type(x) == str else 0)

    data["belongs_to_collection"] = data["belongs_to_collection"].map(lambda x: 1 if type(x) == str else 0)

    data["release_date"] = data["release_date"].map(lambda x: str(x)[6:])
    data["release_date"] = data["release_date"].map(lambda x: 0 if x == '' or not x.isdigit() else int(x))

    return data

def validate_test_samples(data: pd.DataFrame) -> pd.DataFrame:
    """
    validates the test samples
    :param data: test data
    :return: validated data
    """
    data["budget"] = data["budget"].fillna(0)
    data["budget"] = data["budget"].map(lambda x: abs(x))
    data["vote_count"] = data["vote_count"].fillna(0)
    data["vote_count"] = data["vote_count"].map(lambda x: abs(x))
    data["runtime"] = data["runtime"].fillna(0)
    data["runtime"] = data["runtime"].map(lambda x: abs(x))
    data["homepage"] = data["homepage"].map(lambda x: 1 if type(x) == str else 0)
    data["belongs_to_collection"] = data["belongs_to_collection"].map(lambda x: 1 if type(x) == str else 0)
    data["release_date"] = data["release_date"].map(lambda x: str(x)[6:])
    data["release_date"] = data["release_date"].map(lambda x: 0 if x == '' or not x.isdigit() else int(x))
    return data
#______________________________________________________________________________________#

############################ preprocessing data ########################################
# This part takes care of preprocessing the DataFrames (train & test) #
def preprocess_train_revenue(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing all DateFrame features (including dropping unused data) and making DataFrame ready to be learned from
    :param data: DataFrame to preprocess
    :return: DataFrame after preprocessing
    """
    global top_2_languages
    labels_to_drop = ['id', 'original_title', 'overview', 'tagline', 'title', 'production_companies',
                      'production_countries', "spoken_languages", 'keywords'] #dropping unused data

    data.drop(labels=labels_to_drop, axis=1, inplace=True)
    preprocess_json(data, TRAIN)
    data = validate_training_samples(data)
    for low, high in [(2021, 2022), (2010, 2020), (2000, 2009), (1990, 1999), (1975, 1989), (0, 1974)]:
        data[f"release_date_{low}-{high}"] = 1 * (data['release_date'].between(low, high))
    data.drop(labels=['original_language', 'status', "genres", 'release_date', 'crew', 'cast'], axis=1, inplace=True)
    return data

def preprocess_test_revenue(data: pd.DataFrame) -> pd.DataFrame:
    """
    Responsible of the preprocessing stage of test data
    :param data: test data
    :return: processed data
    """
    global top_2_languages
    labels_to_drop = ['id', 'original_title', 'overview', 'tagline', 'title', 'genres', 'production_companies',
                      'production_countries', "spoken_languages", 'keywords']
    data.drop(labels=labels_to_drop, axis=1, inplace=True)
    data = validate_test_samples(data)
    for low, high in [(2021, 2022), (2010, 2020), (2000, 2009), (1990, 1999), (1975, 1989), (0, 1974)]:
        if low != 2021:
            data[f"release_date_{low}-{high}"] = 1 * (data['release_date'].between(low, high))
        else:
            data[f"release_date_{low}-{high}"] = data['release_date'].between(low, high)
            data[f"release_date_{low}-{high}"] |= (data['status'] != 'Released')
            data[f"release_date_{low}-{high}"] *= 1
    preprocess_json(data, TEST)
    data[f'original_language_{top_2_languages[0]}'] = 1 * (data[["original_language"]] == top_2_languages[0])
    data[f'original_language_{top_2_languages[1]}'] = 1 * (data[["original_language"]] == top_2_languages[1])
    data["original_language_other"] = np.abs(
        (data[f'original_language_{top_2_languages[0]}'] + data[f'original_language_{top_2_languages[1]}']) - 1)
    data.drop(labels=['original_language', 'status', 'release_date', 'crew', 'cast'], axis=1, inplace=True)
    return data


def preprocess_train_vote(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing all DateFrame features (including dropping unused data) and making DataFrame ready to be learned from
    :param data: DataFrame to preprocess
    :return: DataFrame after preprocessing
    """
    global top_2_languages
    labels_to_drop = ['id', 'original_title', 'overview', 'tagline', 'title', 'production_companies', 'production_countries'
                      , "spoken_languages", 'keywords']
    data.drop(labels=labels_to_drop, axis=1, inplace=True)
    preprocess_json(data, TRAIN)
    data = validate_training_samples(data)

    # The significant drop in features is due to testing the model with Lasso. We noticed that those features are less
    # significant for learining with the model
    data.drop(labels=['belongs_to_collection', 'original_language', 'release_date', 'homepage', 'runtime', 'status',
                      "genres", 'crew', 'cast', 'director_level_2', 'director_level_4', 'director_level_3', 'actor_level_2',
                      'actor_level_3', 'original_language_other', f'original_language_{top_2_languages[0]}',
                      f'original_language_{top_2_languages[1]}'], axis=1, inplace=True)
    return data


def preprocess_test_vote(data: pd.DataFrame) -> pd.DataFrame:
    """
    Responsible of the preprocessing stage of test data
    :param data: test data
    :return: processed data
    """
    global top_2_languages
    labels_to_drop = ['id', 'original_title', 'overview', 'tagline', 'title', 'production_companies',
                      'production_countries', "spoken_languages", 'keywords']
    data.drop(labels=labels_to_drop, axis=1, inplace=True)
    data = validate_test_samples(data)
    preprocess_json(data, TEST)
    data.drop(labels=['belongs_to_collection', 'original_language', 'release_date', 'homepage', 'runtime', 'status', "genres", 'crew', 'cast',
                      'director_level_2', 'director_level_4', 'director_level_3', 'actor_level_2', 'actor_level_3'], axis=1, inplace=True)
    return data


def is_train(set_type: int) -> bool:
    """
    :return: bool
    """
    return set_type == TRAIN

# This part takes care of preprocessing certain data features so we can use the data to train the model #

def preprocessing_cast(df: pd.DataFrame, actors_dict: dict, y: str) -> tuple:
    """
    preprocessing of 'cast' feature
    :param df: DataFrame
    :param actors_dict: dictionary of actors
    :param y: response feature
    :return: tuple
    """
    actors = [[] for _ in range(len(df))]
    for i, lst in enumerate(df['cast']):
        for dic in lst[:3]:
            actors_dict[dic['name']][0] += 1
            actors_dict[dic['name']][1] += df[y][i]
            actors[i].append(dic['name'])
    return actors_dict, actors


def preprocessing_crew(df: pd.DataFrame, directors_dict: dict, y: str) -> tuple:
    """
    preprocessing of 'crew' feature
    :param df: DataFrame
    :param directors_dict: dictionary of crew members
    :param y: response feature
    :return: tuple
    """
    directors = [0 for _ in range(len(df))]
    for i, lst in enumerate(df["crew"]):
        for dic in lst:
            if dic['job'] == 'Director':
                directors_dict[dic['name']][0] += 1
                directors_dict[dic['name']][1] += df[y][i]
                directors[i] = dic['name']
    return directors_dict, directors

def create_sets(result_dict: dict) -> list:
    """
    helper function for preprocessing stage
    :param result_dict: dictionary
    :return: list
    """
    sort_dict = sorted(result_dict.items(), key=lambda k_v: k_v[1], reverse=True)
    size = len(sort_dict)
    k, j, l, s = sort_dict[:size // 4], sort_dict[size // 4:size // 2], sort_dict[size // 2: (3 * size) // 4], sort_dict[(3 * size) // 4:]
    set_list = list()
    for temp in [k, j, l, s]:
        set_list.append({i[0] for i in temp})
    return set_list

def preprocessing_cast_and_crew(df: pd.DataFrame, feature: str, y: str) -> tuple:
    """
    preprocessing of 'cast' and 'crew' features
    :param df: DataFrame
    :param feature: certain feature we want to work with
    :param y: response feature
    :return: tuple
    """
    result_dict = collections.defaultdict(lambda: [0, 0])
    result_dict, index_dict = preprocessing_crew(df, result_dict, y) if feature == "crew" else \
        preprocessing_cast(df, result_dict, y)
    result_dict = {k: v[1]/v[0] for k, v in result_dict.items()}
    return create_sets(result_dict), index_dict


def preprocess_revenue_crew(data: pd.DataFrame, set_type: int) -> None:
    """
    preprocess the 'crew' feature for the revenue DataFrame
    :param data: DataFrame to be processed
    :param set_type: whether the DataFrame is the train df or the test df
    :return: None
    """
    global revenue_directors_sets
    data["crew"] = data["crew"].fillna("[]")
    data["crew"] = data["crew"].apply(lambda x: list(ast.literal_eval(x)))
    if is_train(set_type):
        revenue_directors_sets, directors_names = preprocessing_cast_and_crew(data, "crew", "revenue")
    else:
        directors_names = preprocessing_cast_and_crew(data, "crew", "revenue")[1]
    for i, s in enumerate(revenue_directors_sets, 1):
        data[f'director_level_{i}'] = np.zeros(len(data))
        for j, name in enumerate(directors_names):
            data.loc[j, f'director_level_{i}'] = 1 if name in s else 0


def preprocess_revenue_cast(data: pd.DataFrame, set_type: int) -> None:
    """
    preprocess the 'cast' feature for the revenue DataFrame
    :param data: DataFrame to be processed
    :param set_type: whether the DataFrame is the train df or the test df
    :return: None
    """
    global revenue_actors_sets
    data["cast"] = data["cast"].fillna("[]")
    data["cast"] = data["cast"].apply(lambda x: list(ast.literal_eval(x)))
    if is_train(set_type):
        revenue_actors_sets, actors_names = preprocessing_cast_and_crew(data, "cast", "revenue")
    else:
        actors_names = preprocessing_cast_and_crew(data, "cast", "revenue")[1]
    for i, s in enumerate(revenue_actors_sets, 1):
        data[f'actor_level_{i}'] = np.zeros(len(data))
        for j, names in enumerate(actors_names):
            for name in names:
                data.loc[j, f'actor_level_{i}'] += 1 if name in s else 0


def preprocess_vote_crew(data: pd.DataFrame, set_type: int) -> None:
    """
    preprocess the 'crew' feature for the vote_avg DataFrame
    :param data: DataFrame to be processed
    :param set_type: whether the DataFrame is the train df or the test df
    :return: None
    """
    global vote_directors_sets
    if is_train(set_type):
        vote_directors_sets, directors_names = preprocessing_cast_and_crew(data, "crew", "vote_average")
    else:
        directors_names = preprocessing_cast_and_crew(data, "crew", "vote_average")[1]
    for i, s in enumerate(vote_directors_sets, 1):
        data[f'director_level_{i}'] = np.zeros(len(data))
        for j, name in enumerate(directors_names):
            data.loc[j, f'director_level_{i}'] = 1 if name in s else 0


def preprocess_vote_cast(data: pd.DataFrame, set_type: int) -> None:
    """
    preprocess the 'cast' feature for the vote_avg DataFrame
    :param data: DataFrame to be processed
    :param set_type: whether the DataFrame is the train df or the test df
    :return: None
    """
    global vote_actors_sets
    if is_train(set_type):
        vote_actors_sets, actors_names = preprocessing_cast_and_crew(data, "cast", "vote_average")
    else:
        actors_names = preprocessing_cast_and_crew(data, "cast", "vote_average")[1]
    for i, s in enumerate(vote_actors_sets, 1):
        data[f'actor_level_{i}'] = np.zeros(len(data))
        for j, names in enumerate(actors_names):
            for name in names:
                data.loc[j, f'actor_level_{i}'] += 1 if name in s else 0


def preprocess_json(data: pd.DataFrame, set_type: int) -> None:
    """
    preprocess the 'crew' and 'cast' feature
    :param data: data
    :param set_type: type
    :return: None
    """
    preprocess_revenue_crew(data, set_type)
    preprocess_revenue_cast(data, set_type)
    preprocess_vote_crew(data, set_type)
    preprocess_vote_cast(data, set_type)

#______________________________________________________________________________________#

############################### Training Models ########################################

def train_lasso(df: pd.DataFrame, y: pd.DataFrame) -> Lasso:
    """
    Helper function to create a Lasso model
    :param df: DataFrame
    :param y: response feature
    :return: trained Lasso model
    """
    lasso = Lasso(alpha=1)
    lasso.fit(df, y)
    return lasso


def estimate_RMSE(y, y_hat) -> float:
    """
    estimates the RMSE error
    :param y: true labels
    :param y_hat: predictions labels
    :return: error
    """
    return np.sqrt(sklearn.metrics.mean_squared_error(y, y_hat))

# NOTE: we left only the chosen model with the best results according to the RMSE loss function
# The models were imported from sklearn lib
#_____________________________________________________________________________________#


if __name__ == '__main__':
    # PART 1: This part is to be used once to train the model and serialize it, so we do not have to train the model
    # each time we want a single prediction
    # NOTE: After first use COMMENT this part and use part 2 only. Use this part everytime needed to reset the model
    # and train it over new set of data

    a = load_data('movies_dataset.csv')
    b = load_data('movies_dataset_part2.csv')

    # preprocessing loaded data using the preprocess functions
    df_revenue = pd.concat([a, b], axis=0, ignore_index=True)
    df_vote = pd.concat([a, b], axis=0, ignore_index=True)

    df_revenue = preprocess_train_revenue(df_revenue)
    df_vote = preprocess_train_vote(df_vote)
    vote_average = df_vote[["vote_average"]]
    revenue = df_vote[["revenue"]]

    df_revenue.drop(labels=['vote_average', 'revenue'], axis=1, inplace=True)
    df_vote.drop(labels=['vote_average', 'revenue'], axis=1, inplace=True)

    # Creating models for each feature we want to predict and training the models
    randomForestRegressor_vote_average = RandomForestRegressor(n_estimators=100, min_samples_split=30, max_depth=12)
    randomForestRegressor_revenue = RandomForestRegressor(n_estimators=100, min_samples_split=30, max_depth=4)
    randomForestRegressor_vote_average.fit(df_vote, vote_average)
    randomForestRegressor_revenue.fit(df_revenue, revenue)

    mean_values = np.array([revenue.mean(), vote_average.mean()])

    globals_array = np.array([top_2_languages, revenue_directors_sets, revenue_actors_sets, vote_directors_sets, vote_actors_sets])
    serialize.dump(randomForestRegressor_revenue, open('revenue_model.sav', 'wb'))
    serialize.dump(randomForestRegressor_vote_average, open('vote_rate_model.sav', 'wb'))
    serialize.dump(mean_values, open('mean_values.sav', 'wb'))
    serialize.dump(globals_array, open('globals_array.sav', 'wb'))
    # # END OF PART 1

    # PART 2: This part is to be used every time we want to make a new prediction, given a test_data as a csv file
    # In case you want to get a prediction on a different data set- just replace the relative path in the TEST_FILE
    # constant declared at the top of the file
    revenue_predictions, vote_predictions = predict(TEST_FILE)
    print("REVENUE PREDICTIONS:")
    print(revenue_predictions, end='\n\n')
    print("VOTE AVERAGE PREDICTIONS:")
    print(vote_predictions)


