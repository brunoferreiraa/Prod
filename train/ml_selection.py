import math
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.neural_network import MLPRegressor


def ml_selection(X, Y, X_test, y_test, pickle_name):
    # Define the regressors
    classifier1 = ExtraTreesRegressor()
    classifier2 = GradientBoostingRegressor()
    classifier3 = RandomForestRegressor()
    classifier4 = BaggingRegressor()
    classifier5 = linear_model.Ridge()
    classifier6 = linear_model.LassoLars()
    classifier7 = linear_model.LinearRegression()
    classifier8 = LogisticRegression()
    classifier9 = LinearDiscriminantAnalysis()
    classifier10 = QuadraticDiscriminantAnalysis()
    classifier11 = KNeighborsRegressor()
    classifier12 = DecisionTreeRegressor()
    classifier13 = GaussianNB()
    classifier14 = MultinomialNB()
    classifier15 = BernoulliNB()
    classifier16 = SGDRegressor()
    classifier17 = PassiveAggressiveRegressor()
    classifier18 = AdaBoostRegressor()
    classifier19 = SVR()
    classifier20 = MLPRegressor()

    res = float('inf')
    ml_name = "RandomForestRegressor"  # to ensure a model for unittests
    for r in range(1):
        # ETC______________________________________________________________________________________________________________
        try:
            name = "ExtraTreesRegressor"
            print(name)
            classifier1.fit(X, Y)
            y_pred1 = classifier1.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred1)
            mse = mean_squared_error(y_test, y_pred1)
            r2 = r2_score(y_test, y_pred1)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # SVR______________________________________________________________________________________________________________
        try:
            name = "GradientBoostingRegressor"
            print(name)
            classifier2.fit(X, Y)
            y_pred2 = classifier2.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred2)
            mse = mean_squared_error(y_test, y_pred2)
            r2 = r2_score(y_test, y_pred2)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # KNN______________________________________________________________________________________________________________
        try:
            name = "RandomForestRegressor"
            print(name)
            classifier3.fit(X, Y)
            y_pred3 = classifier3.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred3)
            mse = mean_squared_error(y_test, y_pred3)
            r2 = r2_score(y_test, y_pred3)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # RF______________________________________________________________________________________________________________
        try:
            name = "BaggingRegressor"
            print(name)
            classifier4.fit(X, Y)
            y_pred4 = classifier4.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred4)
            mse = mean_squared_error(y_test, y_pred4)
            r2 = r2_score(y_test, y_pred4)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # RIDGE______________________________________________________________________________________________________________
        try:
            name = "Ridge"
            print(name)
            classifier5.fit(X, Y)
            y_pred5 = classifier5.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred5)
            mse = mean_squared_error(y_test, y_pred5)
            r2 = r2_score(y_test, y_pred5)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # LASSO_LARS______________________________________________________________________________________________________________
        try:
            name = "LassoLars"
            print(name)
            classifier6.fit(X, Y)
            y_pred6 = classifier6.predict(X_test)
            print("\n\n")
            # METRICS
            mae = mean_absolute_error(y_test, y_pred6)
            mse = mean_squared_error(y_test, y_pred6)
            r2 = r2_score(y_test, y_pred6)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # LINEAR_REGRESSION__________________________________________________________________
        try:
            name = "LinearRegression"
            print(name)
            classifier7.fit(X, Y)
            y_pred7 = classifier7.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred7)
            mse = mean_squared_error(y_test, y_pred7)
            r2 = r2_score(y_test, y_pred7)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # LOGISTIC_REGRESSION_________________________________________________________________________________
        try:
            name = "LogisticRegression"
            print(name)
            classifier8.fit(X, Y)
            y_pred8 = classifier8.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred8)
            mse = mean_squared_error(y_test, y_pred8)
            r2 = r2_score(y_test, y_pred8)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # LINEAR_DISCRIMINANT_ANALYSIS_______________________________________________________________________________________
        try:
            name = "LinearDiscriminantAnalysis"
            print(name)
            classifier9.fit(X, Y)
            y_pred9 = classifier9.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred9)
            mse = mean_squared_error(y_test, y_pred9)
            r2 = r2_score(y_test, y_pred9)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # QuadraticDiscriminantAnalysis__________________________________________________________________________________________
        try:
            name = "QuadraticDiscriminantAnalysis"
            print(name)
            classifier10.fit(X, Y)
            y_pred10 = classifier10.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred10)
            mse = mean_squared_error(y_test, y_pred10)
            r2 = r2_score(y_test, y_pred10)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # KNeighborsRegressor______________________________________________________________________________________________________________
        try:
            name = "KNeighborsRegressor"
            print(name)
            classifier11.fit(X, Y)
            y_pred11 = classifier11.predict(X_test)

            # METRICS
            mae = mean_absolute_error(y_test, y_pred11)
            mse = mean_squared_error(y_test, y_pred11)
            r2 = r2_score(y_test, y_pred11)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # DecisionTreeRegressor______________________________________________________________________________________________________________
        try:
            name = "DecisionTreeRegressor"
            print(name)
            classifier12.fit(X, Y)
            y_pred12 = classifier12.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred12)
            mse = mean_squared_error(y_test, y_pred12)
            r2 = r2_score(y_test, y_pred12)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # GaussianNB______________________________________________________________________________________________________________
        try:
            name = "GaussianNB"
            print(name)
            classifier13.fit(X, Y)
            y_pred13 = classifier13.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred13)
            mse = mean_squared_error(y_test, y_pred13)
            r2 = r2_score(y_test, y_pred13)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # MultinomialNB______________________________________________________________________________________________________________
        try:
            name = "MultinomialNB"
            print(name)
            classifier14.fit(X, Y)
            y_pred14 = classifier14.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred14)
            mse = mean_squared_error(y_test, y_pred14)
            r2 = r2_score(y_test, y_pred14)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # BernoulliNB______________________________________________________________________________________________________________
        try:
            name = "BernoulliNB"
            print(name)
            classifier15.fit(X, Y)
            y_pred15 = classifier15.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred15)
            mse = mean_squared_error(y_test, y_pred15)
            r2 = r2_score(y_test, y_pred15)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # SGDRegressor______________________________________________________________________________________________________________
        try:
            name = "SGDRegressor"
            print(name)
            classifier16.fit(X, Y)
            y_pred16 = classifier16.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred16)
            mse = mean_squared_error(y_test, y_pred16)
            r2 = r2_score(y_test, y_pred16)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # PassiveAggressiveRegressor______________________________________________________________________________________________________________
        try:
            name = "PassiveAggressiveRegressor"
            print(name)
            classifier17.fit(X, Y)
            y_pred17 = classifier17.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred17)
            mse = mean_squared_error(y_test, y_pred17)
            r2 = r2_score(y_test, y_pred17)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # AdaBoostRegressor______________________________________________________________________________________________________________
        try:
            name = "AdaBoostRegressor"
            print(name)
            classifier18.fit(X, Y)
            y_pred18 = classifier18.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred18)
            mse = mean_squared_error(y_test, y_pred18)
            r2 = r2_score(y_test, y_pred18)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # SVR______________________________________________________________________________________________________________
        try:
            name = "SVR"
            print(name)
            classifier19.fit(X, Y)
            y_pred19 = classifier19.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred19)
            mse = mean_squared_error(y_test, y_pred19)
            r2 = r2_score(y_test, y_pred19)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

        # MLPRegressor______________________________________________________________________________________________________________
        try:
            name = "MLPRegressor"
            print(name)
            classifier20.fit(X, Y)
            y_pred20 = classifier20.predict(X_test)
            # METRICS
            mae = mean_absolute_error(y_test, y_pred20)
            mse = mean_squared_error(y_test, y_pred20)
            r2 = r2_score(y_test, y_pred20)
            print('Mean Absolute Error:', mae)
            print('Mean Squared Error:', mse)
            print('R-squared scores:', r2)
            print('RMSE', math.sqrt(mse))
            print('RMAE', math.sqrt(mae))
            print("\n")

            if res > mse:
                res = (mse)
                ml_name = name

        except Exception as ex:
            print(ex)
            pass

    # TODO - REVER: PARA OS UNITTESTS
    if ml_name == "":
        ml_name = "RandomForestRegressor"
    # TODO - REVER: PARA OS UNITTESTS

    # Print model name and results
    print("Best model: " + ml_name + "with a RMSE of: " + math.sqrt(res))

    # Call optimization and pickle creation
    ml_optimization_and_train(X, Y, X_test, y_test, pickle_name, ml_name)
    # return ml_name


def ml_optimization_and_train(X, Y, X_test, y_test, pickle_name, ml_name):
    if ml_name == "ExtraTreesRegressor":
        print("x")
        # grid_search com parametros a procurar
        # criação do pickle

    # elif ml_name == "GradientBoostingRegressor":
    #     XXX

    # elif ml_name == "RandomForestRegressor":
    #     XXX

    # elif ml_name == "BaggingRegressor":
    #     XXX

    # elif ml_name == "Ridge":
    #     XXX

    # elif ml_name == "LassoLars":
    #     XXX

    # elif ml_name == "LinearRegression":
    #     XXX

    # elif ml_name == "LogisticRegression":
    #     XXX

    # elif ml_name == "LinearDiscriminantAnalysis":
    #     XXX

    # elif ml_name == "QuadraticDiscriminantAnalysis":
    #     XXX

    # elif ml_name == "KNeighborsRegressor":
    #     XXX

    # elif ml_name == "DecisionTreeRegressor":
    #     XXX

    # elif ml_name == "GaussianNB":
    #     XXX

    # elif ml_name == "MultinomialNB":
    #     XXX

    # elif ml_name == "BernoulliNB":
    #     XXX

    # elif ml_name == "SGDRegressor":
    #     XXX

    # elif ml_name == "PassiveAggressiveRegressor":
    #     XXX

    # elif ml_name == "AdaBoostRegressor":
    #     XXX

    # elif ml_name == "SVR":
    #     XXX

    # elif ml_name == "MLPRegressor":
    #     XXX

    # else:
    #     print("ERROR: ML selection")

    # pickle.dump(model, open(pickle_name, 'wb'))
