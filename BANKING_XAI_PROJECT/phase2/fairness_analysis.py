from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference


def evaluate_fairness(y_true, y_pred, sensitive_features):

    dp = demographic_parity_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive_features
    )

    eo = equalized_odds_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive_features
    )

    return {
        'demographic_parity_difference': dp,
        'equalized_odds_difference': eo
    }