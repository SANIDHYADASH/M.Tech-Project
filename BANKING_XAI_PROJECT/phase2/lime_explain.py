from lime.lime_tabular import LimeTabularExplainer


def explain_with_lime(model, X_train, X_test, feature_names):

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=['Low Risk', 'High Risk'],
        mode='classification'
    )

    explanation = explainer.explain_instance(
        X_test[0],
        model.predict,
        num_features=10
    )

    explanation.save_to_file('artifacts/lime/lime_explanation.html')