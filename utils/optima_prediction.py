def optima_prediction(min_predictions, max_predictions):
    min_predictions = (min_predictions[0] > 0.5).float()
    max_predictions = (max_predictions[0] > 0.5).float()

    list = []
    for idx_max, max in enumerate(max_predictions):
        if max == 0:
            list.append(2)
        else:
            list.append(max)
    max_predictions = list
    for idx_min, min in enumerate(min_predictions):
        if min == 1:
            if max_predictions[idx_min] != min:
                print("Error")
            max_predictions[idx_min] = 0
    predictions = max_predictions
    return predictions