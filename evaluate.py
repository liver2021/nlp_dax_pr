import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate_model(model, X_test, y_test, scaler, device, classification_mode = False, rf=False, auc=False):

    if rf :
        x = X_test.shape[0]
        X_test = np.reshape(X_test, (x, -1))
        y_pred = model.predict(X_test)

        if classification_mode :
            roc_auc = roc_auc_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            metric_1, metric_2, metric_3 = -roc_auc, accuracy, f1

        else:
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
            rmse = np.sqrt(np.mean((y_pred_inv - y_test_inv) ** 2))
            mae = mean_absolute_error(y_test_inv, y_pred_inv)

            r2 = r2_score(y_test_inv, y_pred_inv)
            metric_1, metric_2, metric_3 = rmse, mae, r2




    else:
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            y_pred = model(X_test)[0]

            if classification_mode:
                y_pred_tensor = y_pred
                if not auc:

                    BCELoss = nn.BCELoss()
                    if len(y_test.shape) > 1:
                        y_test = y_test.squeeze()

                    BCELoss = BCELoss(y_pred_tensor.squeeze(), y_test)
                    metric_1 = BCELoss
                else:
                    roc_auc = roc_auc_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
                    metric_1 = roc_auc


                y_pred = (y_pred_tensor > 0.5).float()
                accuracy = accuracy_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
                f1 = f1_score(y_test.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
                metric_2, metric_3 = accuracy, f1




            else:
                y_test_inv = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))
                y_pred_inv = scaler.inverse_transform(y_pred.cpu().numpy())

                rmse = np.sqrt(np.mean((y_pred_inv - y_test_inv) ** 2))
                mae = mean_absolute_error(y_test_inv, y_pred_inv)

                r2 = r2_score(y_test_inv, y_pred_inv)
                metric_1, metric_2, metric_3 = rmse, mae, r2



    return metric_1, metric_2, metric_3

