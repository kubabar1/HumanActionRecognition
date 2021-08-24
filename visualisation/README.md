## Visualisation module

### Drawing chart
Draw chart of accuracy or loss values for train an validation basis on *.npy files generated during Human Action Recognition NN training.

Parameters:
* train-path - absolute path to *.npy file with train results
* test-path - absolute path to *.npy file with test results
* step - step size

Example:
```
python draw_chart.py \
    --train-path '../results/lstm_simple_ep_10000_b_128_h_128_lr_1e-05_RMSPROP_train_acc.npy' \
    --test-path '../results/lstm_simple_ep_10000_b_128_h_128_lr_1e-05_RMSPROP_val_acc.npy' \
    --step 50
```