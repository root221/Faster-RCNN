## Image plots of the proposals with the no-background classes and their ground truth boxes
### The thick solid boxes represent the ground truth, while the thin dashed boxes are the proposals
![Proposals](plots/proposals.png)
![Proposals](plots/proposals_2.png)



## Training and Validation curves of the total loss, the loss of the classifier, and the loss of the regressor of the Box Head
![Training Plot](plots/training_plot.png)
![Validation Plot](plots/val_plot.png)

Please note that the plots start from epoch 0, so when referring to 'epoch 14' on the plot, it actually corresponds to epoch 15 in the training process

## AP and mAP Scores of the Test Set
During the training process, we saved model checkpoints at epoch 15 and at the end of training, which is epoch 30. We obtained the following Average Precision (AP) values for the test set using these two models:

**Epoch 15:**
- The Average Precision for the 'Vehicles' class is: 0.923
- The Average Precision for the 'People' class is: 0.970
- The Average Precision for the 'Animals' class is: 0.937
- The mAP is 0.943

**Epoch 30:**
- The Average Precision for the 'Vehicles' class is: 0.931
- The Average Precision for the 'People' class is: 0.969
- The Average Precision for the 'Animals' class is: 0.94
- The mAP is 0.946


## Image plots that contain the top 20 boxes produced by the Box Head before NMS
![Pre NMS Results](plots/pre_nms_results.png)

## Image plots of the regressed boxes after the postprocessing
![Postprocessing Results](plots/postprocessing_results.png)
![Postprocessing Results](plots/postprocessing_results_2.png)
