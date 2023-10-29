import torch
import torch.nn.functional as F
from torch import nn
from utils import *
import torchvision
from torch import optim
import pytorch_lightning as pl
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import roi_align
from utils import compute_iou
from faster_rcnn_datamodule import DataModule
from pretrained_models import pretrained_models_680 
class BoxHead(pl.LightningModule):
    def __init__(self,Classes=3,P=7, pretrained_path='checkpoint680.pth'):
        super().__init__()
        self.C=Classes
        self.P=P
        self.backbone, self.rpn = pretrained_models_680(pretrained_path)
        self.strides = [4, 8, 16, 32]        
        # TODO initialize BoxHead

        self.intermediate_layer = nn.Sequential(
            nn.Linear(P*P*256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024), 
            nn.ReLU()
        )
    
        self.clasifier_head = nn.Sequential(
            nn.Linear(1024, Classes+1), 
            nn.Softmax(dim=1)
        )
        
        self.regressor_head = nn.Sequential(
            nn.Linear(1024, 4*(Classes))
        )
            
    #  This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
    #  Input:
    #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #  Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
    #       labels: (total_proposals,1) (the class that the proposal is assigned)
    #       regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
    def create_ground_truth(self,proposals,gt_labels,bboxes):
        labels = []
        regressor_targets = []
        batch_size = len(proposals)
        for i in range(batch_size):
            num_proposals = len(proposals[i])
            regressor_target = torch.zeros_like(proposals[i])
            label = torch.zeros(num_proposals, 1, device=regressor_target.device)
            ious = compute_iou(proposals[i], bboxes[i]) 
            max_values, max_indices = torch.max(ious, axis=1)
            above_thres_indices = max_values > 0.5
            label[above_thres_indices] = gt_labels[i][max_indices[above_thres_indices]].view(-1,1).to(torch.float32)
            regressor_target[above_thres_indices] = bboxes[i][max_indices[above_thres_indices]]
            labels.append(label)
            regressor_targets.append(regressor_target)
        return labels,regressor_targets



    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
    def MultiScaleRoiAlign(self, fpn_feat_list,proposals,P=7):
        #####################################
        # Here you can use torchvision.ops.RoIAlign check the docs
        #####################################
        feature_vectors = []
        batch_size = len(proposals) 
        for i in range(batch_size):
            width = (proposals[i][:, 2] -  proposals[i][:, 0])
            height = (proposals[i][:, 3] -  proposals[i][:, 1]) 
            k = torch.floor(4 + torch.log2((width*height).sqrt()/224))
            k = torch.clamp(k, 2)
            for j in range(2, 6):
                feature_vectors.append(roi_align(fpn_feat_list[j-2][i:i+1], [proposals[i][k==j]], P, spatial_scale=self.strides[j-2]))
        feature_vectors = torch.vstack(feature_vectors)
        feature_vectors = feature_vectors.reshape(feature_vectors.shape[0], -1)
        return feature_vectors



    # This function does the post processing for the results of the Box Head for a batch of images
    # Use the proposals to distinguish the outputs from each image
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
    #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
    #       conf_thresh: scalar
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50):

        return boxes, scores, labels




    # Compute the total loss of the classifier and the regressor
    # Input:
    #      class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
    #      box_preds: (total_proposals,4*C)      (as outputed from forward)
    #      labels: (total_proposals,1)
    #      regression_targets: (total_proposals,4)
    #      l: scalar (weighting of the two losses)
    #      effective_batch: scalar
    # Outpus:
    #      loss: scalar
    #      loss_class: scalar
    #      loss_regr: scalar
    def compute_loss(self,class_logits, box_preds, labels, regression_targets,l=1,effective_batch=150):

        return loss, loss_class, loss_regr



    # Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
    # Input:
    #        feature_vectors: (total_proposals, 256*P*P)
    # Outputs:
    #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
    #                                               CrossEntropyLoss you should not pass the output through softmax here)
    #        box_pred:     (total_proposals,4*C)
    def forward(self, feature_vectors):
        intermediate_output = self.intermediate_layer(feature_vectors)
        class_logits = self.clasifier_head(intermediate_layer)
        box_pred = self.regressor_head(intermediate_layer)
        return class_logits, box_pred
    
    def training_step(self, batch, batch_idx):
        images, labels, _, bboxes, indices = batch
        batch_size = len(images) 
        self.backbone.eval()
        self.rpn.eval()
        # Take the features from the backbone
        backout = self.backbone(images)

        # The RPN implementation takes as first argument the following image list
        im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
        # Then we pass the image list and the backbone output through the rpn
        rpnout = self.rpn(im_lis, backout)

    
        keep_topK = 200
        # A list of proposal tensors: list:len(bz){(keep_topK,4)}
        proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]
        # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
        fpn_feat_list= list(backout.values())
       
        for i in range(batch_size):
            width = proposals[i][:, 2] - proposals[i][:, 0]
            height = proposals[i][:, 3] - proposals[i][:, 1]
            area = width * height
            sorted_indices = torch.argsort(area) 
            proposals[i] = proposals[i][sorted_indices]
        
            
        self.create_ground_truth(proposals, labels, bboxes)
    
        feature_vectors = self.MultiScaleRoiAlign(fpn_feat_list,proposals,P=7)
        class_logits, box_pred = self(feature_vectors)
   
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
     
if __name__ == '__main__':
    imgs_path = '../partA/data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '../partA/data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = '../partA/data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = '../partA/data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    datamodule = DataModule(paths) 
    box_head= BoxHead() 
    
    trainer = pl.Trainer(max_epochs=40, devices=1)
    trainer.fit(box_head, datamodule=datamodule)
