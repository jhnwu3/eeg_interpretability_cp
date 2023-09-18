import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.st_transformer import *
import cv2
# rule 5 from paper
def avg_heads(cam, grad):
    # force shapes of cam and grad to be the same order
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]) 
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam # elementwise mult
    cam = cam.clamp(min=0).mean(dim=0) # average
    return cam

# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    return torch.matmul(cam_ss, R_ss)

def apply_cross_attention_rules(R_st, cam_st):
    return None # to be implemented

def generate_relevance(model, input, index=None):
    output = model(input, register_hook=True)
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    num_tokens = model.transformer_blocks[0].attention.get_attn_map().shape[-1]
    print("num_tokens:", num_tokens)
    R = torch.eye(num_tokens, num_tokens).cuda() # initialize identity matrix
    for blk in model.transformer_blocks:
        grad = blk.attention.get_attn_grad()
        print(grad.size())
        cam = blk.attention.get_attn_map()
        print(cam.size())
        cam = avg_heads(cam, grad)
        
        R += apply_self_attention_rules(R.cuda(), cam.cuda())
    # to get all true relative relevances, subtract identity matridx

    return R[0,1:] # return the first vector of self attention not including the "self" "self" part

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization(original_image, model,  class_index=None):
    transformer_attribution = generate_relevance(model, original_image.unsqueeze(0).cuda(), index=class_index).detach()
   
    # since attn is a square matrix
    s = int(np.sqrt(transformer_attribution.size()[0]))
    # assuming c x h x w
    sf = int(original_image.size()[1] / s) # get the scaling factor

    transformer_attribution = transformer_attribution.reshape((1, 1, s, s)) # reshape into n,c,h,w format of tokens 
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=sf, mode="bilinear")
    transformer_attribution = transformer_attribution.reshape(original_image.size()[1], original_image.size()[2]).cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()

    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis



# class CheferLanguageExplaienr():
class CheferVisionExplainer():
    def __init__(self, model):
        self.model = model
    
    def get_relevance_matrix(self, input, index):
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        num_tokens = self.model.transformer_blocks[0].attention.get_attn_map().shape[-1]

        R = torch.eye(num_tokens, num_tokens).cuda() # initialize identity matrix
        for blk in self.model.transformer_blocks:
            grad = blk.attention.get_attn_grad()
            cam = blk.attention.get_attn_map()
            cam = avg_heads(cam, grad)
            R += apply_self_attention_rules(R.cuda(), cam.cuda())
        # to get all true relative relevances, subtract identity matrix, but with cls token, one can just take
        # the first row of the relevance matrix excluding the attention related to the cls token
        return R[0,1:] # return the first vector of self attention not including the "self" "self" part



    def get_relevance_scores(self):
        return None
    def visualize(self):
        return None 
    
# STT transformer assumes no cls token so we average it at the end
class STTransformerInterpreter():
    def __init__(self, model : STTransformer):
        self.model = model.cuda()

    def show_cam_on_image(self, img, mask, save_path = None, figsize=(100,20)):
        plt.figure(figsize=figsize)
        plt.imshow(img, cmap='gray')
        plt.imshow(mask, cmap='plasma', alpha=0.35)
        plt.colorbar(shrink=0.5)
        if save_path is not None: 
            plt.savefig(save_path)

    def get_relevance_matrix(self, input, index):
        output = self.model(input.cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        # one_hot.backward()

        num_tokens = self.model.transformer.transformer_blocks[0].mhattn.get_attn_map().shape[-1]

        R = torch.eye(num_tokens, num_tokens).cuda() # initialize identity matrix
        for blk in self.model.transformer.transformer_blocks:
            grad = blk.mhattn.get_attn_grad()
            cam = blk.mhattn.get_attn_map()
            cam = avg_heads(cam, grad)
            R += apply_self_attention_rules(R, cam).detach()
        
        # to get all true relative relevances, subtract identity matrix, but with cls token, one can just take
        # the first row of the relevance matrix excluding the attention related to the cls token
        R -= torch.eye(num_tokens, num_tokens).cuda()
        return torch.mean(R, dim=0).detach() # since no cls token, just sum or average it


    # need a separate relevance computation for relevance computed across each channel
    def get_channel_relevance(self, input, index):
        output = self.model(input.cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        num_tokens = self.model.channel_attention.get_attn_map().shape[-1]

        R = torch.eye(num_tokens, num_tokens).cuda() # initialize identity matrix
        # print(R)
        grad = self.model.channel_attention.get_attn_grad()
        cam = self.model.channel_attention.get_attn_map()
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R, cam).detach()
    
        # cam = None
        # self.model.zero_grad(set_to_none=True)
        # to get all true relative relevances, subtract identity matrix, but with cls token, one can just take
        # the first row of the relevance matrix excluding the attention related to the cls token
        R -= torch.eye(num_tokens, num_tokens).cuda()
        return torch.mean(R, dim=0).detach() # since no cls token, just sum or average it

    # scales attribution to a certain scale and then converts it to numpy for ease of printing
    # assumes temporal nature of interpolation
    def scale_relevance_scores_vis(self, input, out_length, method=""):
        # interpolate such that we can get a bigger 1D vector of weights or scores for each chunk
        # since attn is a square matrix
        s = input.size()[0]
        # since we know that sequence is C x L, divide and ceiling it to recreate the "convolved zones" 
        size_chunk = int(np.ceil(out_length / s)) # get the scaling factor i.e size of each chunk of the sequence
        
        # transformer_attribution = transformer_attribution.cpu().numpy()
        attribution_scores = []
        if method == "linear":
            input = input.unsqueeze(0).unsqueeze(0) # so we can get 1 x 1 x T dimensions for interpolation
            attribution_scores = F.interpolate(input, size=out_length, mode="linear")
            attribution_scores = attribution_scores.squeeze().squeeze().detach().cpu().numpy()
        else: # naive interpolation
            input = input.detach().cpu().numpy()
            for element in range(s):
                for s in range(size_chunk):
                    attribution_scores.append(input[element])

            # compute number of elements overboard, and pop some of them out I guess (need to confirm if this even makes any sense)
            # I might retrain a model that uses embeddings with nice round numbers..
            nPop = 0
            if out_length != len(attribution_scores):
                nPop = len(attribution_scores) - out_length
                for i in range(nPop):
                    attribution_scores.pop()

            attribution_scores = np.array(attribution_scores)
            
        return attribution_scores
    
    def scale_rel_scores(self, input, out_length, method=""):
        # interpolate such that we can get a bigger 1D vector of weights or scores for each chunk
        # since attn is a square matrix
        s = input.size()[0]
        # since we know that sequence is C x L, divide and ceiling it to recreate the "convolved zones" 
        # transformer_attribution = transformer_attribution.cpu().numpy()
        attribution_scores = []
        if method == "linear":
            input = input.unsqueeze(0).unsqueeze(0) # so we can get 1 x 1 x T dimensions for interpolation
            attribution_scores = F.interpolate(input, size=out_length, mode="linear")
            attribution_scores = attribution_scores
            attribution_scores = attribution_scores.squeeze(0).squeeze(0)# squeeze it back to a normal dim
        else: # naive interpolation where every chunk is simply just what we want
            size_chunk = int(np.ceil(out_length / s)) # get the scaling factor i.e size of each chunk of the sequence
            for element in range(s):
                for s in range(size_chunk):
                    attribution_scores.append(input[element])
            # compute number of elements overboard, and pop some of them out I guess (need to confirm if this even makes any sense)
            # I might retrain a model that uses embeddings with nice round numbers..
            nPop = 0
            if out_length != len(attribution_scores):
                nPop = len(attribution_scores) - out_length
                for i in range(nPop):
                    attribution_scores.pop()

            attribution_scores = torch.tensor(attribution_scores)

        return attribution_scores
    
    # assume sequence is B x C x D
    def visualize(self, sequence, class_index=None, save_path = None, figsize=(100,20), method = "", show=True):
        transformer_attribution = self.get_relevance_matrix(sequence, index=class_index).detach()
        
        transformer_attribution = self.scale_relevance_scores_vis(transformer_attribution, sequence.size()[2], method=method)
        # transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
        # want to make sure it matches the sequence dimensions!
        transformer_attribution = np.tile(transformer_attribution, (sequence.size()[1],1))# unsqueeze

        # get channel attention
        channel_attribution = self.get_channel_relevance(sequence, index=class_index).detach().cpu().numpy()
        # normalize s.t sum to 1 as well
        # channel_attribution = (channel_attribution - channel_attribution.min()) / (channel_attribution.max() - channel_attribution.min())
        channel_attribution = channel_attribution[:,np.newaxis]# reshape into columnwise addition

        # combine channel attribution to transformer attribution across each channel.
        transformer_attribution = transformer_attribution + channel_attribution
        # normalize again
        transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

        sequence = sequence.cpu().numpy()
        # normalize such that the seq w/ 16 channels sums to 1! 
        sequence = (sequence - sequence.min()) / (sequence.max() - sequence.min())
        sequence = sequence.reshape(transformer_attribution.shape)
        if show:
            self.show_cam_on_image(sequence, transformer_attribution, save_path=save_path, figsize=figsize)

        return sequence, transformer_attribution
    
    # implement in torch for fast fast implementation!
    def get_cam_on_image(self, sequence, class_index= None, method = ""):

        channel_length = sequence.size()[2]
        n_channels = sequence.size()[1]
        sequence = sequence.cuda()
        # transformer_attribution = torch.zeros(sequence.size())
        transformer_attribution = self.get_relevance_matrix(sequence, index=class_index).detach()
        transformer_attribution = self.scale_rel_scores(transformer_attribution, channel_length, method=method)
        # want to make sure it matches the sequence dimensions!
        transformer_attribution = transformer_attribution.repeat(n_channels, 1)

        # get channel attention
        channel_attribution = self.get_channel_relevance(sequence, index=class_index)
        # channel_attribution = torch.zeros(sequence.size())
        channel_attribution = channel_attribution.unsqueeze(1)# reshape into columnwise addition

        # combine channel attribution to transformer attribution across each channel.
        transformer_attribution += channel_attribution
        # normalize again
        transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

        return transformer_attribution * sequence


    def get_top_classes(self, input, class_labels = None, n=5):
        softmax = np.squeeze(self.model(input.cuda()).detach().cpu().numpy())
    
        sorted_indices = np.argsort(-softmax)
        
        if class_labels is not None:
            print("Top 5 Scores in Increasing Order")
            for i in range(n):
                print(f"Class {class_labels[sorted_indices[i]]} with score: {softmax[sorted_indices[i]]}")
        else: 
            print('Need A Map of Labels to Print')
