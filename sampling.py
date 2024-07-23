import torch


def entropy_sampling(xyz, points, nBins, npoint):
    """
    Input:
        xyz: (batch_size, number of data points, 3)
        points: (batch_size, number of data points, number of features)
        nBins: number of bins for histogram
        npoint: number of samples
    Return:
        centroids: sampled pointcloud points, [B, npoint]
    """
    sampled_xyz_batches = []
    sampled_feats_batches = []
    for bidx in range(points.shape[0]):
        # build histogram and sort by number of elements
        minval = torch.min(points[bidx][...,0])
        maxval = torch.max(points[bidx][...,0])
        hist = torch.histc(points[bidx][...,0], bins=nBins, min=minval, max=maxval)
        bin_min = minval
        bin_width = (maxval - minval) / nBins
        suffle_idx = torch.randperm(points[bidx].size()[0])
        shuffle_xyz = xyz[bidx][suffle_idx]
        shuffle_feat = points[bidx][suffle_idx]
        sort_hist = []
        for i in range(len(hist)):
            sort_hist.append(torch.Tensor([hist[i], minval + bin_width*i, minval + bin_width*(i+1)]))
        sort_hist = torch.stack(sort_hist)
        hist, ind = sort_hist.sort(0)
        sort_hist = sort_hist[ind[...,0]]

        # compute the number of points for each bin
        sampled_xyz = []
        sampled_feats = []
        sampled_hist_list = [0 for i in range(nBins)]
        sampled_hist = []
        curSamples = npoint
        curBins = nBins
        for i in range(nBins):
            npoint_bin = curSamples // curBins
            npoint_bin = min(npoint_bin, sort_hist[i][0].to(int))
            sampled_hist_list[i] = npoint_bin
            curSamples -= npoint_bin
            curBins -= 1
            sampled_hist.append(torch.Tensor([sort_hist[i][1], sort_hist[i][2], sampled_hist_list[i]]))
        sampled_hist = torch.stack(sampled_hist)
        hist, ind = sampled_hist.sort(0)
        sort_hist = sampled_hist[ind[...,0]]
        sampled_hist_list = sort_hist[...,2]
        
        # select points
        for i in range(shuffle_xyz.shape[0]):
            idx = min(torch.floor((shuffle_feat[i][0] - bin_min)/bin_width).to(int), nBins-1)
            if sampled_hist_list[idx] > 0:
                sampled_xyz.append(shuffle_xyz[i])
                sampled_feats.append(shuffle_feat[i])
                sampled_hist_list[idx] -= 1
            if torch.count_nonzero(sampled_hist_list) == 0:
                break
        sampled_xyz = torch.stack(sampled_xyz)
        sampled_feats = torch.stack(sampled_feats)
        sampled_xyz_batches.append(sampled_xyz)
        sampled_feats_batches.append(sampled_feats)
    sampled_xyz_batches = torch.stack(sampled_xyz_batches)
    sampled_feats_batches = torch.stack(sampled_feats_batches)
    return sampled_xyz_batches, sampled_feats_batches


# sanity check
x = torch.linspace(-1, 1, steps=10)  
y = torch.linspace(-1, 1, steps=10)  
z = torch.linspace(-1, 1, steps=10)
X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
batch_size = 16
X = X.reshape(-1).view(1, -1, 1).repeat(batch_size, 1, 1)
Y = Y.reshape(-1).view(1, -1, 1).repeat(batch_size, 1, 1)
Z = Z.reshape(-1).view(1, -1, 1).repeat(batch_size, 1, 1)

xyz = torch.cat((X,Y,Z), dim=2)
points = torch.rand_like(X)
nBins = 50
npoint = 128

xyz_samples, feature_samples = entropy_sampling(xyz, points, nBins, npoint)

print(xyz_samples.shape)
print(feature_samples.shape)