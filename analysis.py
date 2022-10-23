import torch
import torchvision
from tqdm.auto import tqdm

import common_utils
from common_utils.image import get_ssim_all, get_ssim_pairs_kornia
from evaluations import l2_dist, ncc_dist, normalize_batch, transform_vmin_vmax_batch


@torch.no_grad()
def get_dists(x, y, search, use_bb):
    """D: x -> y"""
    xxx = x.clone()
    yyy = y.clone()

    # Search Real --> Extracted
    if search == 'l2':
        D = l2_dist(xxx, yyy, div_dim=True)
    if search == 'ncc':
        D = ncc_dist(xxx, yyy, div_dim=True)
    elif search == 'ncc2':
        x2search = torch.nn.functional.interpolate(xxx, scale_factor=1 / 2, mode='bicubic', align_corners=False)
        y2search = torch.nn.functional.interpolate(yyy, scale_factor=1 / 2, mode='bicubic', align_corners=False)
        D = ncc_dist(x2search, y2search, div_dim=True)
    elif search == 'ncc4':
        x2search = torch.nn.functional.interpolate(xxx, scale_factor=1/4, mode='bicubic', align_corners=False)
        y2search = torch.nn.functional.interpolate(yyy, scale_factor=1/4, mode='bicubic', align_corners=False)
        D = ncc_dist(x2search, y2search, div_dim=True)

    # Consider each reconstruction for only one train-samples
    if use_bb:
        bb_mask = D.mul(-100000000).softmax(dim=0).mul(10).round().div(10).round()
        assert bb_mask.sum(dim=0).abs().sum() == D.shape[1]
        D[bb_mask != 1] = torch.inf

    dists, idxs = D.sort(dim=1, descending=False)
    return dists, idxs


@torch.no_grad()
def find_nearest_neighbour(X, x0, search='ncc', vote='mean', use_bb=True, nn_threshold=None, ret_idxs=False):
    xxx = X.clone()
    yyy = x0.clone()

    # Search Real --> Extracted
    if search == 'l2':
        D = l2_dist(yyy, xxx, div_dim=True)
    if search == 'ncc':
        D = ncc_dist(yyy, xxx, div_dim=True)
    elif search == 'ncc2':
        x2search = torch.nn.functional.interpolate(xxx, scale_factor=1 / 2, mode='bicubic', align_corners=False)
        y2search = torch.nn.functional.interpolate(yyy, scale_factor=1 / 2, mode='bicubic', align_corners=False)
        D = ncc_dist(y2search, x2search, div_dim=True)
    elif search == 'ncc4':
        x2search = torch.nn.functional.interpolate(xxx, scale_factor=1/4, mode='bicubic', align_corners=False)
        y2search = torch.nn.functional.interpolate(yyy, scale_factor=1/4, mode='bicubic', align_corners=False)
        D = ncc_dist(y2search, x2search, div_dim=True)
    elif search == 'dssim':
        D_ssim = get_ssim_all(yyy, xxx)
        D_dssim = (1 - D_ssim)/2
        D = D_dssim

    # Only consider Best-Bodies
    if use_bb:
        bb_mask = D.mul(-100000000).softmax(dim=0).mul(10).round().div(10).round()
        assert bb_mask.sum(dim=0).abs().sum() == D.shape[1]
        D[bb_mask != 1] = torch.inf

    dists, idxs = D.sort(dim=1, descending=False)

    # yy = yyy
    if vote == 'min' or vote is None:
        xx = xxx[idxs[:, 0]]
    else:
        # Ignore distant nearest-neighbours
        if nn_threshold is None:
            xs_idxs = idxs[:, :int(0.01*x0.shape[0])]
        else:
            xs_idxs = []
            for i in range(dists.shape[0]):
                x_idxs = [idxs[i, 0].item()]
                for j in range(1, dists.shape[1]):
                    if (dists[i, j] / dists[i, 0]) < nn_threshold:
                        x_idxs.append(idxs[i, j].item())
                    else:
                        break
                xs_idxs.append(x_idxs)

        # Voting
        xs = []
        for x_idxs in xs_idxs:
            if vote == 'min':
                x_voted = xxx[x_idxs[0]].unsqueeze(0)
            elif vote == 'mean':
                x_voted = xxx[x_idxs].mean(dim=0, keepdim=True)
            elif vote == 'median':
                x_voted = xxx[x_idxs].median(dim=0, keepdim=True).values
            elif vote == 'mode':
                x_voted = xxx[x_idxs].mode(dim=0, keepdim=True).values
            else:
                raise
            xs.append(x_voted)
        xx = torch.cat(xs, dim=0).clone()

    if ret_idxs:
        return xx, idxs[:, 0]

    return xx


@torch.no_grad()
def scale(xx, x0, ds_mean, xx_add_ds_mean=True):
    xx = xx.clone()
    x0 = x0.clone()
    ds_mean = ds_mean.clone()
    # Scale to images
    yy = x0 + ds_mean
    if xx_add_ds_mean:
        xx = transform_vmin_vmax_batch(xx + ds_mean)
    else:
        xx = transform_vmin_vmax_batch(xx)

    return xx, yy


@torch.no_grad()
def sort_by_metric(xx, yy, sort='ssim'):
    xx = xx.clone()
    yy = yy.clone()

    # Score
    psnr = lambda a, b: 20 * torch.log10(1.0 / (a - b).pow(2).reshape(a.shape[0], -1).mean(dim=1).sqrt())

    # Sort
    if sort == 'ssim':
        dists = get_ssim_pairs_kornia(xx, yy)
        dssim = (1 - dists) / 2
        _, sort_idxs = dists.sort(descending=True)
    elif sort == 'ncc':
        dists = (normalize_batch(xx) - normalize_batch(yy)).reshape(xx.shape[0], -1).norm(dim=1)
        _, sort_idxs = dists.sort()
    elif sort == 'l2':
        dists = (xx - yy).reshape(xx.shape[0], -1).norm(dim=1)
        _, sort_idxs = dists.sort()
    elif sort == 'psnr':
        dists = psnr(xx, yy)
        _, sort_idxs = dists.sort(descending=True)
    else:
        raise

    xx = xx[sort_idxs]
    yy = yy[sort_idxs]
    return xx, yy, dists, sort_idxs


@torch.no_grad()
def plot_table(xx, yy, fig_elms_in_line, fig_lines_per_page, fig_type='side_by_side',
               figpath=None, show=False, dpi=100, color_by_labels=None):
    # PRINT TABLES
    import matplotlib.pyplot as plt
    xx = xx.clone()
    yy = yy.clone()

    RED = torch.tensor([1, 0, 0])[None, :, None, None]
    BLUE = torch.tensor([0, 1, 0])[None, :, None, None]
    def add_colored_margin(x, labels, p=1):
        n, c, h, w = x.shape
        bg = torch.zeros(n, c, h + 2 * p, w + 2 * p)
        bg[labels == 0] += RED
        bg[labels == 1] += BLUE
        bg[:, :, p:-p, p:-p] = x
        return bg

    if color_by_labels is not None:
        yy = add_colored_margin(yy, color_by_labels, p=2)
        xx = add_colored_margin(xx, color_by_labels, p=2)

    if fig_type == 'side_by_side':
        qq = torch.stack(common_utils.common.flatten(list(zip(xx, yy))))
    elif fig_type == 'one_above_another':
        q_zip = common_utils.common.flatten(list(zip(torch.split(xx, fig_elms_in_line), torch.split(yy, fig_elms_in_line))))
        if len(q_zip) > 2:
            q_zip = q_zip[:-2]
            print('CUT the end of the zipped bla because it might have different shape before torch.cat')
        qq = torch.cat(q_zip)
    else:
        raise

    lines_num = qq.shape[0] // fig_elms_in_line
    print(qq.shape, lines_num)
    for page_num, line_num in enumerate(tqdm(range(0, lines_num, fig_lines_per_page))):
        s = line_num * fig_elms_in_line
        e = (line_num + fig_lines_per_page) * fig_elms_in_line
        print(page_num, s, e)
        grid = torchvision.utils.make_grid(qq[s:e], normalize=False, nrow=fig_elms_in_line, pad_value=1)
        if figpath is not None:
            plt.imsave(figpath, grid.permute(1, 2, 0).cpu().numpy(), dpi=dpi)
            print('Saved fig at:', figpath)
        if show:
            plt.figure(figsize=(80 * 2, 10 * 2))
            plt.axis('off')
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.show()
        plt.close('all')
        break
    print('DONE!')
