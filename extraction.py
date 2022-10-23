import torch
import torchvision

import wandb
from common_utils.common import now
from CreateModel import Flatten
from evaluations import get_evaluation_score_dssim, viz_nns


def l2_dist(x, y):
    """x, y should be of shape [batch, D]"""
    xx = x.pow(2).sum(1).view(-1, 1)
    yy = y.pow(2).sum(1).view(1, -1)
    xy = torch.einsum('id,jd->ij', x, y)
    dists = xx + yy - 2 * xy
    return dists


def diversity_loss(x, min_dist):
    flat_x = Flatten()(x)
    D = l2_dist(flat_x, flat_x)
    D.fill_diagonal_(torch.inf)
    nn_dist = D.min(dim=1).values
    relevant_nns = nn_dist[nn_dist < min_dist]
    if relevant_nns.shape[0] > 0:
        return relevant_nns.mul(-20).sigmoid().mean()
    else:
        return torch.tensor(0)


# def send_input_data(args, model, x0, y0):
#     if not args.wandb_active: return
#     _, c, h, w = x0.shape
#     n_weights = model.layers[0].weight.shape[0]
#     w = model.layers[0].weight.reshape(n_weights, c, h, w)
#     w_nns, _ = viz_nns(w.data, x0, max_per_nn=2)
#     w_viz = torchvision.utils.make_grid(w_nns[:100], normalize=False, nrow=20)
#     wandb.log({
#         "weights_of_first_layer": wandb.Image(w_viz),
#     })


def get_trainable_params(args, x0):
    n, c, h, w = x0.shape
    x = torch.randn(args.extraction_data_amount, c, h, w).to(args.device) * args.extraction_init_scale
    x.requires_grad_(True)
    l = torch.rand(args.extraction_data_amount, 1).to(args.device)
    l.requires_grad_(True)
    opt_x = torch.optim.SGD([x], lr=args.extraction_lr, momentum=0.9)
    opt_l = torch.optim.SGD([l], lr=args.extraction_lambda_lr, momentum=0.9)
    return l, opt_l, opt_x, x


def get_kkt_loss(args, values, l, y, model):
    l = l.squeeze()
    # all three shape should be (n)
    assert values.dim() == 1
    assert l.dim() == 1
    assert y.dim() == 1
    assert values.shape == l.shape == y.shape

    output = values * l * y
    grad = torch.autograd.grad(
        outputs=output,
        inputs=model.parameters(),
        grad_outputs=torch.ones_like(output, requires_grad=False, device=output.device).div(args.extraction_data_amount),
        create_graph=True,
        retain_graph=True,
    )
    kkt_loss = 0

    for i, (p, grad) in enumerate(zip(model.parameters(), grad)):
        assert p.shape == grad.shape
        l = (p.detach().data - grad).pow(2).sum()
        kkt_loss += l
    return kkt_loss


def get_verify_loss(args, x, l):
    loss_verify = 0
    loss_verify += 1 * (x - 1).relu().pow(2).sum()
    loss_verify += 1 * (-1 - x).relu().pow(2).sum()
    loss_verify += 5 * (-l + args.extraction_min_lambda).relu().pow(2).sum()

    return loss_verify


def calc_extraction_loss(args, l, model, values, x, y):
    kkt_loss, loss_verify = torch.tensor(0), torch.tensor(0)
    if args.extraction_loss_type == 'kkt':
        kkt_loss = get_kkt_loss(args, values, l, y, model)
        loss_verify = get_verify_loss(args, x, l)
        loss = kkt_loss + loss_verify

    elif args.extraction_loss_type == 'naive':
        loss_naive = -(values[y == 1].mean() - values[y == -1].mean())
        loss_verify = loss_verify.to(args.device).to(torch.get_default_dtype())
        loss_verify += (x - 1).relu().pow(2).sum()
        loss_verify += (-1 - x).relu().pow(2).sum()

        loss = loss_naive + loss_verify
    else:
        raise ValueError(f'unknown args.extraction_loss_type={args.extraction_loss_type}')

    return loss, kkt_loss, loss_verify


def evaluate_extraction(args, epoch, loss_extract, loss_verify, x, x0, y0, ds_mean):
    x_grad = x.grad.clone().data
    x = x.clone().data
    if args.wandb_active:
        wandb.log({
            "extraction epoch": epoch,
            "loss extract": loss_extract,
            "loss verify": loss_verify,
        })

    xx = x.data.clone()
    yy = x0.clone()
    metric = 'ncc'
    if args.dataset == 'mnist':
        metric = 'l2'

    qq, _ = viz_nns(xx, yy, max_per_nn=4, metric=metric)
    extraction_grid = torchvision.utils.make_grid(qq[:100], normalize=False, nrow=10)
    _, v = viz_nns(xx, yy, max_per_nn=1, metric=metric)
    extraction_score = v[:10].mean().item()

    xx += ds_mean
    yy += ds_mean
    qq, _ = viz_nns(xx, yy, max_per_nn=4, metric=metric)
    extraction_grid_with_mean = torchvision.utils.make_grid(qq[:100], normalize=False, nrow=10)
    _, v = viz_nns(xx, yy, max_per_nn=1, metric=metric)
    extraction_score_with_mean = v[:10].mean().item()

    # SSIM EVALUATION
    xx = x.data.clone()
    yy = x0.clone()
    dssim_score, dssim_grid = get_evaluation_score_dssim(xx, yy, ds_mean, vote=None, show=False)

    if args.wandb_active:
        wandb.log({
            "extraction": wandb.Image(extraction_grid),
            "extraction score": extraction_score,
            "extraction with mean": wandb.Image(extraction_grid_with_mean),
            "extraction score with mean": extraction_score_with_mean,
            "dssim score": dssim_score,
            "extraction dssim": wandb.Image(dssim_grid),
        })

    print(f'{now()} T={epoch} ; Losses: extract={loss_extract.item():5.10g} verify={loss_verify.item():5.5g} grads={x_grad.abs().mean()} Extraction-Score={extraction_score} Extraction-DSSIM={dssim_score}')

    return extraction_score
