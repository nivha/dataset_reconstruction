import torch
import numpy as np


def create_2d_data(args):
    n = args.data_amount
    if args.data_2d_shape == 'spiral':
        angle = np.linspace(0, 12 * 2 * np.pi, n)
        radius = np.linspace(.5, 1., n)

        x_coord = radius * np.cos(angle)
        y_coord = radius * np.sin(angle)

        x0 = torch.tensor(list(zip(x_coord, y_coord)))
        y0 = torch.zeros_like(torch.tensor(x_coord))

        y0[y0.shape[0] // 2:] = 1
        r = torch.randperm(y0.shape[0])
        y0 = y0.view(-1)[r].view(y0.size())

    if args.data_2d_shape == 'grid':
        # generate grid (x)
        x_coord = torch.linspace(0, 1, n).mul(2).add(-1)
        y_coord = torch.linspace(0, 1, n).mul(2).add(-1)
        x0 = torch.stack(torch.meshgrid([x_coord, y_coord], indexing=None)).reshape(2, -1).t()

        # generate labels (y)
        y0 = torch.zeros(x0.shape[0])
        if args.data_labels == 'chess':
            if n % 2 == 1:
                for i in range(len(y0)):
                    if i % 2 == 0:
                        y0[i] = 1
            if n % 2 == 0:
                parity = 0
                for i in range(len(y0)):
                    if i % n == 0:
                        parity = abs(parity - 1)
                    if i % 2 == parity:
                        y0[i] = 1
        elif args.data_labels == 'random':
            y0[y0.shape[0] // 2:] = 1
            r = torch.randperm(y0.shape[0])
            y0 = y0.view(-1)[r].view(y0.size())
        elif args.data_labels == 'linear':
            y0[len(y0) // 2:] = 1
        else:
            raise NotImplemented(f'args.data_labels={args.data_labels} not implemented')

    return [(x0, y0)]

