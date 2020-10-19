import torch
import torch.nn.functional as F

import pdb

def flag_biased(model_forward, perturb_shape, y, args, optimizer, device, criterion, training_idx) :
    unlabel_idx = list(set(range(perturb_shape[0])) - set(training_idx))

    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-args.step_size, args.step_size).to(device)
    perturb.data[unlabel_idx] *= args.amp
    perturb.requires_grad_()
    out = forward(perturb)
    loss = criterion(out, y)
    loss /= args.m

    for _ in range(args.m-1):
        loss.backward()

        perturb_data_training = perturb[training_idx].detach() + args.step_size * torch.sign(perturb.grad[training_idx].detach())
        perturb.data[training_idx] = perturb_data_training.data

        perturb_data_unlabel = perturb[unlabel_idx].detach() + args.amp*args.step_size * torch.sign(perturb.grad[unlabel_idx].detach())
        perturb.data[unlabel_idx] = perturb_data_unlabel.data

        perturb.grad[:] = 0
        out = forward(perturb)
        loss = criterion(out, y)
        loss /= args.m

    loss.backward()
    optimizer.step()

    return loss, out

def flag(model_forward, perturb_shape, y, args, optimizer, device, criterion) :
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-args.step_size, args.step_size).to(device)
    perturb.requires_grad_()
    out = forward(perturb)
    loss = criterion(out, y)
    loss /= args.m

    for _ in range(args.m-1):
        loss.backward()
        perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out = forward(perturb)
        loss = criterion(out, y)
        loss /= args.m

    loss.backward()
    optimizer.step()

    return loss, out


def flag_products(model, clean, y, adjs, args, optimizer, device, criterion, train_idx=None) :
    model.train()
    if train_idx is not None:
        model_forward = lambda x: model(x, adjs)[train_idx]
    else:
        model_forward = lambda x: model(x, adjs)
    optimizer.zero_grad()

    perturb_t = torch.FloatTensor(*clean[:args.batch_size].shape).uniform_(-args.step_size, args.step_size).to(device)
    perturb_un = torch.FloatTensor(*clean[args.batch_size:].shape).uniform_(-args.amp*args.step_size, args.amp*args.step_size).to(device)
    perturb_t.requires_grad_()
    perturb_un.requires_grad_()


    perturb = torch.cat((perturb_t, perturb_un), dim=0)
    out = model_forward(clean + perturb)
    loss = criterion(out, y)
    loss /= args.m


    for _ in range(args.m-1):
        loss.backward()

        perturb_data_t = perturb_t.detach() + args.step_size * torch.sign(perturb_t.grad.detach())
        perturb_t.data = perturb_data_t.data
        perturb_t.grad[:] = 0

        perturb_data_un = perturb_un.detach() + args.amp*args.step_size * torch.sign(perturb_un.grad.detach())
        perturb_un.data = perturb_data_un.data
        perturb_un.grad[:] = 0

        perturb = torch.cat((perturb_t, perturb_un), dim=0)
        out = model_forward(clean + perturb)
        loss = criterion(out, y)
        loss /= args.m

    loss.backward()
    optimizer.step()

    return loss, out
