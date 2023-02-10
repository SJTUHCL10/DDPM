import torch
from tqdm.auto import tqdm


def train(diffusion_model, dataloader, optimizer, scheduler=None, num_epoch=1, verbose=False, print_steps=100):
    progress_bar = tqdm(range(num_epoch*len(dataloader)))
    device = next(diffusion_model.parameters()).device
    for epoch in range(num_epoch):
        training_loss = 0
        sample_cnt = 0
        diffusion_model.train()
        for step, (images, labels) in enumerate(dataloader):
            x_0 = images.to(device)
            loss = diffusion_model(x_0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            progress_bar.update(1)

            bs = x_0.shape[0]
            sample_cnt += bs
            training_loss += loss.detach().cpu() * bs

            if verbose and step % print_steps == print_steps-1:
                # print training loss
                training_loss /= sample_cnt
                sample_cnt = 0
                print('step:', step+1, ' training loss={:.4f}'.format(training_loss))


def evaluate(diffusion_model, dataloader, verbose=True):
    progress_bar = tqdm(range(len(dataloader)))
    device = next(diffusion_model.parameters()).device
    diffusion_model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            x_0 = images.to(device)
            loss = diffusion_model(x_0)
            bs = images.shape[0]
            test_loss += bs*loss
            progress_bar.update(1)

        test_loss /= len(dataloader.dataset)
        if verbose:
            print('eval loss={:.4f}'.format(test_loss))

    return test_loss
