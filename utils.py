import torch
import time
from torchvision import transforms

def train(model, test_img, train_loader, optimizer, criterion, epochs= 10, noise_level= 50):
    device= 'cpu'
    if torch.cuda.is_available():
        device= 'cuda'

    losses = []

    chosen_img = test_img
    totensor = transforms.PILToTensor()
    chosen_img = totensor(chosen_img).float().to(device).unsqueeze(0)

    noise = torch.FloatTensor(chosen_img.size()).normal_(mean=0, std=noise_level).to(device)

    chosen_img_noisy = chosen_img + noise

    outputs = []

    model.train()
    for epoch in range(epochs):
        avg_loss = 0
        start = time.time()
        for i, batch in enumerate(train_loader):

            img = batch.to(device)
            noise = torch.FloatTensor(img.size()).normal_(mean=0, std=noise_level).to(device)

            img_noisy = img + noise
            op = model(img_noisy)

            loss = criterion(op, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

        with torch.no_grad():
            op_btw = model(chosen_img_noisy)
            op_btw = op_btw.squeeze().cpu().numpy()
            outputs.append(op_btw)

        end = time.time()
        avg_loss /= len(train_loader)

        print(f'epoch {epoch}: avg loss= {avg_loss}, time taken= {(end - start)//60} min {(end - start)%60:.3f}sec')
        losses.append(avg_loss)

    chosen_img_noisy = chosen_img_noisy.squeeze().cpu().numpy()
    return losses, outputs, chosen_img_noisy
