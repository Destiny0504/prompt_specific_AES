import torch

def supervised_contrative_loss(pred, target, tau=0.1):

    batch_size = pred.size()[0]

    # Calculate the similarity.
    similarities = torch.nn.functional.cosine_similarity(
        pred.unsqueeze(1), pred.unsqueeze(0), dim=-1
    )
    # print(similarities)
    # Tau is the softmax temperature
    similarities = similarities / tau

    # Calculate the sum of exponential similarity
    similarities = torch.exp(similarities)
    # print(similarities)
    supervised_contrative_mask = (target.repeat(batch_size, 1) == target.unsqueeze(dim=1).repeat(1, batch_size))
    # print(supervised_contrative_mask)

    similarities_sum = similarities.sum(dim=-1).unsqueeze(dim=-1)

    # Calculate the loss of each index.
    similarities = similarities / similarities_sum

    # Only keep the similarity of positive instance.
    similarities = torch.masked_select(
        similarities, supervised_contrative_mask
    )

    # Calculate the Average loss
    loss = torch.mean(-torch.log(similarities))

    return loss

def SimCSE_loss(pred, positive_pair, tau=0.1):

    batch_size = pred.size()[0]

    # Calculate the similarity.
    similarities = torch.nn.functional.cosine_similarity(
        pred.unsqueeze(1), positive_pair.unsqueeze(0), dim=-1
    )

    # Tau is the softmax temperature
    similarities = similarities / tau

    # Calculate the sum of exponential similarity
    similarities = torch.exp(similarities)
    similarities_sum = similarities.sum(dim=-1).unsqueeze(dim=-1)

    # Calculate the loss of each index.
    similarities = similarities / similarities_sum

    # Only keep the similarity of positive instance.
    similarities = torch.masked_select(
        similarities, torch.eye(batch_size, device=pred.device) == 1
    )

    # Calculate the Average loss
    loss = torch.mean(-torch.log(similarities))

    return loss

def margin_loss(pred, target):
    loss_fn = torch.nn.L1Loss()
    pred = pred.unsqueeze(0)
    target = target.unsqueeze(0)
    return loss_fn((pred - pred.T), (target - target.T))

# def margin_loss(pred, target):
#     loss_fn = torch.nn.MSELoss(reduction='sum')
#     pred = pred.unsqueeze(0)
#     target = target.unsqueeze(0)
#     scalar = pred.size()[-1] * (pred.size()[-1] - 1)
#     return loss_fn((pred - pred.T), (target - target.T)) / scalar
