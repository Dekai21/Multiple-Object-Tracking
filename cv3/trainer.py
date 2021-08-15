import tqdm
import torch
from torch import nn
from torch.nn import functional as F

@torch.no_grad()
def compute_class_metric(pred, target, class_metrics = ('accuracy', 'recall', 'precision')):
    TP = ((target == 1) & (pred == 1)).sum().float()
    FP = ((target == 0) & (pred == 1)).sum().float()
    TN = ((target == 0) & (pred == 0)).sum().float()
    FN = ((target == 1) & (pred == 0)).sum().float()

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    recall = TP / (TP + FN) if TP + FN > 0 else torch.tensor(0)
    precision = TP / (TP + FP) if TP + FP > 0 else torch.tensor(0)

    class_metrics_dict =  {'accuracy': accuracy.item(), 'recall': recall.item(), 'precision': precision.item()}
    class_metrics_dict = {met_name: class_metrics_dict[met_name] for met_name in class_metrics}

    return class_metrics_dict



def train_one_epoch(model, data_loader, optimizer, accum_batches = 1, print_freq= 200):    
    model.train()
    device = next(model.parameters()).device
    metrics_accum = {'loss': 0., 'accuracy': 0., 'recall': 0., 'precision': 0.}
    #for i, batch in tqdm.tqdm(enumerate(data_loader)):
    for i, batch in tqdm.notebook.tqdm(enumerate(data_loader)):
        optimizer.zero_grad()

        # Since our model does not support automatic batching, we do manual
        # gradient accumulation        
        for sample in batch:
            past_frame, curr_frame, next_frame = sample
            track_feats, track_coords, track_ids = past_frame['features'].to(device), past_frame['boxes'].to(device), past_frame['ids'].to(device)
            current_feats, current_coords, curr_ids = curr_frame['features'].to(device), curr_frame['boxes'].to(device), curr_frame['ids'].to(device)
            next_feats, next_coords, next_ids = next_frame['features'].to(device), next_frame['boxes'].to(device), next_frame['ids'].to(device)
            track_t, curr_t, next_t = past_frame['time'].to(device), curr_frame['time'].to(device), next_frame['time'].to(device)

            assign_sim =model.forward(track_app = track_feats, 
                                                             current_app = current_feats.cuda(), 
                                                             next_app = next_feats.cuda(),
                                                             track_coords = track_coords.cuda(),
                                                             current_coords=current_coords.cuda(),
                                                             next_coords = next_coords.cuda(),
                                                             track_t = track_t,
                                                             curr_t = curr_t,
                                                             next_t = next_t
                                                             )

            same_id = (track_ids.view(-1, 1) == curr_ids.view(1, -1)).type(assign_sim.dtype)
            same_id = same_id.unsqueeze(0).expand(assign_sim.shape[0], -1, -1)

            loss = F.binary_cross_entropy_with_logits(assign_sim, same_id, pos_weight=torch.as_tensor(20.)) / float(len(batch))
            loss.backward()

            # Keep track of metrics
            with torch.no_grad():
                pred = (assign_sim[-1] > 0.5).view(-1).float()
                target = same_id[-1].view(-1)
                metrics = compute_class_metric(pred, target)

                for m_name, m_val in metrics.items():
                    metrics_accum[m_name] += m_val / float(len(batch))
                metrics_accum['loss'] += loss.item()

        if (i+1) %print_freq == 0 and i > 0:
            log_str = ". ".join([f"{m_name.capitalize()}: {m_val/ (print_freq if i !=0 else 1):.3f}" for m_name, m_val in metrics_accum.items()])
            print(f"Iter {i + 1}. " + log_str)
            metrics_accum = {'loss': 0., 'accuracy': 0., 'recall': 0., 'precision': 0.}
    
        optimizer.step()
    model.eval()



