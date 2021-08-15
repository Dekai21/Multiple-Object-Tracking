import sys
sys.path.append('..')
sys.path.append('.')

from torch.utils.data import Dataset
from tracker.data_track import MOT16Sequences
import itertools
import torch

from PIL import Image

class LongTrackTrainingDataset(Dataset):
    # TODO: More past frames
    def __init__(self, dataset, db, root_dir, vis_threshold=0.2, box_distort_perc = 0.05,
                 max_past_frames = 10):
        #self.sequences =  MOT16Sequences(dataset, root_dir, vis_threshold=vis_threshold)
        self.sequences =  MOT16Sequences(dataset, root_dir, vis_threshold=0)
        self.db=db
        for seq in self.sequences:
            seq.transforms = lambda x: x # Get rid of the transforms

        # Index the dataset (Determine all pairs of valid (sequence, current_frame) pairs
        self.seq_frames = [] 
        global_ids = []
        for seq_ix, seq in enumerate(self.sequences):
            
            # Collect the unique sequence Ids
            seq_ids = [frame_data['gt'].keys() for frame_data in seq.data]
            seq_ids = list(set(itertools.chain.from_iterable(seq_ids)))
            global_ids.extend([(str(seq), seq_id) for seq_id in seq_ids])

            valid_frames = range(1, len(seq))
            self.seq_frames.extend([(seq_ix, frame) for frame in valid_frames])

        # Assign a global id to each identity for every sequence
        self.global_ids = {seq_id_pair: global_id for global_id, seq_id_pair in enumerate(global_ids)}
        self.box_distort_perc = box_distort_perc
        self.max_past_frames = max_past_frames
        self.vis_threshold = vis_threshold

    def __len__(self):
        return len(self.seq_frames)
    
    def prepare_data(self, blob, seq):    
        #blob = curr_frame['gt']
        #ids = torch.as_tensor(list(blob['gt'].keys()))
        ids = blob['ids'].tolist()
        global_ids = torch.as_tensor([self.global_ids[(str(seq), id_)] for id_ in ids])

        vis = blob['vis'].cpu()
        keep = vis >= self.vis_threshold

        boxes = blob['boxes'].cpu()[keep]
        features = blob['reid'].cpu()[keep]
        vis = vis[keep]
        global_ids = global_ids[keep]

        boxes, keep = self.augment(boxes = boxes)

        return {'ids':global_ids[keep], 
                'vis':vis[keep], 
                'boxes': boxes[keep], 
                'features': features[keep]}
    
    def augment(self, boxes):
        """Wiggle boxes coordinates and randomly drop boxes"""

        keep = torch.rand(boxes.shape[0]) >0.1
        if not keep.any():
            keep = torch.ones_like(keep).bool()
        
        b_width = boxes[:, 2] - boxes[:, 0]
        b_height = boxes[:, 3] - boxes[:, 1]

        n_boxes = boxes.size(0)
        horizont_max = self.box_distort_perc*b_width
        vertical_max = self.box_distort_perc*b_height

        rand_uniform = lambda n: 2*torch.rand(n) -1

        aug_x0 = boxes[:, 0] + rand_uniform(n_boxes)*horizont_max
        aug_x1 = boxes[:, 2] +  rand_uniform(n_boxes)*horizont_max

        aug_y0 = boxes[:, 1] +  rand_uniform(n_boxes)*vertical_max
        aug_y1 = boxes[:, 3] + rand_uniform(n_boxes)*vertical_max

        aug_boxes = torch.stack((aug_x0, aug_y0, aug_x1, aug_y1)).T

        #return aug_boxes, ids, vis, embeddings
        return aug_boxes, keep

    def merge_frame_data(self, past_frame_data, first_past_frame):
        """Only keep the last appearance for every ID """
        final_data = {}
        for t_step, frame_data in enumerate(past_frame_data, first_past_frame):
            for idx, id_ in enumerate(frame_data['ids']):
                final_data[id_.item()] = {'time': t_step,
                                    **{key: frame_data[key][idx] for key in ('vis', 'boxes', 'features')}}

        all_ids = list(final_data.keys())

        return {'ids': torch.as_tensor(all_ids),
                'time': torch.as_tensor([final_data[id_]['time'] for id_ in all_ids]),
                'boxes': torch.as_tensor(torch.stack([final_data[id_]['boxes'] for id_ in all_ids])),
                'features': torch.stack([final_data[id_]['features'] for id_ in all_ids])}
        
    def __getitem__(self, idx):
        seq_ix, frame_ix = self.seq_frames[idx]
        seq = self.sequences[seq_ix]

        next_frame_ix = min(frame_ix+1, len(seq)-1)
        
        seq_db = self.db[str(seq)]

        curr_frame = seq_db[frame_ix]['gt']
        next_frame = seq_db[next_frame_ix]['gt']
        first_past_frame = max(frame_ix - self.max_past_frames - 1, 0)
        past_frames = [seq_db[frame_ix_]['gt'] for frame_ix_ in  range(first_past_frame, frame_ix)]

        curr_frame_data = self.prepare_data(curr_frame, seq)
        next_frame_data = self.prepare_data(next_frame, seq)
        past_frame_datum = [self.prepare_data(past_frame, seq) for past_frame in past_frames]

        past_frame_data = self.merge_frame_data(past_frame_datum, first_past_frame)
        curr_frame_data['time'] = frame_ix*torch.ones_like(curr_frame_data['ids'])
        next_frame_data['time'] = next_frame_ix*torch.ones_like(next_frame_data['ids'])

        return past_frame_data, curr_frame_data, next_frame_data