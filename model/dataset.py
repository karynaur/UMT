import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
import os

def crop_seg(feature, start, end, duration):
    S, D = feature.shape
    start_quantile = start / duration
    end_quantile = end / duration
    start_idx = int(S * start_quantile)
    end_idx = int(S * end_quantile)
    if start_idx == end_idx:
        if start_idx == S:
            start_idx -= 1
        else:
            end_idx += 1
    feature = feature[start_idx:end_idx, :]

    if len(feature) == 0:
        return None
    else:
        return feature


def fill_missing_features(method, feature_size):
    if method == 'random':
        return torch.rand(1, feature_size)
    elif method == 'zero':
        return torch.zeros(1, feature_size).float()


def load_features(path, features, vid_id, start, end, duration, pad_idx):
   
  stacks = {}

  stack_rgb = torch.fromnumpy(np.load(os.path.join(path, f'{video_id}_rgb.npy'))).float()
  stack_flow = torch.fromnumpy(np.load(os.path.join(path, f'{video_id}_flow.npy'))).float()
  
  stack_rgb = crop_seg(stack_rgb, start, end, duration)
  stack_flow = crop_seg(stack_flow, start, end, duration)
  
  stacks['rgb'] = stack_rgb
  stacks['flow'] = stack_flow
  
  return stacks



def caption_iterator(cfg, batch_size, phase):
    print(f'Contructing caption_iterator for "{phase}" phase')
    spacy_en = spacy.load('en')
    
    def tokenize_en(txt):
        return [token.text for token in spacy_en.tokenizer(txt)]
    
    CAPTION = data.ReversibleField(
        tokenize='spacy', init_token=cfg.start_token, eos_token=cfg.end_token, 
        pad_token=cfg.pad_token, lower=True, batch_first=True, is_target=True
    )
    INDEX = data.Field(
        sequential=False, use_vocab=False, batch_first=True
    )
    
    # the order has to be the same as in the table
    fields = [
        ('video_id', None),
        ('caption', CAPTION),
        ('start', None),
        ('end', None),
        ('duration', None),
        ('phase', None),
        ('idx', INDEX),
    ]

    dataset = data.TabularDataset(
        path=cfg.train_meta_path, format='tsv', skip_header=True, fields=fields,
    )
    CAPTION.build_vocab(dataset.caption, min_freq=cfg.min_freq_caps, vectors=cfg.word_emb_caps)
    train_vocab = CAPTION.vocab
    
    if phase == 'val_1':
        dataset = data.TabularDataset(path=cfg.val_1_meta_path, format='tsv', skip_header=True, fields=fields)
    elif phase == 'val_2':
        dataset = data.TabularDataset(path=cfg.val_2_meta_path, format='tsv', skip_header=True, fields=fields)

    # sort_key = lambda x: data.interleave_keys(len(x.caption), len(y.caption))
    datasetloader = data.BucketIterator(dataset, batch_size, sort_key=lambda x: 0, 
                                        device=torch.device(cfg.device), repeat=False, shuffle=True)
    return train_vocab, datasetloader



class I3DFeaturesDataset(Dataset):
    
    def __init__(self, features_path, feature_name, meta_path, device, pad_idx, cfg):
        self.cfg = cfg
        self.features_path = features_path
        self.feature_name = f'{feature_name}_features'
        self.feature_names_list = [self.feature_name]
        self.device = device
        self.dataset = pd.read_csv(meta_path, sep='\t')
        self.pad_idx = pad_idx
        self.feature_size = 1024
    
    def __getitem__(self, indices):
        video_ids, captions, starts, ends, vid_stacks_rgb, vid_stacks_flow = [], [], [], [], [], []

        for idx in indices:
            idx = idx.item()
            video_id, caption, start, end, duration, _, _ = self.dataset.iloc[idx]
            
            stack = load_features_from_npy(
                self.cfg, self.feature_names_list, video_id, start, end, duration, 
                self.pad_idx, self.get_full_feat
            )

            vid_stack_rgb, vid_stack_flow = stack['rgb'], stack['flow']
            
            both_are_None = vid_stack_rgb is None and vid_stack_flow is None
            none_is_None = vid_stack_rgb is not None and vid_stack_flow is not None
            assert both_are_None or none_is_None
            
            # # sometimes stack is empty after the filtering. we replace it with noise
            if both_are_None:
                # print(f'RGB and FLOW are None. Zero (1, D) @: {video_id}')
                vid_stack_rgb = fill_missing_features('zero', self.feature_size)
                vid_stack_flow = fill_missing_features('zero', self.feature_size)
    
            # append info for this index to the lists
            video_ids.append(video_id)
            captions.append(caption)
            starts.append(start)
            ends.append(end)
            vid_stacks_rgb.append(vid_stack_rgb)
            vid_stacks_flow.append(vid_stack_flow)
            
        vid_stacks_rgb = pad_sequence(vid_stacks_rgb, batch_first=True, padding_value=self.pad_idx)
        vid_stacks_flow = pad_sequence(vid_stacks_flow, batch_first=True, padding_value=0)
                
        starts = torch.tensor(starts).unsqueeze(1)
        ends = torch.tensor(ends).unsqueeze(1)

        batch_dict = {
            'video_ids': video_ids,
            'captions': captions,
            'starts': starts.to(self.device),
            'ends': ends.to(self.device),
            'feature_stacks': {
                'rgb': vid_stacks_rgb.to(self.device),
                'flow': vid_stacks_flow.to(self.device),
            }
        }
        
        return batch_dict
    
    def __len__(self):
      return len(self.dataset)
       

class ActivityNetCaptionsDataset(Dataset):
    
    def __init__(self, cfg, phase):

        self.cfg = cfg
        self.phase = phase

        self.feature_names = f'{cfg.video_feature_name}'
        
        if phase == 'train':
            self.meta_path = cfg.train_meta_path
            self.batch_size = cfg.train_batch_size
        elif phase == 'val_1':
            self.meta_path = cfg.val_1_meta_path
            self.batch_size = cfg.inference_batch_size
        elif phase == 'val_2':
            self.meta_path = cfg.val_2_meta_path
            self.batch_size = cfg.inference_batch_size
        else:
            raise NotImplementedError

        # caption dataset *iterator*
        self.train_vocab, self.caption_loader = caption_iterator(cfg, self.batch_size, self.phase)
        
        self.trg_voc_size = len(self.train_vocab)
        self.pad_idx = self.train_vocab.stoi[cfg.pad_token]
        self.start_idx = self.train_vocab.stoi[cfg.start_token]
        self.end_idx = self.train_vocab.stoi[cfg.end_token]
        self.features_dataset = I3DFeaturesDataset(
            cfg.video_features_path, cfg.video_feature_name, self.meta_path, 
            torch.device(cfg.device), self.pad_idx, self.get_full_feat, cfg
        )
            
        # initialize the caption loader iterator
        self.caption_loader_iter = iter(self.caption_loader)
        
    def __getitem__(self, dataset_index):
        caption_data = next(self.caption_loader_iter)
        to_return = self.features_dataset[caption_data.idx]
        to_return['caption_data'] = caption_data

        return to_return

    def __len__(self):
        return len(self.caption_loader)
    
    def update_iterator(self):
        '''This should be called after every epoch'''
        self.caption_loader_iter = iter(self.caption_loader)
        
    def dont_collate(self, batch):
        return batch[0]
