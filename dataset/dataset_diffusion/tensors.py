import torch

def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

def n_joints_to_mask(n_joints, max_joints):
    mask = torch.arange(max_joints + 1, device=n_joints.device).expand(len(n_joints), max_joints + 1) < (n_joints.unsqueeze(1) + 1)
    mask = mask.unsqueeze(2).float() * mask.unsqueeze(1).float() 
    return mask

def length_to_temp_mask(max_len_mask, lengths, max_len):
    mask = torch.arange(max_len + 1, device=lengths.device).expand(len(lengths), max_len + 1) < (lengths.unsqueeze(1) + 1)
    mask = mask.unsqueeze(2).float() * mask.unsqueeze(1).float() 
    mask = mask.logical_and(max_len_mask)
    return mask

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def create_padded_relation(relation_np, max_joints, n_joints):
        # it counts on spatial attention masks!
        relation = torch.from_numpy(relation_np)
        padded_relation = torch.zeros((max_joints, max_joints)) 
        padded_relation[:n_joints, :n_joints ] = relation
        return padded_relation

def truebones_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    tposfirstframebatch = [b['tpos_first_frame'] for b in notnone_batches]
    meanbatch = [b['mean'] for b in notnone_batches]
    stdbatch = [b['std'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]
    if 'n_joints' in notnone_batches[0]:
        jointsnumbatch = [b['n_joints'] for b in notnone_batches]
    else:
        jointsnumbatch = [22 for b in notnone_batches] #smpl n_joints 
        
    if 'temporal_mask' in notnone_batches[0]:
        temporalmasksbatch = [b['temporal_mask'] for b in notnone_batches]
    if 'crop_start_ind' in notnone_batches[0]:
        cropstartindbatch = [b['crop_start_ind'] for b in notnone_batches]
        
    
    
    databatchTensor = collate_tensors(databatch)
    tposfirstframebatchTensor = collate_tensors(tposfirstframebatch)
    meanbatchTensor = collate_tensors(meanbatch)
    stdbatchTensor = collate_tensors(stdbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    cropstartindTensor = torch.as_tensor(cropstartindbatch)
    lengthsmaskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
    jointsnumbatchTensor = torch.as_tensor(jointsnumbatch)
    jointsmaskbatchTensor = n_joints_to_mask(jointsnumbatchTensor, databatchTensor.shape[1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    collated_temporalmasksbatch = collate_tensors(temporalmasksbatch)
    maskbatchTensor = length_to_temp_mask(collated_temporalmasksbatch, lenbatchTensor, collated_temporalmasksbatch[0].size(0) - 1).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor, 'lengths_mask': lengthsmaskbatchTensor, 'tpos_first_frame': tposfirstframebatchTensor, 'crop_start_ind': cropstartindTensor, 'mean': meanbatchTensor, 'std':stdbatchTensor}}
    
    if 'object_type' in notnone_batches[0]:
        objecttypebatch = [b['object_type'] for b in notnone_batches]
        cond['y'].update({'object_type': objecttypebatch})
    
    if 'parents' in notnone_batches[0]:
        parentsbatch = [b['parents'] for b in notnone_batches]
        cond['y'].update({'parents': parentsbatch})
          
    if 'joints_names_embs' in notnone_batches[0]:
        jointsnamesembsbatch = [b['joints_names_embs'] for b in notnone_batches]
        jointsnamesembsbatchTensor = collate_tensors(jointsnamesembsbatch)
        cond['y'].update({'joints_names_embs': jointsnamesembsbatchTensor})
    
    if 'text_embed' in notnone_batches[0]:
        textembedbatch = [b['text_embed'] for b in notnone_batches]
        cond['y'].update({'text_embed': torch.stack(textembedbatch)})

    if 'real_text' in notnone_batches[0]:
        realtextbatch = [b['real_text'] for b in notnone_batches]
        cond['y'].update({'real_text': realtextbatch})
        
    if 'joints_relations' in notnone_batches[0]:
        jointsrelationsbatch = [b['joints_relations'] for b in notnone_batches]

    if 'graph_dist' in notnone_batches[0]:
        graphdistbatch = [b['graph_dist'] for b in notnone_batches]

    cond['y'].update({'joints_mask': jointsmaskbatchTensor})
    cond['y'].update({'n_joints': jointsnumbatchTensor})
    cond['y'].update({'joints_relations': torch.stack(jointsrelationsbatch)})
    cond['y'].update({'graph_dist': torch.stack(graphdistbatch)})

    return motion, cond

""" recieves list of tuples of the form: 
 motion, m_length, parents, tpos_first_frame, offsets, self.temporal_mask_template, joints_graph_dist, joints_relations, object_type, joints_names_embs, ind, mean, std, max_joints, text_embed, real_text
"""
def truebones_batch_collate(batch):
    max_joints = batch[0][13]
    adapted_batch = []
    for b in batch:  
        max_len, n_joints, n_feats = b[0].shape
        tpos_first_frame = torch.zeros((max_joints, n_feats))
        tpos_first_frame[:n_joints] = torch.tensor(b[3])
        motion = torch.zeros((max_len, max_joints, n_feats)) # (frames, max_joints, feature_len) 
        motion[:, :b[0].shape[1], :] = torch.tensor(b[0])   
        joints_names_embs = torch.zeros((max_joints, b[9].shape[1]))
        joints_names_embs[:n_joints] = torch.tensor(b[9])
        crop_start_ind = b[10]
        mean = torch.zeros((max_joints, n_feats))
        mean[:n_joints] = torch.tensor(b[11])
        std = torch.ones((max_joints, n_feats))
        std[:n_joints] = torch.tensor(b[12])
        n_joints = b[0].shape[1]
        temporal_mask = b[5][:max_len + 1, :max_len + 1].clone()
        padded_joints_relations =  create_padded_relation(b[7], max_joints, n_joints)
        padded_graph_dist =  create_padded_relation(b[6], max_joints, n_joints)
        object_type = b[8]
        text_embed = b[14] if len(b) > 14 else None
        real_text = b[15] if len(b) > 15 else None

        item = {
            'inp': motion.permute(1, 2, 0).float(), # [seqlen , J, 13] -> [J, 13,  seqlen]
            'n_joints': n_joints,
            'lengths': b[1],
            'parents': b[2],
            'temporal_mask' : temporal_mask,
            'graph_dist' : padded_graph_dist,
            'joints_relations':  padded_joints_relations,
            'object_type': object_type,
            'joints_names_embs': joints_names_embs,
            'tpos_first_frame': tpos_first_frame, 
            'crop_start_ind': crop_start_ind,
            'mean': mean,
            'std': std,
            'text_embed': text_embed,
            'real_text': real_text
        } 
        adapted_batch.append(item)

    return truebones_collate(adapted_batch)