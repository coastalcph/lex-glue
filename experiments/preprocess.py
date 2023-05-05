    def preprocess_function(examples):
        if model_args.hierarchical:
            ehr_template = [[0] * data_args.max_seg_length]
            if config.model_type == 'bert':
                batch = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
                for ehr in examples['TEXT']:
                    encoded_text = tokenizer(ehr, padding=padding,max_length=data_args.max_seq_length, truncation=True)
                    # cut text to segments
                    segments = []
                    for i in range(0, data_args.max_segments * data_args.max_seg_length, data_args.max_seg_length):
                        segment = encoded_text[i:i + data_args.max_seg_length]
                        if not segment:
                            break
                        segments.append(segment)
                    decoded_segments = [tokenizer.decode(segment) for segment in segments]
                    ehr_encodings = tokenizer(decoded_segments[:data_args.max_segments], padding=padding,
                                               max_length=data_args.max_seg_length, truncation=True)                    
                    batch['input_ids'].append(ehr_encodings['input_ids'] + ehr_template * (
                                                    data_args.max_segments - len(ehr_encodings['input_ids'])))
                    batch['attention_mask'].append(ehr_encodings['attention_mask'] + ehr_template * (
                                                    data_args.max_segments - len(ehr_encodings['attention_mask'])))
                    batch['token_type_ids'].append(ehr_encodings['token_type_ids'] + ehr_template * (
                                                    data_args.max_segments - len(ehr_encodings['token_type_ids'])))                           
        
        elif config.model_type in ['longformer', 'big_bird']:
            ehrs = []
            max_position_embeddings = config.max_position_embeddings - 2 if config.model_type == 'longformer' \
                else config.max_position_embeddings
            for ehr in examples['TEXT']:
                ehrs.append(ehr)
            batch = tokenizer(ehrs, padding=padding, max_length=max_position_embeddings, truncation=True)
            if config.model_type == 'longformer':
                global_attention_mask = np.zeros((len(ehrs), max_position_embeddings), dtype=np.int32)
                # global attention on cls token
                global_attention_mask[:, 0] = 1
                batch['global_attention_mask'] = list(global_attention_mask)
        else:
            ehrs = []
            for ehr in examples['TEXT']:
                ehrs.append(ehr)
            batch = tokenizer(ehrs, padding=padding, max_length=512, truncation=True)            

        batch["labels"] = [[1 if label in labels else 0 for label in label_list] for labels in examples["labels"]]

        return batch