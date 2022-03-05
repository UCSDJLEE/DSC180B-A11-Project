from torch_geometric.data import DataListLoader, Batch


def prepare_dataloader(datasets, batch_size):
    '''
    Intermediary function for formatting datasets to be compatible with 
    our model's input layer. Function uses torch_geometric.data.DataListLoader object,
    which batches data objects into Python list
    Function also assign custom collate function to help creating batch without
    repetition in doing su

    Parameters:
    datasets -- Graphdataset object of datasets to use for batch
    batch_size -- `batch_size` for DataListLoader hyperparam
    '''
    def collate(items): return Batch.from_data_list(sum(items, []))

    train_loader = DataListLoader(datasets, batch_size=batch_size, pin_memory=True, shuffle=True)
    train_loader.collate_fn = collate
    
    return train_loader 