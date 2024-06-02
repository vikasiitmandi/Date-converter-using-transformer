import torch
import numpy as np
from faker import Faker  # Library to generate fake data
import random
from babel.dates import format_date  # Library to format dates
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm  # Library to display progress bars

# Different date formats to be used for generating human-readable dates
FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY', 
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']

# Function to transform human-readable dates to machine-readable dates
def transform(human_readable, machine_readable, human_vocab, machine_vocab):
    # Convert human-readable date to indices using human_vocab
    X = list(map(lambda x: human_vocab.get(x, '<unk>'), human_readable))
    # Convert machine-readable date to indices using machine_vocab
    Y = list(map(lambda x: machine_vocab.get(x, '<unk>'), machine_readable))

    '''
    Add padding tokens to the target sequence for transformer compatibility.
    We input [machine_vocab['<pad>']] + Y to the network.
    The output target is Y + [machine_vocab['<pad>']].
    The initial decoding uses only the first <pad> (controlled by a mask), so the first decoded character should be Y[0].
    The final decoding uses the entire [machine_vocab['<pad>']] + Y, so the target output is Y + [machine_vocab['<pad>']].
    '''
    Y = [machine_vocab['<pad>']] + Y + [machine_vocab['<pad>']]
    
    # Function to create one-hot vectors
    def zcs(length, idx):
        ret = np.zeros(length)
        ret[idx] = 1
        return ret

    # Convert input and output sequences to one-hot encoded format
    Xoh = np.array(list(map(partial(zcs, len(human_vocab)), X)), dtype=np.float32)
    Yoh = np.array(list(map(partial(zcs, len(machine_vocab)), Y)), dtype=np.float32)

    return Xoh, Yoh, {'human_readable': human_readable, 'machine_readable': machine_readable}

# Function to pad sequences to the same length
def collate_fn(pad_vec, batch):
    '''
    For shorter inputs, the one-hot vectors are fewer, and we need to pad them with the <pad> corresponding one-hot vector, i.e., pad_vec.
    '''
    pad_vec = pad_vec.reshape(-1, 1)
    batch_x, batch_y, extra = zip(*batch)
    max_len_x = max(x.shape[0] for x in batch_x)

    batch_x_paded = []
    for x in batch_x:
        pad_len = max_len_x - x.shape[0]
        pad = np.repeat(pad_vec, pad_len, axis=1).T
        batch_x_paded.append(np.vstack((x, pad)))

    batch_x = torch.FloatTensor(batch_x_paded)
    batch_y = torch.FloatTensor(batch_y)
    return batch_x, batch_y, extra

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, n_datas=10000, seed=12345):
        self.transform = transform

        self.fake = Faker()
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)

        self.human_vocab = set()
        self.machine_vocab = set()
        self.dataset = []
        for i in tqdm(range(n_datas)):
            human_readable, machine_readable = self.load_date()
            self.dataset.append((human_readable, machine_readable))
            self.human_vocab.update(tuple(human_readable))
            self.machine_vocab.update(tuple(machine_readable))

        self.human_vocab = dict(zip(sorted(self.human_vocab) + ['<unk>', '<pad>'], list(range(len(self.human_vocab) + 2))))
        
        '''
        When using a transformer, the target sequence needs to be passed to the decoder, but with a mask.
        The decoder uses the first 4 features when decoding the 4th output.
        Since softmax is used after adding the mask, there must be at least one valid element.
        The first decoded element uses a target sequence length of 0.
        Therefore, we must add a <pad> at the very beginning.
        '''
        self.inv_machine_vocab = dict(enumerate(sorted(self.machine_vocab) + ['<pad>']))
        self.machine_vocab = {v: k for k, v in self.inv_machine_vocab.items()}

    def __getitem__(self, idx):
        human_readable, machine_readable = self.dataset[idx] # dataset is a list of tuples
        return self.transform(human_readable, machine_readable, self.human_vocab, self.machine_vocab)

    def __len__(self):
        return len(self.dataset)

    def load_date(self):
        """
        Loads some fake dates.
        :returns: tuple containing human readable string, machine readable string, and date object
        """
        dt = self.fake.date_object()

        human_readable = format_date(dt, format=random.choice(FORMATS), locale='en_US')
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',', '')
        machine_readable = dt.isoformat()

        return human_readable, machine_readable

if __name__ == '__main__':
    dataset = Dataset(transform=transform, n_datas=1000)
    pad_vec = np.zeros(len(dataset.human_vocab))
    pad_vec[dataset.human_vocab['<pad>']] = 1
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=0,
                                             collate_fn=partial(collate_fn, pad_vec))

    for step, (batch_x, batch_y, extra) in enumerate(dataloader):
        print(step, batch_x.shape, batch_y.shape)
        print(batch_y[0], extra[0]['machine_readable']) # extra contains the human and machine readable dates
        if step >= 0:
            break
