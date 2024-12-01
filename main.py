import wandb

import numpy as np
import torch
import os

import matplotlib.pyplot as plt

from bytecover.models.train_module import TrainModule
from bytecover.utils import initialize_logging, load_config
from bytecover.models.data_loader import bytecover_dataloader, ByteCoverDataset
from bytecover.models.modules import Bottleneck, Resnet50

config = load_config(config_path="config/config.yaml")
initialize_logging(config_path="config/logging_config.yaml", debug=False)
if config["wandb"]:
    wandb.init(
        # set the wandb project where this run will be logged
        project="ByteCover",
        # track hyperparameters and run metadata
        config=config["test"],
    )

data_split = "test"

dl = bytecover_dataloader(
                    data_path=config["data_path"],
                    file_ext=config["file_extension"],
                    dataset_path=config["dataset_path"],
                    data_split=data_split,
                    debug=False,
                    max_len=100,
                    **config["test"],)
ds: ByteCoverDataset = dl.dataset

model = Resnet50(
            Bottleneck,
            num_channels=config["num_channels"],
            num_classes=config["train"]["num_classes"],
            compress_ratio=config["train"]["compress_ratio"],
            tempo_factors=config["train"]["tempo_factors"],
        )

model.to(config["device"])


checkpoint = torch.load('models/orfium-bytecover.pt')
model.load_state_dict(checkpoint, strict=False)
model.eval()

albums = ('25', 'A Night At The Opera', 'Appetite For Destruction', 'Divide', 'Hozier', 
          'Ink Spots', 'Let It Be', 'Lumineers', 'Master of Puppets', 'Starboy', 'Teenage Dream', 'V')

singles = ('Adele - Hello', 'Queen - Bohemian Rhapsody (Remastered 2011)', 'Guns N\' Roses - Welcome To The Jungle',
           'Ed Sheeran - Shape of You', 'Hozier - Take Me To Church', 'The Ink Spots - I Don\'t Want To Set The World On Fire',
           'The Beatles - Let It Be (Remastered 2009)', 'The Lumineers - Ho Hey', 'Metallica - Master Of Puppets',
           'The Weeknd,Daft Punk - Starboy', 'Katy Perry - Teenage Dream', 'Maroon 5 - Sugar')

song_counter = 0
total_accuracy = 0

with torch.no_grad():
    basis_embeddings = []
    print('Basis generation started')
    for i in range(len(albums)):
        input = ds._read_audio(os.path.join(albums[i], singles[i])).to(config['device'])
        input = input[None,:].to(config['device'])
        result = model.forward(input)['f_c'].cpu()[0]
        basis_embeddings.append(result)
    print('Basis generation ended')
    # print(basis_embeddings)

    basis_embeddings = torch.stack(basis_embeddings).numpy()

    song_names = []
    print('Song list started')
    for i in range(len(albums)):
        base_dir = os.path.join(config['dataset_path'], albums[i])
        for (dirpath, dirnames, filenames) in os.walk(base_dir):
            song_names.append(filenames)
            break
    # print(song_names)
    print('Song list ended')

    print('Album detection started')

    accuracies = []
    for i in range(len(albums)):
        print('Album', albums[i], 'started')
        accuracy = 0
        for song in song_names[i]:
            input = ds._read_audio(os.path.join(albums[i], song[:-4])).to(config['device'])
            input = input[None,:].to(config['device'])
            result = model.forward(input)['f_c'].cpu()[0]

            # Target vector
            target = result.numpy()

            # Calculate Euclidean distances
            distances = np.linalg.norm(basis_embeddings - target, axis=1)
            # print(len(basis_embeddings), basis_embeddings)
            # print(len(target), target)

            # Find the index of the closest vector
            closest_index = np.argmin(distances)

            # Closest vector
            closest_vector = basis_embeddings[closest_index]

            if (closest_index == i):
                accuracy += 1
                total_accuracy += 1

            song_counter += 1

            print(song, '– closest album: ', albums[closest_index])
        accuracy = accuracy / len(song_names[i])

        accuracies.append(accuracy)
        print('Album', albums[i], 'finished, accuracy:', accuracy * 100, '%')
    
    
    print('Album detection ended')

    for i in range(len(albums)):
        print(albums[i], 'accuracy:', accuracies[i] * 100, '%')

    print()
    print('Total accuracy:', total_accuracy/song_counter * 100, '%')

x = albums
y = accuracies

# Plot the data
plt.plot(x, y, label='Accuracies per Album', color='blue', marker='o')

# Add labels and title
plt.xlabel('Album name')
plt.ylabel('Accuracy')
plt.title('Accuracies per Album')

# Add a legend
plt.legend()

# Show the plot
plt.show()






    
