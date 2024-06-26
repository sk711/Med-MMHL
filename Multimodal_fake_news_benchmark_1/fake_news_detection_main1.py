
#21mARCH 7:21
#==============================
import os
import argparse
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import git
import train
from fake_article_dataset import PD_Dataset, PS_Dataset
import MyFunc
from eval import eval
from baseline_models import bert_classifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

parser = argparse.ArgumentParser(description='fk_det_model text classifier')

# data
parser.add_argument('-root-path', type=str, default='./', help='the data directory')
parser.add_argument('-dataset-type', type=str, default='fakenews_article', help='choose dataset to run [options: fakenews_article, sentence, fakenews_tweet]')
parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
parser.add_argument('-split-ratio', type=str, default='[0.7, 0.8, 1.0]', help='the split ratio of tr, dev, te sets')
parser.add_argument('-benchmark-path', type=str, default='./benchmarked_data/', help='the benchmark data directory')

# learning
parser.add_argument('-lr', type=float, default=0.00001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 32]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=1, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=1, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=15, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
parser.add_argument('-small_data', action='store_true', default=False, help='whether use small dataset for fast debug')
parser.add_argument('-frequency', type=int, default=1, help='frequency of saving the model')

# model
parser.add_argument('-freeze-bert', action='store_true', default=True, help='freeze bert parameters')
parser.add_argument('-bert-type', type=str, default='bert-base-cased', help='the bert embedding choice')
parser.add_argument('-model-type', type=int, default=1, help='different structures of metric model, see document for details')
parser.add_argument('-model-name', type=str, default='bert_classifier', help='different structures of metric model, see document for details')
parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.3]')

# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

# evaluation
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-test', action='store_true', default=False, help='train or test')

# Add a new argument to specify the directory for saving checkpoints
parser.add_argument('-checkpoint-dir', type=str, default='Med-MMHL/checkpoints/', help='directory to save checkpoints')
# Add a new argument to specify the directory for saving snapshots
parser.add_argument('-snapshot-dir', type=str, default='Med-MMHL/snapshot/', help='directory to save snapshots in the Git repository')

args = parser.parse_args()

# update args and print
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# Define Git repository and checkpoint directory paths
repo_url = "https://github.com/sk711/Med-MMHL.git"
repo_folder = "Med-MMHL"
# Clone the Git repository if it doesn't exist
if not os.path.exists(repo_folder):
    repo = git.Repo.clone_from(repo_url, repo_folder)
else:
    repo = git.Repo(repo_folder)

# Define the checkpoint directory path inside the cloned repository
checkpoint_dir = os.path.join(repo.working_tree_dir, "checkpoints")

# Ensure the checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

# Check if the directory for saving snapshots exists, if not, create it
if not os.path.exists(args.snapshot_dir):
    os.makedirs(args.snapshot_dir)

benchmark_dt_path = args.benchmark_path + args.dataset_type
tr, dev, te = MyFunc.read_benchmark_set(benchmark_dt_path)

if 'sentence' in args.dataset_type:
    tr_dataset = PS_Dataset(args, tr)
    dev_dataset = PS_Dataset(args, dev)
    te_dataset = PS_Dataset(args, te)
else:
    tr_dataset = PD_Dataset(args, tr)
    dev_dataset = PD_Dataset(args, dev)
    te_dataset = PD_Dataset(args, te)

tr_dataloader = torch.utils.data.DataLoader(
    tr_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True
)
dev_dataloader = torch.utils.data.DataLoader(
    dev_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=True
)
te_dataloader = torch.utils.data.DataLoader(
    te_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=True
)

# model
if args.model_name == 'bert_classifier':
    fk_det_model = bert_classifier.BertClassifier
if args.model_name == 'bert_classifier':
    fk_det_model = bert_classifier.BertClassifier(args)
else:
    raise ValueError('the model_name is set wrongly')

# Load snapshot if available
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    fk_det_model.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    fk_det_model = fk_det_model.cuda()

# Training loop
if not args.test:
    print('Training mode')
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_loss = float('inf')
    best_epoch = 0

try:
    print('Start training')
    print('Inside try') 
    
    # Initialize start_epoch
    start_epoch = 1
    print('start_epoch value: ',start_epoch)
    
    # Check if there are any existing checkpoints in the save directory
    existing_checkpoints = [f for f in os.listdir(args.checkpoint_dir) if f.startswith('model_epoch_')]
    if existing_checkpoints:
        print('Inside existing checkpoints if condition') 
        # If existing checkpoints are found, load the latest checkpoint
        latest_checkpoint = max(existing_checkpoints)
        start_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0]) + 1
        print(f"Resuming training from epoch {start_epoch} using checkpoint {latest_checkpoint}")

        # Load the state dictionary from the latest checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        fk_det_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No existing checkpoints found. Starting training from the first epoch.")
    
    # Continue training from start_epoch
    for epoch in range(start_epoch, args.epochs + 1):
        print('Inside for loop line 167') 
        train_loss, train_accuracy = train.train(tr_dataloader, dev_dataloader, fk_det_model, args)

        val_loss, val_accuracy = eval(dev_dataloader, fk_det_model, args)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            if args.save_best:
                torch.save(fk_det_model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))

        if epoch % args.frequency == 0:
            print('inside if condition of saving model snapshot')
            # Save the model snapshot
            snapshot_path = os.path.join(args.snapshot_dir, f'model_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': fk_det_model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy
            }, snapshot_path)
            print(f'Model snapshot saved at epoch {epoch}')
            
            # Commit changes to Git
            repo.git.add('--all')
            repo.index.commit('Added snapshots')
            print("Snapshot saved.")                             

except KeyboardInterrupt:
    print('\n' + '-' * 89)
    print('Exiting from training early')



    print('Start testing')
    eval(te_dataloader, fk_det_model, args)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(start_epoch, args.epochs + 1), train_losses[start_epoch - 1:])
    plt.plot(range(start_epoch, args.epochs + 1), train_losses[start_epoch - 1:], label='Train Loss')
    plt.plot(range(start_epoch, args.epochs + 1), val_losses[start_epoch - 1:], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(start_epoch, args.epochs + 1), train_accuracies[start_epoch - 1:], label='Train Accuracy')
    plt.plot(range(start_epoch, args.epochs + 1), val_accuracies[start_epoch - 1:], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.savefig(os.path.join(args.save_dir, 'training_metrics.png'))
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

# Assuming y_true and y_pred are available for test set
plot_confusion_matrix(y_true, y_pred, ['Fake', 'Real'])

# End of code



# import os
# import argparse
# import datetime
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import git

# import train
# from fake_article_dataset import PD_Dataset, PS_Dataset
# import MyFunc
# from eval import eval
# from baseline_models import bert_classifier
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# parser = argparse.ArgumentParser(description='fk_det_model text classifier')

# # data
# parser.add_argument('-root-path', type=str, default='./', help='the data directory')
# parser.add_argument('-dataset-type', type=str, default='fakenews_article', help='choose dataset to run [options: fakenews_article, sentence, fakenews_tweet]')
# parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
# parser.add_argument('-split-ratio', type=str, default='[0.7, 0.8, 1.0]', help='the split ratio of tr, dev, te sets')
# parser.add_argument('-benchmark-path', type=str, default='./benchmarked_data/', help='the benchmark data directory')

# # learning
# parser.add_argument('-lr', type=float, default=0.00001, help='initial learning rate [default: 0.001]')
# parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 256]')
# parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 32]')
# parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
# parser.add_argument('-test-interval', type=int, default=1, help='how many steps to wait before testing [default: 100]')
# parser.add_argument('-save-interval', type=int, default=1, help='how many steps to wait before saving [default:500]')
# parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# parser.add_argument('-early-stop', type=int, default=15, help='iteration numbers to stop without performance increasing')
# parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# parser.add_argument('-small_data', action='store_true', default=False, help='whether use small dataset for fast debug')
# parser.add_argument('-frequency', type=int, default=5, help='frequency of saving the model')

# # model
# parser.add_argument('-freeze-bert', action='store_true', default=True, help='freeze bert parameters')
# parser.add_argument('-bert-type', type=str, default='bert-base-cased', help='the bert embedding choice')
# parser.add_argument('-model-type', type=int, default=1, help='different structures of metric model, see document for details')
# parser.add_argument('-model-name', type=str, default='bert_classifier', help='different structures of metric model, see document for details')
# parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.3]')

# # device
# parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
# parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

# # evaluation
# parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
# parser.add_argument('-test', action='store_true', default=False, help='train or test')

# # Add a new argument to specify the directory for saving checkpoints
# parser.add_argument('-checkpoint-dir', type=str, default='./checkpoints/', help='directory to save checkpoints')

# args = parser.parse_args()

# # update args and print
# args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
# args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# # Define Git repository and checkpoint directory paths
# repo_url = "https://github.com/sk711/Med-MMHL.git"
# repo_folder = "Med-MMHL"
# # Clone the Git repository if it doesn't exist
# if not os.path.exists(repo_folder):
#     repo = git.Repo.clone_from(repo_url, repo_folder)
# else:
#     repo = git.Repo(repo_folder)

# # Define the checkpoint directory path inside the cloned repository
# checkpoint_dir = os.path.join(repo.working_tree_dir, "checkpoints")

# # Ensure the checkpoint directory exists
# os.makedirs(checkpoint_dir, exist_ok=True)
# if not os.path.exists(args.checkpoint_dir):
#     os.makedirs(args.checkpoint_dir)
# # Check if the directory for saving snapshots exists, if not, create it
# if not os.path.exists(args.save_dir):
#     os.makedirs(args.save_dir)

# benchmark_dt_path = args.benchmark_path + args.dataset_type
# tr, dev, te = MyFunc.read_benchmark_set(benchmark_dt_path)

# if 'sentence' in args.dataset_type:
#     tr_dataset = PS_Dataset(args, tr)
#     dev_dataset = PS_Dataset(args, dev)
#     te_dataset = PS_Dataset(args, te)
# else:
#     tr_dataset = PD_Dataset(args, tr)
#     dev_dataset = PD_Dataset(args, dev)
#     te_dataset = PD_Dataset(args, te)

# tr_dataloader = torch.utils.data.DataLoader(
#     tr_dataset,
#     batch_size=args.batch_size,
#     shuffle=True,
#     drop_last=True
# )
# dev_dataloader = torch.utils.data.DataLoader(
#     dev_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     drop_last=True
# )
# te_dataloader = torch.utils.data.DataLoader(
#     te_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     drop_last=True
# )

# # model
# if args.model_name == 'bert_classifier':
#     fk_det_model = bert_classifier.BertClassifier(args)
# else:
#     raise ValueError('the model_name is set wrongly')

# # Load snapshot if available
# if args.snapshot is not None:
#     print('\nLoading model from {}...'.format(args.snapshot))
#     fk_det_model.load_state_dict(torch.load(args.snapshot))

# if args.cuda:
#     torch.cuda.set_device(args.device)
#     fk_det_model = fk_det_model.cuda()

# # Training loop
# if not args.test:
#     print('Training mode')
#     train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
#     best_val_loss = float('inf')
#     best_epoch = 0

#     try:
#         print('Start training')
        
#         # Initialize start_epoch
#         start_epoch = 1
        
#         # Check if there are any existing checkpoints in the save directory
#         existing_checkpoints = [f for f in os.listdir(args.save_dir) if f.startswith('model_epoch_')]
#         if existing_checkpoints:
#             # If existing checkpoints are found, load the latest checkpoint
#             latest_checkpoint = max(existing_checkpoints)
#             start_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0]) + 1
#             print(f"Resuming training from epoch {start_epoch} using checkpoint {latest_checkpoint}")

#             # Load the state dictionary from the latest checkpoint
#             checkpoint_path = os.path.join(args.save_dir, latest_checkpoint)
#             checkpoint = torch.load(checkpoint_path)
#             fk_det_model.load_state_dict(checkpoint['model_state_dict'])
#         else:
#             print("No existing checkpoints found. Starting training from the first epoch.")
#         # Continue training from start_epoch
#         for epoch in range(start_epoch, args.epochs + 1):
#             train_loss, train_accuracy = train.train(tr_dataloader, dev_dataloader, fk_det_model, args)

#             val_loss, val_accuracy = eval(dev_dataloader, fk_det_model, args)

#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#             train_accuracies.append(train_accuracy)
#             val_accuracies.append(val_accuracy)

#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_epoch = epoch
#                 if args.save_best:
#                     torch.save(fk_det_model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))

#             if epoch % args.frequency == 0:
#                 # Save the model snapshot
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': fk_det_model.state_dict(),
#                     'train_loss': train_loss,
#                     'val_loss': val_loss,
#                     'train_accuracy': train_accuracy,
#                     'val_accuracy': val_accuracy
#                 }, os.path.join(args.checkpoint_dir, f'model_epoch_{epoch}.pt'))
#                 print(f'Model saved at epoch {epoch}')

#     except KeyboardInterrupt:
#         print('\n' + '-' * 89)
#         print('Exiting from training early')

#     print('Start testing')
#     eval(te_dataloader, fk_det_model, args)

#     # Plotting
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(range(start_epoch, args.epochs + 1), train_losses[start_epoch - 1:])
#     plt.plot(range(start_epoch, args.epochs + 1), train_losses[start_epoch - 1:], label='Train Loss')
#     plt.plot(range(start_epoch, args.epochs + 1), val_losses[start_epoch - 1:], label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(range(start_epoch, args.epochs + 1), train_accuracies[start_epoch - 1:], label='Train Accuracy')
#     plt.plot(range(start_epoch, args.epochs + 1), val_accuracies[start_epoch - 1:], label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.legend()

#     plt.savefig(os.path.join(args.save_dir, 'training_metrics.png'))
#     plt.show()

# # Plot confusion matrix
# def plot_confusion_matrix(y_true, y_pred, classes):
#     cm = confusion_matrix(y_true, y_pred)
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
#     plt.xlabel('Predicted labels')
#     plt.ylabel('True labels')
#     plt.title('Confusion Matrix')
#     plt.show()

# # Assuming y_true and y_pred are available for test set
# plot_confusion_matrix(y_true, y_pred, ['Fake', 'Real'])

# # End of code





# #21MArch-6:21
# #====================

# import os
# import argparse
# import datetime
# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# import train
# from fake_article_dataset import PD_Dataset, PS_Dataset
# import MyFunc
# from eval import eval
# from baseline_models import bert_classifier
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# parser = argparse.ArgumentParser(description='fk_det_model text classifier')

# # data
# parser.add_argument('-root-path', type=str, default='./', help='the data directory')
# parser.add_argument('-dataset-type', type=str, default='fakenews_article', help='choose dataset to run [options: fakenews_article, sentence, fakenews_tweet]')
# parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
# parser.add_argument('-split-ratio', type=str, default='[0.7, 0.8, 1.0]', help='the split ratio of tr, dev, te sets')
# parser.add_argument('-benchmark-path', type=str, default='./benchmarked_data/', help='the benchmark data directory')
# parser.add_argument('-checkpoint-dir', type=str, default='./checkpoints/', help='directory to save checkpoints')

# # learning
# parser.add_argument('-lr', type=float, default=0.00001, help='initial learning rate [default: 0.001]')
# parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 256]')
# parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 32]')
# parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
# parser.add_argument('-test-interval', type=int, default=1, help='how many steps to wait before testing [default: 100]')
# parser.add_argument('-save-interval', type=int, default=1, help='how many steps to wait before saving [default:500]')
# parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# parser.add_argument('-early-stop', type=int, default=15, help='iteration numbers to stop without performance increasing')
# parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# parser.add_argument('-small_data', action='store_true', default=False, help='whether use small dataset for fast debug')
# parser.add_argument('-frequency', type=int, default=5, help='frequency of saving the model')

# # model
# parser.add_argument('-freeze-bert', action='store_true', default=True, help='freeze bert parameters')
# parser.add_argument('-bert-type', type=str, default='bert-base-cased', help='the bert embedding choice')
# parser.add_argument('-model-type', type=int, default=1, help='different structures of metric model, see document for details')
# parser.add_argument('-model-name', type=str, default='bert_classifier', help='different structures of metric model, see document for details')
# parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.3]')

# # device
# parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
# parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

# # evaluation
# parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
# parser.add_argument('-test', action='store_true', default=False, help='train or test')

# # Add a new argument to specify the directory for saving checkpoints
# parser.add_argument('-checkpoint-dir', type=str, default='./checkpoints/', help='directory to save checkpoints')

# args = parser.parse_args()

# # update args and print
# args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
# args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# # Check if the directory for saving checkpoints exists, if not, create it

# # Define Git repository and checkpoint directory paths
# repo = git.Repo("https://github.com/sk711/Med-MMHL/tree/2d7e0732a41548e224aeca7a517f4e69ac35e8e9/Multimodal_fake_news_benchmark_1")
# checkpoint_dir = os.path.join(repo.working_tree_dir, "checkpoints")

# # Ensure the checkpoint directory exists
# os.makedirs(checkpoint_dir, exist_ok=True)
# if not os.path.exists(args.checkpoint_dir):
#     os.makedirs(args.checkpoint_dir)
# # Check if the directory for saving snapshots exists, if not, create it
# if not os.path.exists(args.save_dir):
#     os.makedirs(args.save_dir)

# benchmark_dt_path = args.benchmark_path + args.dataset_type
# tr, dev, te = MyFunc.read_benchmark_set(benchmark_dt_path)

# if 'sentence' in args.dataset_type:
#     tr_dataset = PS_Dataset(args, tr)
#     dev_dataset = PS_Dataset(args, dev)
#     te_dataset = PS_Dataset(args, te)
# else:
#     tr_dataset = PD_Dataset(args, tr)
#     dev_dataset = PD_Dataset(args, dev)
#     te_dataset = PD_Dataset(args, te)

# tr_dataloader = torch.utils.data.DataLoader(
#     tr_dataset,
#     batch_size=args.batch_size,
#     shuffle=True,
#     drop_last=True
# )
# dev_dataloader = torch.utils.data.DataLoader(
#     dev_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     drop_last=True
# )
# te_dataloader = torch.utils.data.DataLoader(
#     te_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     drop_last=True
# )

# # model
# if args.model_name == 'bert_classifier':
#     fk_det_model = bert_classifier.BertClassifier(args)
# else:
#     raise ValueError('the model_name is set wrongly')

# # Load snapshot if available
# if args.snapshot is not None:
#     print('\nLoading model from {}...'.format(args.snapshot))
#     fk_det_model.load_state_dict(torch.load(args.snapshot))

# if args.cuda:
#     torch.cuda.set_device(args.device)
#     fk_det_model = fk_det_model.cuda()

# # Training loop
# if not args.test:
#     print('Training mode')
#     train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
#     best_val_loss = float('inf')
#     best_epoch = 0

#     try:
#         print('Start training')
        
#         # Initialize start_epoch
#         start_epoch = 1
        
#         # Check if there are any existing checkpoints in the save directory
#         existing_checkpoints = [f for f in os.listdir(args.save_dir) if f.startswith('model_epoch_')]
#         if existing_checkpoints:
#             # If existing checkpoints are found, load the latest checkpoint
#             latest_checkpoint = max(existing_checkpoints)
#             start_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0]) + 1
#             print(f"Resuming training from epoch {start_epoch} using checkpoint {latest_checkpoint}")

#             # Load the state dictionary from the latest checkpoint
#             checkpoint_path = os.path.join(args.save_dir, latest_checkpoint)
#             checkpoint = torch.load(checkpoint_path)
#             fk_det_model.load_state_dict(checkpoint['model_state_dict'])
#         else:
#             print("No existing checkpoints found. Starting training from the first epoch.")
#         # Continue training from start_epoch
#         for epoch in range(start_epoch, args.epochs + 1):
#             train_loss, train_accuracy = train.train(tr_dataloader, dev_dataloader, fk_det_model, args)

#             val_loss, val_accuracy = eval(dev_dataloader, fk_det_model, args)

#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#             train_accuracies.append(train_accuracy)
#             val_accuracies.append(val_accuracy)

#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_epoch = epoch
#                 if args.save_best:
#                     torch.save(fk_det_model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))

#             if epoch % args.frequency == 0:
#                 # Save the model snapshot
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': fk_det_model.state_dict(),
#                     'train_loss': train_loss,
#                     'val_loss': val_loss,
#                     'train_accuracy': train_accuracy,
#                     'val_accuracy': val_accuracy
#                 }, os.path.join(args.checkpoint_dir, f'model_epoch_{epoch}.pt'))
#                 print(f'Model saved at epoch {epoch}')

#     except KeyboardInterrupt:
#         print('\n' + '-' * 89)
#         print('Exiting from training early')

#     print('Start testing')
#     eval(te_dataloader, fk_det_model, args)

#     # Plotting
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(range(start_epoch, args.epochs + 1), train_losses[start_epoch - 1:])
#     plt.plot(range(start_epoch, args.epochs + 1), train_losses[start_epoch - 1:], label='Train Loss')
#     plt.plot(range(start_epoch, args.epochs + 1), val_losses[start_epoch - 1:], label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(range(start_epoch, args.epochs + 1), train_accuracies[start_epoch - 1:], label='Train Accuracy')
#     plt.plot(range(start_epoch, args.epochs + 1), val_accuracies[start_epoch - 1:], label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.legend()

#     plt.savefig(os.path.join(args.save_dir, 'training_metrics.png'))
#     plt.show()

# # Plot confusion matrix
# def plot_confusion_matrix(y_true, y_pred, classes):
#     cm = confusion_matrix(y_true, y_pred)
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
#     plt.xlabel('Predicted labels')
#     plt.ylabel('True labels')
#     plt.title('Confusion Matrix')
#     plt.show()

# # Assuming y_true and y_pred are available for test set
# plot_confusion_matrix(y_true, y_pred, ['Fake', 'Real'])

# # # End of code






#21MArch - 12:53
#===============================================
"""
This code implements a text classifier for fake news detection using a BERT-based model.
It sets up command-line arguments for configuration, loads the dataset, initializes the model,
and trains it while logging performance metrics. After training, it evaluates the model on a test set
and displays the results. The training and validation losses and accuracies are visualized over epochs,
and a confusion matrix is generated. Additionally, checkpoints are saved after each epoch,
allowing for the resumption of training from the last saved checkpoint.
"""
# import os
# import argparse
# import datetime
# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# import train
# from fake_article_dataset import PD_Dataset, PS_Dataset
# import MyFunc
# from eval import eval
# from baseline_models import bert_classifier
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# parser = argparse.ArgumentParser(description='fk_det_model text classifier')

# # data
# parser.add_argument('-root-path', type=str, default='./', help='the data directory')
# parser.add_argument('-dataset-type', type=str, default='fakenews_article', help='choose dataset to run [options: fakenews_article, sentence, fakenews_tweet]')
# parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
# parser.add_argument('-split-ratio', type=str, default='[0.7, 0.8, 1.0]', help='the split ratio of tr, dev, te sets')
# parser.add_argument('-benchmark-path', type=str, default='./benchmarked_data/', help='the benchmark data directory')

# # learning
# parser.add_argument('-lr', type=float, default=0.00001, help='initial learning rate [default: 0.001]')
# parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 256]')
# parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 32]')
# parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
# parser.add_argument('-test-interval', type=int, default=1, help='how many steps to wait before testing [default: 100]')
# parser.add_argument('-save-interval', type=int, default=1, help='how many steps to wait before saving [default:500]')
# parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# parser.add_argument('-early-stop', type=int, default=15, help='iteration numbers to stop without performance increasing')
# parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# parser.add_argument('-small_data', action='store_true', default=False, help='whether use small dataset for fast debug')
# parser.add_argument('-frequency', type=int, default=5, help='frequency of saving the model')

# # model
# parser.add_argument('-freeze-bert', action='store_true', default=True, help='freeze bert parameters')
# parser.add_argument('-bert-type', type=str, default='bert-base-cased', help='the bert embedding choice')
# parser.add_argument('-model-type', type=int, default=1, help='different structures of metric model, see document for details')
# parser.add_argument('-model-name', type=str, default='bert_classifier', help='different structures of metric model, see document for details')
# parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.3]')

# # device
# parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
# parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

# # evaluation
# parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
# parser.add_argument('-test', action='store_true', default=False, help='train or test')

# # Add a new argument to specify the directory for saving checkpoints
# parser.add_argument('-checkpoint-dir', type=str, default='./checkpoints/', help='directory to save checkpoints')

# args = parser.parse_args()

# # update args and print
# args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
# args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# # Check if the directory for saving checkpoints exists, if not, create it
# if not os.path.exists(args.checkpoint_dir):
#     os.makedirs(args.checkpoint_dir)
# # Check if the directory for saving snapshots exists, if not, create it
# if not os.path.exists(args.save_dir):
#     os.makedirs(args.save_dir)

# benchmark_dt_path = args.benchmark_path + args.dataset_type
# tr, dev, te = MyFunc.read_benchmark_set(benchmark_dt_path)

# if 'sentence' in args.dataset_type:
#     tr_dataset = PS_Dataset(args, tr)
#     dev_dataset = PS_Dataset(args, dev)
#     te_dataset = PS_Dataset(args, te)
# else:
#     tr_dataset = PD_Dataset(args, tr)
#     dev_dataset = PD_Dataset(args, dev)
#     te_dataset = PD_Dataset(args, te)

# tr_dataloader = torch.utils.data.DataLoader(
#     tr_dataset,
#     batch_size=args.batch_size,
#     shuffle=True,
#     drop_last=True
# )
# dev_dataloader = torch.utils.data.DataLoader(
#     dev_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     drop_last=True
# )
# te_dataloader = torch.utils.data.DataLoader(
#     te_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     drop_last=True
# )

# # model
# if args.model_name == 'bert_classifier':
#     fk_det_model = bert_classifier.BertClassifier(args)
# else:
#     raise ValueError('the model_name is set wrongly')

# # Load snapshot if available
# if args.snapshot is not None:
#     print('\nLoading model from {}...'.format(args.snapshot))
#     fk_det_model.load_state_dict(torch.load(args.snapshot))

# if args.cuda:
#     torch.cuda.set_device(args.device)
#     fk_det_model = fk_det_model.cuda()

# # Training loop
# if not args.test:
#     print('Training mode')
#     train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
#     best_val_loss = float('inf')
#     best_epoch = 0

#     try:
#         print('Start training')
        
#         # Initialize start_epoch
#         start_epoch = 1
        
#         # Check if there are any existing checkpoints in the save directory
#         existing_checkpoints = [f for f in os.listdir(args.save_dir) if f.startswith('model_epoch_')]
#         if existing_checkpoints:
#             # If existing checkpoints are found, load the latest checkpoint
#             latest_checkpoint = max(existing_checkpoints)
#             start_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0]) + 1
#             print(f"Resuming training from epoch {start_epoch} using checkpoint {latest_checkpoint}")

#             # Load the state dictionary from the latest checkpoint
#             checkpoint_path = os.path.join(args.save_dir, latest_checkpoint)
#             checkpoint = torch.load(checkpoint_path)
#             fk_det_model.load_state_dict(checkpoint['model_state_dict'])
#         else:
#             print("No existing checkpoints found. Starting training from the first epoch.")
#         # Continue training from start_epoch
#         for epoch in range(start_epoch, args.epochs + 1):
#             train_loss, train_accuracy = train.train(tr_dataloader, dev_dataloader, fk_det_model, args)

#             val_loss, val_accuracy = eval(dev_dataloader, fk_det_model, args)

#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#             train_accuracies.append(train_accuracy)
#             val_accuracies.append(val_accuracy)

#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_epoch = epoch
#                 if args.save_best:
#                     torch.save(fk_det_model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))

#             if epoch % args.frequency == 0:
#                 # Save the model snapshot
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': fk_det_model.state_dict(),
#                     'train_loss': train_loss,
#                     'val_loss': val_loss,
#                     'train_accuracy': train_accuracy,
#                     'val_accuracy': val_accuracy
#                 }, os.path.join(args.checkpoint_dir, f'model_epoch_{epoch}.pt'))
#                 print(f'Model saved at epoch {epoch}')

#     except KeyboardInterrupt:
#         print('\n' + '-' * 89)
#         print('Exiting from training early')

#     print('Start testing')
#     eval(te_dataloader, fk_det_model, args)

#     # Plotting
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(range(start_epoch, args.epochs + 1), train_losses[start_epoch - 1:])
#     plt.plot(range(start_epoch, args.epochs + 1), train_losses[start_epoch - 1:], label='Train Loss')
#     plt.plot(range(start_epoch, args.epochs + 1), val_losses[start_epoch - 1:], label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(range(start_epoch, args.epochs + 1), train_accuracies[start_epoch - 1:], label='Train Accuracy')
#     plt.plot(range(start_epoch, args.epochs + 1), val_accuracies[start_epoch - 1:], label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.legend()

#     plt.savefig(os.path.join(args.save_dir, 'training_metrics.png'))
#     plt.show()

# # Plot confusion matrix
# def plot_confusion_matrix(y_true, y_pred, classes):
#     cm = confusion_matrix(y_true, y_pred)
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
#     plt.xlabel('Predicted labels')
#     plt.ylabel('True labels')
#     plt.title('Confusion Matrix')
#     plt.show()

# # Assuming y_true and y_pred are available for test set
# plot_confusion_matrix(y_true, y_pred, ['Fake', 'Real'])

# # End of code





#21MArch - 11:57
#===============================================

"""
This code implements a text classifier for fake news detection using a BERT-based model.
It sets up command-line arguments for configuration, loads the dataset, initializes the model,
and trains it while logging performance metrics. After training, it evaluates the model on a test set
and displays the results. The training and validation losses and accuracies are visualized over epochs,
and a confusion matrix is generated. Additionally, checkpoints are saved after each epoch,
allowing for the resumption of training from the last saved checkpoint.
"""

# import os
# import argparse
# import datetime
# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# import train
# from fake_article_dataset import PD_Dataset, PS_Dataset
# import MyFunc
# from eval import eval
# from baseline_models import bert_classifier
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# parser = argparse.ArgumentParser(description='fk_det_model text classifier')

# # data
# parser.add_argument('-root-path', type=str, default='./', help='the data directory')
# parser.add_argument('-dataset-type', type=str, default='fakenews_article', help='choose dataset to run [options: fakenews_article, sentence, fakenews_tweet]')
# parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
# parser.add_argument('-split-ratio', type=str, default='[0.7, 0.8, 1.0]', help='the split ratio of tr, dev, te sets')
# parser.add_argument('-benchmark-path', type=str, default='./benchmarked_data/', help='the benchmark data directory')

# # learning
# parser.add_argument('-lr', type=float, default=0.00001, help='initial learning rate [default: 0.001]')
# parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 256]')
# parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 32]')
# parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
# parser.add_argument('-test-interval', type=int, default=1, help='how many steps to wait before testing [default: 100]')
# parser.add_argument('-save-interval', type=int, default=1, help='how many steps to wait before saving [default:500]')
# parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# parser.add_argument('-early-stop', type=int, default=15, help='iteration numbers to stop without performance increasing')
# parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# parser.add_argument('-small_data', action='store_true', default=False, help='whether use small dataset for fast debug')
# parser.add_argument('-frequency', type=int, default=5, help='frequency of saving the model')

# # model
# parser.add_argument('-freeze-bert', action='store_true', default=True, help='freeze bert parameters')
# parser.add_argument('-bert-type', type=str, default='bert-base-cased', help='the bert embedding choice')
# parser.add_argument('-model-type', type=int, default=1, help='different structures of metric model, see document for details')
# parser.add_argument('-model-name', type=str, default='bert_classifier', help='different structures of metric model, see document for details')
# parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.3]')

# # device
# parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
# parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

# # evaluation
# parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
# parser.add_argument('-test', action='store_true', default=False, help='train or test')
# args = parser.parse_args()

# # update args and print
# args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
# args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# benchmark_dt_path = args.benchmark_path + args.dataset_type
# tr, dev, te = MyFunc.read_benchmark_set(benchmark_dt_path)

# if 'sentence' in args.dataset_type:
#     tr_dataset = PS_Dataset(args, tr)
#     dev_dataset = PS_Dataset(args, dev)
#     te_dataset = PS_Dataset(args, te)
# else:
#     tr_dataset = PD_Dataset(args, tr)
#     dev_dataset = PD_Dataset(args, dev)
#     te_dataset = PD_Dataset(args, te)

# tr_dataloader = torch.utils.data.DataLoader(
#     tr_dataset,
#     batch_size=args.batch_size,
#     shuffle=True,
#     drop_last=True
# )
# dev_dataloader = torch.utils.data.DataLoader(
#     dev_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     drop_last=True
# )
# te_dataloader = torch.utils.data.DataLoader(
#     te_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     drop_last=True
# )

# # model
# if args.model_name == 'bert_classifier':
#     fk_det_model = bert_classifier.BertClassifier(args)
# else:
#     raise ValueError('the model_name is set wrongly')

# # Load snapshot if available
# if args.snapshot is not None:
#     print('\nLoading model from {}...'.format(args.snapshot))
#     fk_det_model.load_state_dict(torch.load(args.snapshot))

# if args.cuda:
#     torch.cuda.set_device(args.device)
#     fk_det_model = fk_det_model.cuda()

# # Training loop
# if not args.test:
#     print('Training mode')
#     train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
#     best_val_loss = float('inf')
#     best_epoch = 0

#     try:
#         print('Start training')
        
#         # Initialize start_epoch
#         start_epoch = 1
        
#         # Check if there are any existing checkpoints in the save directory
#         existing_checkpoints = [f for f in os.listdir(args.save_dir) if f.startswith('model_epoch_')]
#         if existing_checkpoints:
#             # If existing checkpoints are found, load the latest checkpoint
#             latest_checkpoint = max(existing_checkpoints)
#             start_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0]) + 1
#             print(f"Resuming training from epoch {start_epoch} using checkpoint {latest_checkpoint}")

#             # Load the state dictionary from the latest checkpoint
#             checkpoint_path = os.path.join(args.save_dir, latest_checkpoint)
#             checkpoint = torch.load(checkpoint_path)
#             fk_det_model.load_state_dict(checkpoint['model_state_dict'])

#         # Continue training from start_epoch
#         for epoch in range(start_epoch, args.epochs + 1):
#             train_loss, train_accuracy = train.train(tr_dataloader, dev_dataloader, fk_det_model, args)

#             val_loss, val_accuracy = eval(dev_dataloader, fk_det_model, args)

#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#             train_accuracies.append(train_accuracy)
#             val_accuracies.append(val_accuracy)

#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_epoch = epoch
#                 if args.save_best:
#                                     torch.save(fk_det_model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))

#             if epoch % args.frequency == 0:
#                 # Save the model snapshot
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': fk_det_model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'train_loss': train_loss,
#                     'val_loss': val_loss,
#                     'train_accuracy': train_accuracy,
#                     'val_accuracy': val_accuracy
#                 }, os.path.join(args.save_dir, f'model_epoch_{epoch}.pt'))
#                 print(f'Model saved at epoch {epoch}')

#     except KeyboardInterrupt:
#         print('\n' + '-' * 89)
#         print('Exiting from training early')

#     print('Start testing')
#     eval(te_dataloader, fk_det_model, args)

#     # Plotting
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(range(start_epoch, args.epochs + 1), train_losses[start_epoch - 1:])
#     plt.plot(range(start_epoch, args.epochs + 1), train_losses[start_epoch - 1:], label='Train Loss')
#     plt.plot(range(start_epoch, args.epochs + 1), val_losses[start_epoch - 1:], label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(range(start_epoch, args.epochs + 1), train_accuracies[start_epoch - 1:], label='Train Accuracy')
#     plt.plot(range(start_epoch, args.epochs + 1), val_accuracies[start_epoch - 1:], label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.legend()

#     plt.savefig(os.path.join(args.save_dir, 'training_metrics.png'))
#     plt.show()

# # Plot confusion matrix
# def plot_confusion_matrix(y_true, y_pred, classes):
#     cm = confusion_matrix(y_true, y_pred)
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
#     plt.xlabel('Predicted labels')
#     plt.ylabel('True labels')
#     plt.title('Confusion Matrix')
#     plt.show()


# # Assuming y_true and y_pred are available for test set
# plot_confusion_matrix(y_true, y_pred, ['Fake', 'Real'])

# # End of code








# 21March - Morning===================================================================
# import os
# import argparse
# import datetime
# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# import train
# from fake_article_dataset import PD_Dataset, PS_Dataset
# import MyFunc
# from eval import eval
# from baseline_models import bert_classifier
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# parser = argparse.ArgumentParser(description='fk_det_model text classifier')

# # data
# parser.add_argument('-root-path', type=str, default='./', help='the data directory')
# parser.add_argument('-dataset-type', type=str, default='fakenews_article', help='choose dataset to run [options: fakenews_article, sentence, fakenews_tweet]')
# parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
# parser.add_argument('-split-ratio', type=str, default='[0.7, 0.8, 1.0]', help='the split ratio of tr, dev, te sets')
# parser.add_argument('-benchmark-path', type=str, default='./benchmarked_data/', help='the benchmark data directory')

# # learning
# parser.add_argument('-lr', type=float, default=0.00001, help='initial learning rate [default: 0.001]')
# parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 256]')
# parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 32]')
# parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
# parser.add_argument('-test-interval', type=int, default=1, help='how many steps to wait before testing [default: 100]')
# parser.add_argument('-save-interval', type=int, default=1, help='how many steps to wait before saving [default:500]')
# parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# parser.add_argument('-early-stop', type=int, default=15, help='iteration numbers to stop without performance increasing')
# parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# parser.add_argument('-small_data', action='store_true', default=False, help='whether use small dataset for fast debug')
# parser.add_argument('-frequency', type=int, default=5, help='frequency of saving the model')

# # model
# parser.add_argument('-freeze-bert', action='store_true', default=True, help='freeze bert parameters')
# parser.add_argument('-bert-type', type=str, default='bert-base-cased', help='the bert embedding choice')
# parser.add_argument('-model-type', type=int, default=1, help='different structures of metric model, see document for details')
# parser.add_argument('-model-name', type=str, default='bert_classifier', help='different structures of metric model, see document for details')
# parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.3]')

# # device
# parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
# parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

# # evaluation
# parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
# parser.add_argument('-test', action='store_true', default=False, help='train or test')
# args = parser.parse_args()

# # update args and print
# args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
# args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# benchmark_dt_path = args.benchmark_path + args.dataset_type
# tr, dev, te = MyFunc.read_benchmark_set(benchmark_dt_path)

# if 'sentence' in args.dataset_type:
#     tr_dataset = PS_Dataset(args, tr)
#     dev_dataset = PS_Dataset(args, dev)
#     te_dataset = PS_Dataset(args, te)
# else:
#     tr_dataset = PD_Dataset(args, tr)
#     dev_dataset = PD_Dataset(args, dev)
#     te_dataset = PD_Dataset(args, te)

# tr_dataloader = torch.utils.data.DataLoader(
#     tr_dataset,
#     batch_size=args.batch_size,
#     shuffle=True,
#     drop_last=True
# )
# dev_dataloader = torch.utils.data.DataLoader(
#     dev_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     drop_last=True
# )
# te_dataloader = torch.utils.data.DataLoader(
#     te_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     drop_last=True
# )

# # model
# if args.model_name == 'bert_classifier':
#     fk_det_model = bert_classifier.BertClassifier(args)
# else:
#     raise ValueError('the model_name is set wrongly')

# if args.snapshot is not None:
#     print('\nLoading model from {}...'.format(args.snapshot))
#     fk_det_model.load_state_dict(torch.load(args.snapshot))

# if args.cuda:
#     torch.cuda.set_device(args.device)
#     fk_det_model = fk_det_model.cuda()

# # Training loop
# if not args.test:
#     print('Training mode')
#     train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
#     best_val_loss = float('inf')
#     best_epoch = 0

#     try:
#         print('Start training')
#         for epoch in range(1, args.epochs + 1):
#             train_loss, train_accuracy = train.train(tr_dataloader, dev_dataloader, fk_det_model, args)

#             val_loss, val_accuracy = eval(dev_dataloader, fk_det_model, args)

#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#             train_accuracies.append(train_accuracy)
#             val_accuracies.append(val_accuracy)

#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_epoch = epoch
#                 if args.save_best:
#                     torch.save(fk_det_model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))

#             if epoch % args.frequency == 0:
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': fk_det_model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'train_loss': train_loss,
#                     'val_loss': val_loss,
#                     'train_accuracy': train_accuracy,
#                     'val_accuracy': val_accuracy
#                 }, os.path.join(args.save_dir, f'model_epoch_{epoch}.pt'))
#                 print(f'Model saved at epoch {epoch}')

#     except KeyboardInterrupt:
#         print('\n' + '-' * 89)
#         print('Exiting from training early')

#     print('Start testing')
#     eval(te_dataloader, fk_det_model, args)

#     # Plotting
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(range(1, args.epochs + 1), train_losses)
#     plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(range(1, args.epochs + 1), train_accuracies, label='Train Accuracy')
#     plt.plot(range(1, args.epochs + 1), val_accuracies, label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.legend()

#     plt.savefig(os.path.join(args.save_dir, 'training_metrics.png'))
#     plt.show()

# # Plot confusion matrix
# def plot_confusion_matrix(y_true, y_pred, classes):
#     cm = confusion_matrix(y_true, y_pred)
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
#     plt.xlabel('Predicted labels')
#     plt.ylabel('True labels')
#     plt.title('Confusion Matrix')
#     plt.show()


# # Assuming y_true and y_pred are available for test set
# plot_confusion_matrix(y_true, y_pred, ['Fake', 'Real'])

# # End of code




# #! /usr/bin/env python
# # -- coding: utf-8 --
# import os
# import argparse
# import datetime
# import torch

# import train

# from fake_article_dataset import PD_Dataset, PS_Dataset
# import MyFunc

# from eval import eval
# from baseline_models import bert_classifier

# parser = argparse.ArgumentParser(description='fk_det_model text classificer')

# # data
# parser.add_argument('-root-path', type=str, default='./', help='the data directory')
# parser.add_argument('-dataset-type', type=str, default='fakenews_article', help='choose dataset to run [options: fakenews_article, sentence, fakenews_tweet]')
# parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
# parser.add_argument('-split-ratio', type=str, default='[0.7, 0.8, 1.0]', help='the split ratio of tr, dev, te sets')
# #parser.add_argument('-benchmark-path', type=str, default='/kaggle/input/medmmhl/', help='the benchmark data directory')
# parser.add_argument('-benchmark-path', type=str, default='./benchmarked_data/', help='the benchmark data directory')

# # learning
# parser.add_argument('-lr', type=float, default=0.00001, help='initial learning rate [default: 0.001]')
# parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 256]')
# parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 32]')
# parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
# parser.add_argument('-test-interval', type=int, default=1, help='how many steps to wait before testing [default: 100]')
# parser.add_argument('-save-interval', type=int, default=1, help='how many steps to wait before saving [default:500]')
# parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# parser.add_argument('-early-stop', type=int, default=15, help='iteration numbers to stop without performance increasing')
# parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# parser.add_argument('-small_data', action='store_true', default=False, help='whether use small dataset for fast debug')
# parser.add_argument('-frequency', type=int, default=5, help='frequency of saving the model')

# # model
# parser.add_argument('-freeze-bert', action='store_true', default=True, help='freeze bert parameters')
# parser.add_argument('-bert-type', type=str, default='bert-base-cased', help='the bert embedding choice') # to be replaced by https://huggingface.co/models?sort=downloads&search=BERTweet
# parser.add_argument('-model-type', type=int, default=1, help='different structures of metric model, see document for details')
# parser.add_argument('-model-name', type=str, default='bert_classifier', help='different structures of metric model, see document for details')
# parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.3]')

# # device
# parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
# parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

# # evaluation
# parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
# parser.add_argument('-test', action='store_true', default=False, help='train or test')
# args = parser.parse_args()

# # update args and print
# args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
# args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


# benchmark_dt_path=args.benchmark_path + args.dataset_type
# tr, dev, te = MyFunc.read_benchmark_set(benchmark_dt_path)

# if 'sentence' in args.dataset_type:
#     tr_dataset = PS_Dataset(args, tr)
#     dev_dataset = PS_Dataset(args, dev)
#     te_dataset = PS_Dataset(args, te)
# else:
#     tr_dataset = PD_Dataset(args, tr)
#     dev_dataset = PD_Dataset(args, dev)
#     te_dataset = PD_Dataset(args, te)

# tr_dataloader = torch.utils.data.DataLoader(
#     tr_dataset,
#     batch_size=args.batch_size,
#     shuffle=True,
#     drop_last=True
# )
# dev_dataloader = torch.utils.data.DataLoader(
#     dev_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     drop_last=True
# )
# te_dataloader = torch.utils.data.DataLoader(
#     te_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     drop_last=True
# )
# # model
# if args.model_name == 'bert_classifier': # used for any transformer model
#     fk_det_model = bert_classifier.BertClassifier(args)
# else:
#     raise ValueError('the model_name is set wrongly')

# if args.snapshot is not None:
#     print('\nLoading model from {}...'.format(args.snapshot))
#     fk_det_model.load_state_dict(torch.load(args.snapshot))

# if args.cuda:
#     torch.cuda.set_device(args.device)
#     fk_det_model = fk_det_model.cuda()

# # train or predict
# # if args.predict is not None:
# #     label = train.predict(args.predict, fk_det_model, x_test, y_test, args.cuda)
# #     print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
# if args.test:
#     print('use test mode')
#     eval(te_dataloader, fk_det_model, args)

# else:
#     print('use train mode')
#     try:
#         print('start training')
#         for epoch in range(1, args.epochs + 1):
#             train.train(tr_dataloader, dev_dataloader, fk_det_model, args)
#             if epoch % args.frequency == 0:
#                 torch.save(fk_det_model.state_dict(), os.path.join(args.save_dir, f'model_epoch_{epoch}.pt'))
#                 print(f'Model saved at epoch {epoch}')
#     except KeyboardInterrupt:
#         print('\n' + '-' * 89)
#         print('Exiting from training early')
#     print('start testing')
#     eval(te_dataloader, fk_det_model, args)











# #! /usr/bin/env python
# # -- coding: utf-8 --
# import os
# import argparse
# import datetime
# import torch

# import train

# from fake_article_dataset import PD_Dataset, PS_Dataset
# import MyFunc

# from eval import eval
# from baseline_models import bert_classifier

# parser = argparse.ArgumentParser(description='fk_det_model text classificer')

# # data
# parser.add_argument('-root-path', type=str, default='./', help='the data directory')
# parser.add_argument('-dataset-type', type=str, default='fakenews_article', help='choose dataset to run [options: fakenews_article, sentence, fakenews_tweet]')
# parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
# parser.add_argument('-split-ratio', type=str, default='[0.7, 0.8, 1.0]', help='the split ratio of tr, dev, te sets')
# #parser.add_argument('-benchmark-path', type=str, default='/kaggle/input/medmmhl/', help='the benchmark data directory')
# parser.add_argument('-benchmark-path', type=str, default='./benchmarked_data/', help='the benchmark data directory')

# # learning
# parser.add_argument('-lr', type=float, default=0.00001, help='initial learning rate [default: 0.001]')
# parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 256]')
# parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 32]')
# parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
# parser.add_argument('-test-interval', type=int, default=1, help='how many steps to wait before testing [default: 100]')
# parser.add_argument('-save-interval', type=int, default=10, help='how many steps to wait before saving [default:500]')
# parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# parser.add_argument('-early-stop', type=int, default=15, help='iteration numbers to stop without performance increasing')
# parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# parser.add_argument('-small_data', action='store_true', default=False, help='whether use small dataset for fast debug')

# # model
# parser.add_argument('-freeze-bert', action='store_true', default=True, help='freeze bert parameters')
# parser.add_argument('-bert-type', type=str, default='bert-base-cased', help='the bert embedding choice') # to be replaced by https://huggingface.co/models?sort=downloads&search=BERTweet
# parser.add_argument('-model-type', type=int, default=1, help='different structures of metric model, see document for details')
# parser.add_argument('-model-name', type=str, default='bert_classifier', help='different structures of metric model, see document for details')
# parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.3]')

# # device
# parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
# parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

# # evaluation
# parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
# parser.add_argument('-test', action='store_true', default=False, help='train or test')
# args = parser.parse_args()

# # update args and print
# args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
# args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


# benchmark_dt_path=args.benchmark_path + args.dataset_type
# tr, dev, te = MyFunc.read_benchmark_set(benchmark_dt_path)

# if 'sentence' in args.dataset_type:
#     tr_dataset = PS_Dataset(args, tr)
#     dev_dataset = PS_Dataset(args, dev)
#     te_dataset = PS_Dataset(args, te)
# else:
#     tr_dataset = PD_Dataset(args, tr)
#     dev_dataset = PD_Dataset(args, dev)
#     te_dataset = PD_Dataset(args, te)

# tr_dataloader = torch.utils.data.DataLoader(
#     tr_dataset,
#     batch_size=args.batch_size,
#     shuffle=True,
#     drop_last=True
# )
# dev_dataloader = torch.utils.data.DataLoader(
#     dev_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     drop_last=True
# )
# te_dataloader = torch.utils.data.DataLoader(
#     te_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     drop_last=True
# )

# # (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataset.generate_data()
# # args.class_num = tr_dataset.get_class_num()

# print("\nParameters:")
# for attr, value in sorted(args.__dict__.items()):
#     print("\t{}={}".format(attr.upper(), value))

# # model
# if args.model_name == 'bert_classifier': # used for any transformer model
#     fk_det_model = bert_classifier.BertClassifier(args)
# else:
#     raise ValueError('the model_name is set wrongly')

# if args.snapshot is not None:
#     print('\nLoading model from {}...'.format(args.snapshot))
#     fk_det_model.load_state_dict(torch.load(args.snapshot))

# if args.cuda:
#     torch.cuda.set_device(args.device)
#     fk_det_model = fk_det_model.cuda()

# # train or predict
# # if args.predict is not None:
# #     label = train.predict(args.predict, fk_det_model, x_test, y_test, args.cuda)
# #     print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
# if args.test:
#     print('use test mode')
#     eval(te_dataloader, fk_det_model, args)

# else:
#     print('use train mode')
#     try:
#         print('start training')
#         train.train(tr_dataloader, dev_dataloader, fk_det_model, args)
#     except KeyboardInterrupt:
#         print('\n' + '-' * 89)
#         print('Exiting from training early')
#     print('start testing')
#     eval(te_dataloader, fk_det_model, args)


