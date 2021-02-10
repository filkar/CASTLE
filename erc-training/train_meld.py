import numpy as np, argparse, time, pickle, random, math
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as PLT
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataloader import MELDRobertaCometDataset
from model import MaskedNLLLoss
from commonsense_model import CommonsenseLSTMModel 
#CommonsenseLSTMModel
from sklearn.metrics import f1_score, accuracy_score

import os

from ray import tune
from ray.tune.schedulers import ASHAScheduler

def create_class_weight(mu=1):
    unique = [0, 1, 2, 3, 4, 5, 6]
    labels_dict = {0: 6436, 1: 1636, 2: 358, 3: 1002, 4: 2308, 5: 361, 6: 1607}        
    total = np.sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = math.log(mu*total/labels_dict[key])
        weights.append(score)
    return weights

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def plot_training(e, test_fscore, figsize=(20, 14), show=False):
    
  plt.ylim(0,100)

  x.append(e)
  y.append(test_fscore)
  
  plt.title('Training accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('F_score')
  
  
  plt.legend()
  
  print(show)
  if e == n_epochs - 1 or show is True:
    plt.plot(x, y, 'bo-')  
    plt.show()


def get_MELD_loaders(batch_size=32, classify='emotion', num_workers=0, pin_memory=False):
    trainset = MELDRobertaCometDataset('train', classify)
    validset = MELDRobertaCometDataset('valid', classify)
    testset = MELDRobertaCometDataset('test', classify)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks, losses_sense  = [], [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(seed)
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        r1, r2, r3, r4, \
        x1, x2, x3, x4, x5, x6, \
        o1, o2, o3, \
        qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        
        log_prob, _, alpha, alpha_f, alpha_b, _ = model(r1, r2, r3, r4, x5, x6, x1, o2, o3, qmask, umask, att2=False)

        #log_prob, _, alpha, alpha_f, alpha_b, _ = model(r1, r2, r3, r4, x5, x6, x1, o2, o3, qmask, umask)

        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

        if train:
            total_loss = loss
            total_loss.backward()
            if args.tensorboard:
                #print(type(model.named_parameters()))
                for param in model.named_parameters():
                    # print(param)
                    # print(type(param))
                    # print(param[0])
                    # print(param[1])
                    # print(type(param[0]))
                    # print(type(param[1]))
                    # print(param[1].grad)
                    # print(type(param[1].grad))
                    # print(param[1][0])
                    writer.add_histogram(param[0], param[1], epoch) #param[1].grad
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_sense_loss = round(np.sum(losses_sense)/np.sum(masks), 4)

    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, [avg_fscore], [alphas, alphas_f, alphas_b, vids]

def main(config):
    os.chdir('/content/drive/.shortcut-targets-by-id/16kT2mHiq8R98TdrPiCaaRBk3jAmvxJuP/COSMIC/erc-training')

    args.batch_size = config['batch_size']
    args.dropout    = config['dropout']
    args.lr         = config['learning_rate']

    
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    emo_lstm = True
    if args.classify == 'emotion':
        n_classes  = 7
    elif args.classify == 'sentiment':
        n_classes  = 3
    
    global cuda
    global n_epochs   
    global batch_size 
    
    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size

    global  comet_features

    roberta_features = 1024
    comet_features = 768
    context_state = 150
    internal_state = 150
    external_state = 150
    intent_state = 150
    hidden_dim = 100
    attention_dim = 100

    emotion_state = internal_state + external_state + intent_state

    global seed
    seed = args.seed
    # seed_everything(seed)

    
    
    #model = nn.GRU(roberta_features ,hidden_dim, bidirectional=True)
    
    model = CommonsenseLSTMModel(roberta_features, comet_features, context_state, internal_state, external_state, intent_state, emotion_state, hidden_dim, attention_dim,
                                n_classes=n_classes,
                                listener_state=args.active_listener,
                                context_attention=args.attention,
                                dropout_rec=args.rec_dropout,
                                dropout=args.dropout,
                                emo_lstm=emo_lstm,
                                mode1=args.mode1,
                                norm=args.norm,
                                #residual=args.residual
                                )

    print ('MELD COSMIC Model.')
    #model2 = nn.LSTM(model.outputsize, model.hideensize, bidirectional=True)

    if cuda:
        model.cuda()

    if args.classify == 'emotion':
        if args.class_weight:
            if args.mu > 0:
                loss_weights = torch.FloatTensor(create_class_weight(args.mu))
            else:   
                loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696, 
                0.84847735, 5.42461417, 1.21859721])
            loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            loss_function = MaskedNLLLoss()
            
    else:
        loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    
    if args.classify == 'emotion':
        lf = open('logs/cosmic_meld_emotion_logs.txt', 'a')
    elif args.classify == 'sentiment':
        lf = open('logs/cosmic_meld_sentiment_logs.txt', 'a')

    train_loader, valid_loader, test_loader = get_MELD_loaders(batch_size=batch_size, 
                                                               classify=args.classify,
                                                               num_workers=0)
    train_losses, train_fscores, train_accuracies = [], [], []
    valid_losses, valid_fscores, valid_accuracies = [], [], []
    test_fscores, test_losses, test_accuracies = [], [], []
    best_loss, best_label, best_pred, best_mask = None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True)
        valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)
            
        train_losses.append(train_loss)
        train_fscores.append(train_fscore) 
        train_accuracies.append(train_acc) 
        
        valid_losses.append(valid_loss)
        valid_fscores.append(valid_fscore)
        valid_accuracies.append(valid_acc)
        
        test_losses.append(test_loss)
        test_fscores.append(test_fscore) 
        test_accuracies.append(test_acc)

        #tune.report(score = valid_fscore[0])
        
        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc/test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc/train_loss, e)
        
        x = 'epoch: {}, train_loss: {}, acc: {}, fscore: {}, valid_loss: {}, acc: {}, fscore: {}, test_loss: {}, acc: {}, fscore: {}, time: {} sec'.format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2))
        
        print (x)
        #lf.write(x + '\n')

    if args.tensorboard:
        writer.close()
        
    valid_fscores = np.array(valid_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()

    valid_losses.sort()
    
    valid_list, test_list = [], []
    
    valid_list.append(valid_fscores)
    test_list.append(test_fscores)
    
    #plot_training(n_epochs, test_fscore, show)
    #if show is True:
    #    break
    
    
    print(f"alla valid_fscores: {valid_fscores}")
    print(f"alla valid_losses: {valid_losses}")
    print(f"alla test_fscores: {test_fscores}")
    print(f"alla test_losses: {test_losses}")
    
    
    #Test loss och F1 baserat på högsta valid F1:
    
    score1 = test_losses[np.argmax(valid_fscores[0])]       #Test loss för epochen där valid F1 är som högst
    score2 = test_fscores[0][np.argmax(valid_fscores[0])]   #Test F1 för epochen där valid F1 är som högst
    
    #Test loss och F1 baserat på lägsta valid loss:
    
    #score1 = test_losses[np.argmin(valid_losses)]           #Test loss för epochen där valid loss är som lägst
    #score2 = test_fscores[0][np.argmin(valid_losses)]       #Test F1 för epochen där valid loss är som lägst
    
    #Test F1 för lägsta valid loss och högsta valid F1 (Vad COSMIC har)
    #score1 = test_fscores[0][np.argmin(valid_losses)]       #Test F1 för epochen där valid loss är som lägst
    #score2 = test_fscores[0][np.argmax(valid_fscores[0])]   #Test F1 för epochen där valid F1 är som högst
    
    #Test loss för lägsta valid loss och Test F1 för högsta valid F1:
    #score1 = test_losses[np.argmin(valid_losses)]       #Test loss för epochen där valid loss är som lägst
    #score2 = test_fscores[0][np.argmax(valid_fscores[0])]   #Test F1 för epochen där valid F1 är som högst
    

    scores = [score1, score2]
    scores = [str(item) for item in scores]
    
    print ('Test Scores: Weighted F1')
    print('@Best Valid Loss: {}'.format(score1))
    print('@Best Valid F1: {}'.format(score2))

    if args.classify == 'emotion':
        rf = open('results/cosmic_meld_emotion_results.txt', 'a')
    elif args.classify == 'sentiment':
        rf = open('results/cosmic_meld_sentiment_results.txt', 'a')
    
    rf.write('\t'.join(scores) + '\t' + str(args) + '\n')
    rf.close()

    return {"loss": score1, "fscore": score2}

def analyze():
    analysis = tune.run(
        main,
        config={
            "dropout":          tune.grid_search([0.35]),
            "batch_size":       tune.grid_search([128]),
            "learning_rate":    tune.grid_search([0.0006969]),
        }, 
        resources_per_trial={
            'gpu': 1, 
            #'cpu': 4,
        },
        num_samples=20,
    )
        
def brute():
    nmbr_runs = 2
    nmbr_epochs = [10, 20, 30]
    dropouts = [0.1, 0.5, 0.9]
    batch_sizes = [8, 64]
    
    loss = {}
    f1_score = {}
    config = {}
    
    score1 = -1
    score2 = -2
    
    for epoch in nmbr_epochs:
      loss[epoch] = {} 
      f1_score[epoch] = {}
      config["epochs"] = epoch
      for dropout in dropouts:
        loss[epoch][dropout] = {} 
        f1_score[epoch][dropout] = {}
        config["dropout"] = dropout
        for batch_size in batch_sizes:    
            loss[epoch][dropout][batch_size] = {} 
            f1_score[epoch][dropout][batch_size] = {}
            config["batch_size"] = batch_size
            for i in range(nmbr_runs):
              print(f"--------- Epochs:{epoch}, Dropout:{dropout}, Batch_size:{batch_size}, Runs:{i} ---------------")
              score1, score2 = main(config)  
              loss[epoch][dropout][batch_size][i+1] = score1 
              f1_score[epoch][dropout][batch_size][i+1] = score2
    print(loss)
    print(f1_score)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.5, metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=8, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=40, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    parser.add_argument('--attention', default='simple', help='Attention type in context GRU')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--mode1', type=int, default=2, help='Roberta features to use')
    parser.add_argument('--seed', type=int, default=100, metavar='seed', help='seed')
    parser.add_argument('--norm', type=int, default=0, help='normalization strategy')
    parser.add_argument('--mu', type=float, default=0, help='class_weight_mu')
    parser.add_argument('--classify', default='emotion')
    #parser.add_argument('--residual', action='store_true', default=False, help='use residual connection')

    args = parser.parse_args()
    
    analyze()
    #main()