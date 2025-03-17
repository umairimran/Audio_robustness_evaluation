import argparse
import sys
import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoFeatureExtractor
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from utils import compute_eer
from models import Hubert, Wav2Vec2, Wav2Vec2BERT, Whisper, CLAP
from data_utils import get_wavefake_loader, get_in_the_wild_loader, get_libri_loader, get_custom_loader
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_validation(model, feature_extractor, data_loader, sr):

    outputs_list = []
    labels_list = []
    train_loss = []
    num_total = 0

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, name in tqdm(data_loader, desc="Evaluating"):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.numpy()
            inputs = feature_extractor(batch_x, sampling_rate=sr, return_attention_mask=True,padding_value=0, return_tensors="pt").to(device)
            batch_y = batch_y.to(device)
            inputs['labels'] = batch_y
            outputs = model(**inputs)
            train_loss.append(outputs.loss.item())
            batch_probs = outputs.logits.softmax(dim=-1)
            batch_label = batch_y.detach().to('cpu').numpy().tolist()
            outputs_list.extend(batch_probs[:, 1].tolist())
            labels_list.extend(batch_label)

        auroc = roc_auc_score(labels_list, outputs_list)
        eer = compute_eer(np.array(labels_list), np.array(outputs_list))
        preds = (np.array(outputs_list) > eer[1]).astype(int)
        acc = np.mean(np.array(labels_list) == np.array(preds))
        prec = precision_score(labels_list, preds)
        recall = recall_score(labels_list, preds)
        f1 = f1_score(labels_list, preds)
        print(f'Validation Accuracy: {acc} \t F1: {f1} \t Precision: {prec} \t Recall: {recall}, AUROC: {auroc} \t EER: {eer}')

        return acc, auroc, eer
def main(args):

    if args.model == 'wave2vec2bert':
        model_name = "facebook/w2v-bert-2.0"
        model = Wav2Vec2BERT(model_name)
        sampling_rate = 16000
    elif args.model == 'wave2vec2':
        model_name = "facebook/wav2vec2-large-960h"
        model = Wav2Vec2(model_name)
        sampling_rate = 16000
    elif args.model == 'hubert':
        model_name = "facebook/hubert-large-ls960-ft"
        model = Hubert(model_name)
        sampling_rate = 16000
    elif args.model == 'whisper-small':
        model_name = f"openai/{args.model}"
        model = Whisper(model_name)
        sampling_rate = 16000
    elif args.model == 'clap':
        model_name = "laion/clap-htsat-unfused"
        model = CLAP(model_name)
        sampling_rate = 48000
    else:
        raise ValueError(f"Model {args.model} not supported")
    model = model.to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    wave_trn_loader, wave_dev_loader, wave_eval_loader = get_wavefake_loader(args.database_dir, seed=args.seed, batch_size=args.batch_size)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.eval:
        ckpt = torch.load(args.eval_ckpt, map_location=device)
        model.load_state_dict(ckpt)

        print("Start evaluating Wavefake")
        if args.pert_method == "gaussian_noise":
            pert_levels = [5,10,20,30,40]
        elif args.pert_method == "background_noise":
            pert_levels = [5, 10, 20, 30, 40]
        elif args.pert_method == "smooth":
            pert_levels = [6, 10, 14, 18, 22, 26]
        elif args.pert_method == "echo":
            pert_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        elif args.pert_method == "high_pass":
            pert_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        elif args.pert_method == "low_pass":
            pert_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        elif args.pert_method == "pitch_shift":
            pert_levels = [-2, -1, -0.5, 0.5, 1, 2]
        elif args.pert_method == "time_stretch":
            pert_levels = [0.7, 0.9, 1.1, 1.3, 1.5]
        elif args.pert_method == "quantization":
            pert_levels = [2 ** 2, 2 ** 4, 2 ** 6, 2 ** 8, 2 ** 10]
        elif args.pert_method == "opus":
            pert_levels = [1, 2, 4, 8 ,16, 31]
        elif args.pert_method == "mp3":
            pert_levels = [8, 16, 24, 32, 40]
        elif args.pert_method == "encodec":
            pert_levels = [1.5, 3.0, 6.0, 12.0, 24.0]
        else:
            raise ValueError(f"Not support {args.pert_method}")

        for pert_level in pert_levels:
            wave_trn_loader, wave_dev_loader, wave_eval_loader = get_wavefake_loader(args.database_dir, seed=1234, batch_size=128,
                                                                                 pert_method=args.pert_method, pert_level=pert_level)
            print(f'{args.pert_method}: {wave_eval_loader.dataset.pert_level}')
            eval_acc, eval_auroc, eval_eer = run_validation(model, feature_extractor, wave_eval_loader, sampling_rate)



        sys.exit(0)

    outputs_list = []
    labels_list = []
    train_loss = []
    auroc_list = []
    acc_list = []
    err_list = []
    num_total = 0

    model.train()
    for epoch in range(args.epochs):
        steps = 0
        for batch_x, batch_y, name in tqdm(wave_trn_loader, desc="Finetuning"):
            batch_size = batch_x.size(0)
            num_total += batch_size
            steps += 1
            batch_x = batch_x.numpy()
            inputs = feature_extractor(batch_x, sampling_rate=sampling_rate, return_attention_mask=True, padding_value=0, return_tensors="pt").to(device)
            batch_y = batch_y.to(device)
            inputs['labels'] = batch_y
            outputs = model(**inputs)
            train_loss.append(outputs.loss.item())
            batch_probs = outputs.logits.softmax(dim=-1)
            batch_label = batch_y.detach().to('cpu').numpy().tolist()
            outputs_list.extend(batch_probs[:, 1].tolist())
            labels_list.extend(batch_label)
            optim.zero_grad()
            outputs.loss.backward()
            optim.step()

            if steps % args.eval_steps == 0:
                torch.save(model.state_dict(),
                           f'{args.output_dir}/{args.model}_epoch_{args.epochs}_lr_{args.lr}_bs_{args.batch_size}.pth')
                eval_acc, eval_auroc, eval_eer = run_validation(model, feature_extractor, wave_dev_loader,sr=sampling_rate)
                auroc_list.append(eval_auroc)
                acc_list.append(eval_acc)
                err_list.append(eval_eer[0])
                model.train()

        auroc = roc_auc_score(labels_list, outputs_list)
        eer = compute_eer(np.array(labels_list), np.array(outputs_list))
        print(f'Training epoch: {epoch} \t AUROC: {auroc} \t EER: {eer}')

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{args.output_dir}/{args.model}_epoch_{args.epoch}_lr_{args.lr}_bs_{args.batch_size}.pth')

    print('finished')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SONAR system")
    parser.add_argument("--output_dir",
                        default="./ckpt/",
                        type=str,
                        help="output directory of model checkpoints")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 123)")
    parser.add_argument(
        "--eval",
        action="store_true",
        default=True,
        help="when this flag is given, evaluates given model and exit")

    parser.add_argument("--dataset",type=str,default='Wavefake',help="dataset")
    parser.add_argument("--database_dir", type=str,default='./data/wavefake/',help="path to dataset")
    parser.add_argument("--resume", type=bool, default=False, help="resume from checkpoint")
    parser.add_argument("--model", type=str, default='hubert')
    parser.add_argument("--epochs", type=int, default=3, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="weight decay")
    parser.add_argument("--eval_steps", type=int, default=200, help="weight decay")
    parser.add_argument("--eval_ckpt", type=str, default='./ckpt/hubert.pth',help="path to checkpoints")
    parser.add_argument("--pert_method", type=str, default="pitch_shift",help="Specify perturbation to be added on audio for evaluation")

    main(parser.parse_args())