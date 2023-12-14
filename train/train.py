import os
import argparse
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adafactor
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
import wandb
from tqdm import tqdm
import random
import numpy as np

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_acc(embeddings):
    batch_size = embeddings.shape[0] // 2
    with torch.no_grad():
        scores = torch.matmul(embeddings[:batch_size].detach(), embeddings[batch_size:].T).cpu().numpy()
    a1 = (scores.argmax(1) == np.arange(batch_size)).mean()
    a2 = (scores.argmax(0) == np.arange(batch_size)).mean()
    return (a1 + a2) / 2

def cleanup():
    torch.cuda.empty_cache()

def train(args):
    accelerator = Accelerator()
    device = accelerator.device

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if accelerator.is_distributed:
        accelerator.random_seed = args.seed

    # Initialize wandb
    wandb.init(project=args.project_name, config=args)
    wandb.config.update(args)

    # Load and preprocess data
    df = pd.read_csv(args.data_path, delimiter=';')
    df = df[['specializations.names', 'description']]
    all_pairs = list(df.itertuples(index=False, name=None))

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model = model.to(device)
    model.train()

    # Optimizer and loss function
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False, relative_step=False, lr=args.learning_rate, clip_threshold=1.0
    )
    loss_fn = CrossEntropyLoss()

    # Training loop
    num_steps = args.epochs * len(df) // args.batch_size
    tq = tqdm(total=num_steps, desc="Training")
    for step in range(num_steps):
        df = df.sample(frac=1).reset_index(drop=True)
        now = df.drop_duplicates(subset='specializations.names', keep='first')
        now = now.drop_duplicates(subset='description', keep='first')
        all_pairs = list(now.itertuples(index=False, name=None))
        t, q = [list(p) for p in zip(*random.choices(all_pairs, k=args.batch_size))]

        try:
            with accelerator.prepare([t, q]) as prepared_data:
                encoded_question = tokenizer(
                    prepared_data[0] + prepared_data[1],
                    padding=True,
                    truncation=True,
                    return_tensors='pt').to(device)
                model_output = model(**encoded_question)
                embeddings = mean_pooling(model_output, encoded_question['attention_mask'])

                all_scores = torch.matmul(embeddings[:args.batch_size].detach(),
                                          embeddings[args.batch_size:].T) - torch.eye(args.batch_size,
                                                                                         device=device) * args.margin
                loss = loss_fn(all_scores, torch.arange(args.batch_size, device=device)) + loss_fn(
                    all_scores.T, torch.arange(args.batch_size, device=device))

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                wandb.log({"Loss": loss.item()})

                if step % args.checkpoint_steps == 0:
                    torch.save(model.state_dict(), args.checkpoint_path)

                tq.set_postfix({"Loss": loss.item()})
                tq.update()

        except RuntimeError:
            optimizer.zero_grad(set_to_none=True)
            cleanup()
            continue

    tq.close()

    # Save final model
    torch.save(model.state_dict(), args.output_path)

    # Clean up
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paraphrase Model Training")
    parser.add_argument("--data_path", type=str, help="Path to the data file")
    parser.add_argument("--model_name", type=str, default="paraphrase-multilingual-mpnet-base-v2",
                        help="Name of the pre-trained model")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--margin", type=float, default=0.3, help="Margin for pairwise similarity")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--project_name", type=str, default="paraphrase-training",
                        help="Name of the wandb project")
    parser.add_argument("--checkpoint_path", type=str, default="paraphrase_checkpoint.pt",
                        help="Path to save model checkpoints")
    parser.add_argument("--checkpoint_steps", type=int, default=1000,
                        help="Steps between saving model checkpoints")
    parser.add_argument("--output_path", type=str, default="paraphrase_model.pt", help="Path to save the final model")

    args = parser.parse_args()

    train(args)
