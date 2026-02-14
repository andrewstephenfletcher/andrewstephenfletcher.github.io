from datasets import load_dataset
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_dataset_ALL_CAPS(n_samples=1000):
    """Generate dataset with normal and ALL CAPS text pairs."""
    subjects = ["The cat", "A dog", "The car", "My friend", "The bird",
                "A plane", "The code", "This model", "The robot", "A machine"]
    verbs = ["jumps over", "runs to", "flies above", "looks at", "sits on",
             "moves to", "likes", "sees", "devours", "builds"]
    things = ["the fence", "the hill", "the cloud", "the screen", "the table",
              "the city", "the pizza", "the park", "the book", "the house"]

    data = []
    labels = []

    for subject in subjects:
        for verb in verbs:
            for thing in things:
                sequence = f"{subject} {verb} {thing}"
                data.append(sequence)
                labels.append(0)
                data.append(sequence.upper())
                labels.append(1)

    return data[:n_samples], labels[:n_samples]


def generate_dataset_HTML(n_samples=50):
    """Generate dataset with normal and HTML formatted text pairs."""
    alpaca = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    html_alpaca = load_dataset("ttbui/html_alpaca", split="train", streaming=True)

    data = []
    labels = []

    for i, (normal_ex, html_ex) in enumerate(zip(alpaca, html_alpaca)):
        if i >= n_samples:
            break

        data.append(normal_ex['output'])
        labels.append(0)
        data.append(html_ex['output'])
        labels.append(1)

    return data, labels


def generate_dataset_HARMFUL(n_samples=1000):
    """Generate dataset with normal and harmful instruction pairs."""
    alpaca = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    advbench = load_dataset("S3IC/advbench", split="train", streaming=True)

    data = []
    labels = []

    for i, (normal_ex, harmful_ex) in enumerate(zip(alpaca, advbench)):
        if i >= n_samples:
            break

        data.append(normal_ex['instruction'])
        labels.append(0)
        data.append(harmful_ex['goal'])
        labels.append(1)

    return data, labels


class LinearProbe(nn.Module):
    """Simple linear probe for binary classification."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.linear(x)
        probs = self.sigmoid(logits)
        return probs


def extract_activations(model, tokenizer, dataset, layer_idx=6, max_length=50):
    """Extract hidden state activations from a specific layer."""
    inputs = tokenizer(
        dataset,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length
    )

    print(f"  Tokenized {inputs['input_ids'].shape[0]} sequences "
          f"with max length {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )

    # Extract activations from specified layer (mean pooling over sequence)
    activations = outputs.hidden_states[layer_idx].mean(dim=1).float()

    print(f"  Extracted activations with shape: {activations.shape}")
    return activations


def train_probe(X_train, y_train, X_test, y_test, epochs=500, lr=0.001):
    """Train a linear probe and return training history."""
    probe = LinearProbe(hidden_dim=X_train.shape[1])
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)

    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    for epoch in tqdm(range(epochs), desc="Training probe"):
        # Forward pass
        train_prediction = probe(X_train).squeeze()
        test_prediction = probe(X_test).squeeze()

        # Calculate loss
        train_loss = nn.BCELoss()(train_prediction, y_train.float())
        test_loss = nn.BCELoss()(test_prediction, y_test.float())

        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Calculate accuracy
        train_predicted_classes = (train_prediction >= 0.5).long()
        train_accuracy = (train_predicted_classes == y_train).float().mean().item()

        test_predicted_classes = (test_prediction >= 0.5).long()
        test_accuracy = (test_predicted_classes == y_test).float().mean().item()

        # Store history
        train_loss_history.append(train_loss.item())
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss.item())
        test_accuracy_history.append(test_accuracy)

    return {
        'train_loss': train_loss_history,
        'train_accuracy': train_accuracy_history,
        'test_loss': test_loss_history,
        'test_accuracy': test_accuracy_history,
        'probe': probe
    }


def plot_training_results(results, dataset_name, save_dir='blog/posts/residual-stream/images'):
    """Plot and save training results."""
    from pathlib import Path
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss curves
    ax1.plot(results['train_loss'], label='Training', alpha=0.8, linewidth=2)
    ax1.plot(results['test_loss'], label='Validation', alpha=0.8, linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Binary Cross-Entropy Loss', fontsize=12)
    ax1.set_title(f'Loss Curves - {dataset_name}', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot accuracy curves
    ax2.plot(results['train_accuracy'], label='Training', alpha=0.8, linewidth=2)
    ax2.plot(results['test_accuracy'], label='Validation', alpha=0.8, linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Accuracy Curves - {dataset_name}', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # Add final metrics as text
    final_train_acc = results['train_accuracy'][-1]
    final_test_acc = results['test_accuracy'][-1]
    ax2.text(0.02, 0.02,
             f'Final Training: {final_train_acc:.1%}\nFinal Validation: {final_test_acc:.1%}',
             transform=ax2.transAxes,
             fontsize=10,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot as SVG
    save_path = Path(save_dir) / f'{dataset_name.lower().replace(" ", "_")}_results.svg'
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"  Saved plot to {save_path}")

    plt.close()

    return final_train_acc, final_test_acc


def process_dataset(model, tokenizer, dataset_name, dataset, labels,
                    layer_idx=6, max_length=50, test_fraction=0.2,
                    epochs=500, save_dir='plots'):
    """Process a single dataset: extract activations, train probe, and plot results."""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")

    # Extract activations
    print("Extracting activations...")
    activations = extract_activations(model, tokenizer, dataset, layer_idx, max_length)

    # Split into train/test
    n_samples = len(dataset)
    split_idx = int(n_samples * (1 - test_fraction))

    X_train = activations[:split_idx].detach()
    y_train = labels[:split_idx]
    X_test = activations[split_idx:].detach()
    y_test = labels[split_idx:]

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train probe
    print("Training probe...")
    results = train_probe(X_train, y_train, X_test, y_test, epochs=epochs)

    # Plot and save results
    print("Generating plots...")
    train_acc, test_acc = plot_training_results(results, dataset_name, save_dir)

    print(f"\nResults for {dataset_name}:")
    print(f"  Final Training Accuracy: {train_acc:.2%}")
    print(f"  Final Validation Accuracy: {test_acc:.2%}")

    return results


def main():
    # Configuration
    MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
    N_SAMPLES = 50
    LAYER_IDX = 16
    MAX_LENGTH = 50
    EPOCHS = 200
    SAVE_DIR = "blog/posts/residual-stream/images"

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Dataset configurations
    datasets_config = [
        {
            'name': 'ALL CAPS',
            'generator': generate_dataset_ALL_CAPS,
            'n_samples': N_SAMPLES,
            'max_length': MAX_LENGTH
        },
        {
            'name': 'HTML Formatting',
            'generator': generate_dataset_HTML,
            'n_samples': N_SAMPLES // 2,  # Fewer samples for streaming datasets
            'max_length': 30  # Shorter for HTML
        },
        {
            'name': 'Harmful Instructions',
            'generator': generate_dataset_HARMFUL,
            'n_samples': N_SAMPLES // 2,  # Fewer samples for streaming datasets
            'max_length': MAX_LENGTH
        }
    ]

    # Process each dataset
    all_results = {}

    for config in datasets_config:
        print(f"\nGenerating {config['name']} dataset...")
        dataset, labels_list = config['generator'](n_samples=config['n_samples'])
        labels = torch.tensor(labels_list)

        print(f"  Generated {len(dataset)} sequences ({len(dataset)//2} pairs)")
        print(f"  Sample 0: {dataset[0][:60]}...")
        print(f"  Sample 1: {dataset[1][:60]}...")

        results = process_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name=config['name'],
            dataset=dataset,
            labels=labels,
            layer_idx=LAYER_IDX,
            max_length=config['max_length'],
            epochs=EPOCHS,
            save_dir=SAVE_DIR
        )

        all_results[config['name']] = results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, results in all_results.items():
        train_acc = results['train_accuracy'][-1]
        test_acc = results['test_accuracy'][-1]
        print(f"{name:20s} - Train: {train_acc:.2%}, Val: {test_acc:.2%}")

    print(f"\nAll plots saved to '{SAVE_DIR}/' directory")


if __name__ == "__main__":
    main()
