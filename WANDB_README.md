# Weights & Biases (wandb) Integration

This document describes how to use Weights & Biases (wandb) logging with the generative-recommenders library.

## Overview

The wandb integration provides comprehensive experiment tracking and logging capabilities, including:
- Automatic logging of training and evaluation metrics
- Loss tracking
- Hyperparameter tracking
- Model configuration logging
- Training step visualization

## Installation

Make sure wandb is installed:

```bash
pip install wandb>=0.16.0
```

Or install the package with wandb support:

```bash
pip install -e .
```

## Setup

1. **Login to wandb** (if not already done):
```bash
wandb login
```

2. **Configure your gin file** to enable wandb logging by setting the following parameters in your `.gin` configuration file:

```gin
# Enable wandb logging
MetricsLogger.use_wandb = True
MetricsLogger.wandb_project = "your-project-name"
MetricsLogger.wandb_entity = "your-wandb-entity"  # Optional
MetricsLogger.wandb_run_name = "experiment-name"  # Optional
MetricsLogger.wandb_tags = ["tag1", "tag2"]  # Optional
```

## Configuration Options

The following wandb configuration options are available in the `MetricsLogger` class:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_wandb` | `bool` | `False` | Enable/disable wandb logging |
| `wandb_project` | `str` | `"generative-recommenders"` | wandb project name |
| `wandb_entity` | `str` | `None` | wandb entity (username/team) |
| `wandb_run_name` | `str` | `None` | Custom run name |
| `wandb_tags` | `List[str]` | `None` | List of tags for the run |
| `wandb_config` | `Dict` | `None` | Additional config to log |

## Example Configurations

### 1. Debug Configuration with wandb

Use the provided `debug_wandb.gin` configuration:

```bash
python train_ranker.py --dataset debug-wandb --mode train
```

### 2. Custom Configuration

Create your own gin file with wandb settings:

```gin
# my_experiment.gin
batch_size = 32
dataset = "movielens-1m"

# ... other configurations ...

# wandb configuration
MetricsLogger.use_wandb = True
MetricsLogger.wandb_project = "movie-recommendations"
MetricsLogger.wandb_entity = "my-team"
MetricsLogger.wandb_run_name = "experiment-v1"
MetricsLogger.wandb_tags = ["movielens", "dlrm_v3", "baseline"]
```

### 3. Environment-based Configuration

You can also control wandb through environment variables:

```bash
export WANDB_PROJECT="generative-recommenders"
export WANDB_ENTITY="my-entity"
export WANDB_RUN_NAME="my-experiment"

python train_ranker.py --dataset debug --mode train
```

## Logged Metrics

The wandb integration automatically logs the following metrics:

### Training Metrics
- **Classification metrics**: AUC, Normalized Entropy (NE)
- **Regression metrics**: MSE, MAE
- **Losses**: All auxiliary losses from the model
- **Step information**: Training step and epoch information

### Evaluation Metrics
- Same metric types as training, but with `eval_` prefix
- Evaluation step tracking

### Model Configuration
- Batch size
- Window size
- Number of tasks
- Task names and types
- Multitask configurations

## Multi-GPU Training

When using distributed training with multiple GPUs:
- Only rank 0 will log to wandb to avoid duplicate logs
- All other ranks will have wandb logging disabled automatically
- This is handled automatically by the `MetricsLogger` class

## Logging Frequency

Control how often metrics are logged using the `metric_log_frequency` parameter in your gin configuration:

```gin
train_loop.metric_log_frequency = 10  # Log every 10 batches
```

## Integration with TensorBoard

wandb logging works alongside existing TensorBoard logging:
- Both loggers can be enabled simultaneously
- Set `tensorboard_log_path` to enable TensorBoard
- Set `use_wandb = True` to enable wandb

```gin
# Enable both logging systems
MetricsLogger.tensorboard_log_path = "/tmp/tensorboard_log_path.log"
MetricsLogger.use_wandb = True
MetricsLogger.wandb_project = "my-project"
```

## Error Handling

The wandb integration includes robust error handling:
- If wandb is not installed, the system will warn and continue without wandb logging
- If wandb initialization fails, training will continue with local logging only
- Failed wandb log calls are caught and logged as warnings

## Best Practices

1. **Use descriptive project names**: Make it easy to find your experiments
2. **Tag your experiments**: Use tags to organize related experiments
3. **Use meaningful run names**: Include key hyperparameters or experiment variants
4. **Set wandb entity**: Especially important for team projects

## Example Usage

```bash
# Simple training with wandb
python train_ranker.py --dataset debug-wandb --mode train

# Training with custom configuration
python train_ranker.py --dataset movielens-1m --mode train-eval
```

## Troubleshooting

### Common Issues

1. **"wandb not found" error**: Install wandb with `pip install wandb`
2. **Login issues**: Run `wandb login` and follow the prompts
3. **Permission errors**: Check your wandb entity and project permissions
4. **Multiple logging**: Only rank 0 logs to wandb in distributed training

### Debug Mode

Enable debug logging to see detailed wandb integration information:

```bash
export WANDB_DEBUG=true
python train_ranker.py --dataset debug-wandb --mode train
```

## API Reference

For more advanced usage, you can directly use the wandb integration in the `MetricsLogger` class:

```python
from generative_recommenders.dlrm_v3.utils import MetricsLogger

# Initialize with wandb
metrics = MetricsLogger(
    multitask_configs=configs,
    batch_size=32,
    window_size=1000,
    device=device,
    rank=0,
    use_wandb=True,
    wandb_project="my-project",
    wandb_run_name="my-experiment",
)

# Log metrics (automatically logs to wandb if enabled)
metrics.compute_and_log(mode="train", additional_logs={"losses": losses})

# Finish wandb run
metrics.finish_wandb()
```
