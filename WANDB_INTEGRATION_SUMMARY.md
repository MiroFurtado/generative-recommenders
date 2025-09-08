# Wandb Integration Summary

This document summarizes all the changes made to add Weights & Biases (wandb) support to the generative-recommenders repository.

## Files Modified

### 1. Dependencies
- **requirements.txt**: Added `wandb>=0.16.0`
- **setup.py**: Added `wandb>=0.16.0` to install_requires

### 2. Core Integration
- **generative_recommenders/dlrm_v3/utils.py**:
  - Added wandb import with availability check
  - Extended `MetricsLogger` class with wandb parameters:
    - `use_wandb`: Enable/disable wandb logging
    - `wandb_project`: Project name
    - `wandb_entity`: Entity (username/team)
    - `wandb_run_name`: Custom run name
    - `wandb_tags`: List of tags
    - `wandb_config`: Additional config to log
  - Modified `compute_and_log()` method to log to wandb
  - Added `finish_wandb()` method for cleanup
  - Added multi-GPU support (only rank 0 logs to avoid duplicates)

### 3. Training Integration
- **generative_recommenders/dlrm_v3/train/utils.py**:
  - Fixed logging condition in `train_loop` (changed `!=` to `==`)

- **generative_recommenders/dlrm_v3/train/train_ranker.py**:
  - Added wandb cleanup in exception handling and finally block
  - Added `debug-wandb` to SUPPORTED_CONFIGS

### 4. Configuration Files
- **generative_recommenders/dlrm_v3/train/gin/debug.gin**: 
  - Added wandb configuration parameters (disabled by default)

- **generative_recommenders/dlrm_v3/train/gin/debug_wandb.gin**: 
  - New configuration file with wandb enabled for testing

- **generative_recommenders/dlrm_v3/train/gin/movielens_1m.gin**: 
  - Added wandb configuration parameters (disabled by default)

### 5. Documentation
- **WANDB_README.md**: Comprehensive documentation including:
  - Installation instructions
  - Configuration options
  - Usage examples
  - Best practices
  - Troubleshooting guide
  - API reference

- **README.md**: Added wandb section with quick start guide

- **wandb_example.py**: Example script demonstrating wandb usage

## Key Features

### 1. **Seamless Integration**
- Works alongside existing TensorBoard logging
- No breaking changes to existing code
- Graceful fallback if wandb is not available

### 2. **Comprehensive Logging**
- Training and evaluation metrics (AUC, NE, MSE, MAE)
- Loss tracking
- Model configuration and hyperparameters
- Step and epoch tracking

### 3. **Multi-GPU Support**
- Only rank 0 logs to wandb to avoid duplicates
- Automatic detection and handling

### 4. **Robust Error Handling**
- Continues training if wandb fails
- Detailed warning messages
- Safe cleanup on exit

### 5. **Flexible Configuration**
- Gin-based configuration
- Environment variable support
- Optional parameters with sensible defaults

## Usage Examples

### Basic Usage
```gin
# In your .gin file
MetricsLogger.use_wandb = True
MetricsLogger.wandb_project = "my-project"
```

### Advanced Configuration
```gin
MetricsLogger.use_wandb = True
MetricsLogger.wandb_project = "generative-recommenders"
MetricsLogger.wandb_entity = "my-team"
MetricsLogger.wandb_run_name = "experiment-v1"
MetricsLogger.wandb_tags = ["dlrm_v3", "baseline"]
```

### Command Line
```bash
# Use the wandb-enabled debug configuration
python train_ranker.py --dataset debug-wandb --mode train

# Use any existing configuration with wandb enabled via gin file
python train_ranker.py --dataset movielens-1m --mode train-eval
```

## Testing

Test the integration using:

1. **Debug mode**: `python train_ranker.py --dataset debug-wandb --mode train`
2. **Example script**: `python wandb_example.py`
3. **Check logs**: Look for wandb initialization and logging messages

## Benefits

1. **Experiment Tracking**: Automatically track all experiments with detailed metrics
2. **Comparison**: Easy comparison between different runs and hyperparameters  
3. **Collaboration**: Share experiments with team members
4. **Visualization**: Rich visualizations and dashboards
5. **Reproducibility**: Complete logging of configurations and results

## Future Enhancements

Potential future improvements:
1. **Model artifact logging**: Save and version trained models
2. **Code logging**: Track code changes with each experiment
3. **Hyperparameter sweeps**: Integration with wandb sweeps
4. **Custom charts**: Domain-specific visualizations
5. **Alerts**: Notifications for experiment completion or failures

## Backward Compatibility

All changes are backward compatible:
- Existing configurations continue to work unchanged
- wandb logging is disabled by default
- No required parameters added to existing functions
- Graceful degradation if wandb is not installed
