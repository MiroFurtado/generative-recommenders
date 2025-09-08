#!/usr/bin/env python3
"""
Example script demonstrating how to use wandb with generative-recommenders.

This script shows how to:
1. Initialize wandb logging
2. Run training with wandb enabled
3. Log custom metrics and configurations

Usage:
    python wandb_example.py
"""

import os
import sys
import logging

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from generative_recommenders.dlrm_v3.utils import MetricsLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Example of how to use wandb integration with generative-recommenders.
    """
    
    logger.info("Wandb Integration Example")
    logger.info("=" * 50)
    
    # Check if wandb is available
    try:
        import wandb
        logger.info("✓ wandb is available")
    except ImportError:
        logger.error("✗ wandb is not installed. Install with: pip install wandb")
        return
    
    # Example 1: Basic wandb configuration
    logger.info("\n1. Basic wandb configuration:")
    logger.info("   - Project: generative-recommenders-example")
    logger.info("   - Tags: ['example', 'tutorial']")
    
    # Example configuration (you would normally get this from your gin file)
    from generative_recommenders.modules.multitask_module import TaskConfig, MultitaskTaskType
    import torch
    
    # Mock task configurations
    mock_configs = [
        TaskConfig(
            task_name="classification_task",
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="regression_task", 
            task_type=MultitaskTaskType.REGRESSION,
        ),
    ]
    
    # Initialize MetricsLogger with wandb
    device = torch.device("cpu")
    
    try:
        metrics_logger = MetricsLogger(
            multitask_configs=mock_configs,
            batch_size=32,
            window_size=100,
            device=device,
            rank=0,
            # Wandb configuration
            use_wandb=True,
            wandb_project="generative-recommenders-example",
            wandb_run_name="tutorial-run",
            wandb_tags=["example", "tutorial"],
            wandb_config={
                "learning_rate": 0.001,
                "optimizer": "Adam",
                "architecture": "DLRM_v3",
            }
        )
        
        logger.info("✓ MetricsLogger initialized with wandb support")
        
        # Example 2: Log some fake metrics
        logger.info("\n2. Logging example metrics...")
        
        # Simulate some training metrics
        fake_predictions = torch.randn(32, 2)  # batch_size=32, 2 tasks
        fake_labels = torch.randint(0, 2, (32, 2)).float()
        fake_weights = torch.ones(32, 2)
        
        # Update metrics (this would normally happen in your training loop)
        for step in range(5):
            metrics_logger.update(
                predictions=fake_predictions,
                labels=fake_labels,
                weights=fake_weights,
                mode="train"
            )
            
            # Log metrics every step
            fake_losses = {
                "mse_loss": torch.tensor(0.5 - step * 0.1),
                "bce_loss": torch.tensor(0.3 - step * 0.05),
            }
            
            computed_metrics = metrics_logger.compute_and_log(
                mode="train",
                additional_logs={"losses": fake_losses}
            )
            
            logger.info(f"   Step {step + 1}: {len(computed_metrics)} metrics logged")
        
        logger.info("✓ Example metrics logged successfully")
        
        # Example 3: Finish the wandb run
        logger.info("\n3. Finishing wandb run...")
        metrics_logger.finish_wandb()
        logger.info("✓ wandb run finished")
        
    except Exception as e:
        logger.error(f"✗ Error during wandb example: {e}")
        return
    
    logger.info("\n" + "=" * 50)
    logger.info("Wandb integration example completed successfully!")
    logger.info("Check your wandb dashboard to see the logged metrics.")
    logger.info("Dashboard: https://wandb.ai/")


if __name__ == "__main__":
    main()
