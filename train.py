import wandb
import math
import random
import time

run_count = 1
step_count = 50

for run in range(run_count):
  run = wandb.init(
            entity="launch-test",
            project="wandb-launch-sweeps",
            config={
                "learning_rate": 0.01 * random.random(),
                "batch_size": 128,
                "momentum": 0.1 * random.random(),
                "dropout": 0.4 * random.random(),
                "architecture": "CNN",
                "dataset": "mountain-view",
            })
  displacement1 = random.random()
  displacement2 = random.random()
  for step in range(step_count):
    wandb.log({
        "acc": .1 + 0.04 * (math.log(1 + step + random.random()) + random.random() * run.config.learning_rate + random.random() + displacement1 + random.random() * run.config.momentum),
        "val_acc": .1 + 0.04 * (math.log(1 + step + random.random()) + random.random() * run.config.learning_rate - random.random() + displacement1),
        "loss": .1 + 0.04 * (4 - math.log(1 + step + random.random()) + random.random() * run.config.momentum + random.random() + displacement2),
        "val_loss": .1 + 0.04 * (5 - math.log(1 + step + random.random()) + random.random() * run.config.learning_rate - random.random() + displacement2),
    })
    time.sleep(2)
# wandb.run.log_code()
wandb.finish()
