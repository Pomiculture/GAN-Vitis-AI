#!/bin/bash

###############################################################################################################

# Evaluate the quality of the GAN's generator.

###############################################################################################################

# Run the GAN's classifier over the output data (Accuracy : fool the discriminator)
source ./workflow/eval/run_discriminator.sh

# Evaluate GAN's generator (FID score : generated vs real images)
source ./workflow/eval/eval_gan.sh

# Compare to gold (SSIM score : similarity between generated and reference images)
source ./workflow/eval/compare_to_gold.sh
