<!-- - change the saving of the feature_sharding  -->
- clean load_dataset_auto in activations_store
- fix this:                 logger.info("Loading dataset...")
                self.raw_ds = load_dataset_auto(cfg.dataset_path, split="train[:250000]")
                logger.info(f"Loaded dataset")
, should be a parameter, or in the split name 
- double check set_norm_scaling_factor_if_needed
- test and fix _synchronize_feature_sharding_gradients (should not be there, could be a check)



- **2 was added to the computation of the feature norms, not sure why. 