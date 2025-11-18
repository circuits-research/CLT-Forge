# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Environment variable `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` for handling long context models
- VLLM configuration with 8192 max_model_len for longer prompts
- Cutting context_window trunk at the end of each document, should be only multilingual (yes for now)? 
- VLLM configuration with 3k max_model_len for longer prompts (too slow otherwise)

### To do 
- clustering
- intervention 
- scoring 
- general testing and cleaning
