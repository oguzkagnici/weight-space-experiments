"""Fixed constants of the weight-space-programming GA harness (LFM3-1.7B x MMLU-Pro, direct answer, CPU)."""
import os
ROOT = os.environ.get("WSP_ROOT", "/home/user/weight-space-experiments")
MODEL_DIR = "/home/user/models/LFM3-1.7B"
WEIGHTS_FILE = os.path.join(MODEL_DIR, "model.safetensors")
DATA = os.path.join(ROOT, "data")
RUNS = os.path.join(ROOT, "runs")
QUEUE = os.path.join(ROOT, "queue")
CACHE = os.path.join(ROOT, "cache")

N_LAYERS = 24
LAYER_TYPES = ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
               "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention",
               "conv", "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
               "full_attention", "conv"]
LETTERS = "ABCDEFGHIJ"

# Out of scope (refused by save_edit), as in the paper: input embedding / tied output layer, final norm.
# In LFM3 the final norm is called `model.embedding_norm` (it is applied after the last layer).
FORBIDDEN = {"model.embed_tokens.weight", "model.embedding_norm.weight", "lm_head.weight"}

# Forward-pass tool limits (CPU-scaled versions of the paper's 512 generations / 1,024 texts)
MAX_SCORE_ITEMS = 512          # questions per model for "score"; with "both" at most 256 per model
MAX_SCORE_ITEMS_BOTH = 256
MAX_READ_TEXTS = 1024
MAX_READ_TOKENS = 200_000      # total tokens of a teacher-forced "read" pass
MAX_TEXT_TOKENS = 4096
MAX_OUTPUT_BYTES = 3 * 2**30   # .npy output per pass
MAX_ATTN_TEXTS = 4             # attention probabilities: at most 4 texts of at most 1,024 tokens
