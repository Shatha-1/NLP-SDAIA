"""Lab 2 starter notebook-as-script.
Complete the marked sections, verify numerical equivalence, inspect parameter
accounting, causal masking, attention heads and pad-attention leakage.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from bayan.attention import MultiHeadAttention, attention

CHECKPOINT = "bert-base-multilingual-cased"

BAYAN_EXAMPLES = [
    "الحاوية ممتلئة في حي العليا ولم تُفرغ منذ 3 أيام",
    "There is a water leak at Jeddah",
]


def verify_equivalence():
    torch.manual_seed(42)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    ours = attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    ok = torch.allclose(ours, ref, atol=1e-6)
    print(f"[1] attention() matches F.scaled_dot_product_attention: {ok}")
    return ok


def inspect_weights():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    _, weights = attention(q, k, v, return_weights=True)

    print("\n[2] Attention weight matrix (1 batch, 1 head, 4x4 query/key positions):")
    print(weights[0, 0].round(decimals=3))
    row_sums = weights.sum(dim=-1)
    print(f"    Each row sums to 1 (valid probability distribution): "
          f"{torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)}")


def exercise_multihead():
    torch.manual_seed(1)
    d_model, num_heads, seq_len, batch = 16, 4, 5, 2
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model)
    out = mha(x)
    print(f"\n[3] MultiHeadAttention: input {tuple(x.shape)} -> output {tuple(out.shape)}")
    assert out.shape == (batch, seq_len, d_model), "MHA must preserve (batch, seq_len, d_model)"
    print("    Shape check passed.")


def causal_mask(seq_len: int) -> torch.Tensor:
    """1s on/below the diagonal: position i may attend only to positions <= i."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


def verify_causal_masking():
    torch.manual_seed(2)
    seq_len = 5
    q = torch.randn(1, 1, seq_len, 8)
    k = torch.randn(1, 1, seq_len, 8)
    v = torch.randn(1, 1, seq_len, 8)
    mask = causal_mask(seq_len)

    _, weights = attention(q, k, v, mask=mask, return_weights=True)
    w = weights[0, 0]
    is_lower_triangular = torch.allclose(w, torch.tril(w))

    print(f"\n[4] Causal mask -> attention matrix is lower-triangular: {is_lower_triangular}")
    print(w.round(decimals=3))
    print("    Model family: decoder-style causal (autoregressive) attention, as used in GPT.")


def _pad_attention_mass(attentions, pad_positions: torch.Tensor) -> float:
    """Mean attention weight (last layer, all heads) directed at PAD key positions."""
    last_layer = attentions[-1]  # (batch, heads, seq_q, seq_k)
    batch, heads, seq_q, seq_k = last_layer.shape
    pad_mask = pad_positions[:, None, None, :].expand(batch, heads, seq_q, seq_k)
    if pad_mask.sum() == 0:
        return 0.0
    return last_layer[pad_mask].mean().item()


def attention_diagnostics():
    print("\n[5] Attention-map diagnostics on real Bayan examples (mBERT):")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModel.from_pretrained(CHECKPOINT, output_attentions=True)
    model.eval()

    enc = tokenizer(BAYAN_EXAMPLES, return_tensors="pt", padding=True, truncation=True)
    tokens_0 = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    pad_positions = enc["input_ids"] == tokenizer.pad_token_id

    with torch.no_grad():
        out_with_mask = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        out_no_mask = model(input_ids=enc["input_ids"])  # deliberately omit attention_mask -> pad leak

    last_layer = out_with_mask.attentions[-1]  # (batch, heads, seq_q, seq_k)
    cls_attn = last_layer[0, :, 0, :]  # attention FROM [CLS], all heads, shortest example
    sep_index = tokens_0.index("[SEP]") if "[SEP]" in tokens_0 else None

    print(f"    Tokens (example 0): {tokens_0}")
    for h in range(cls_attn.shape[0]):
        top_idx = int(cls_attn[h].argmax())
        print(f"    Head {h}: [CLS] attends most to '{tokens_0[top_idx]}' (weight={cls_attn[h][top_idx]:.3f})")
    if sep_index is not None:
        sep_mass = cls_attn[:, sep_index].mean().item()
        print(f"    Mean [CLS]->[SEP] attention across heads: {sep_mass:.3f} "
              f"({'looks like a [SEP] sink' if sep_mass > 0.3 else 'no strong [SEP] sink'})")

    pad_mass_masked = _pad_attention_mass(out_with_mask.attentions, pad_positions)
    pad_mass_unmasked = _pad_attention_mass(out_no_mask.attentions, pad_positions)
    print(f"    Mean attention mass on [PAD] WITH attention_mask:    {pad_mass_masked:.5f}")
    print(f"    Mean attention mass on [PAD] WITHOUT attention_mask: {pad_mass_unmasked:.5f}")
    print("    -> Omitting attention_mask lets real tokens leak attention onto [PAD], "
          "which is why the serving path (Lab 7) must always pass the attention mask.")


def main():
    verify_equivalence()
    inspect_weights()
    exercise_multihead()
    verify_causal_masking()
    attention_diagnostics()


if __name__ == "__main__":
    main()
