# candle-sam3

SAM 3 (Segment Anything with Concepts) image and video inference for [Candle](https://github.com/huggingface/candle), as a library crate.

This crate is a Rust port of Meta's SAM 3 implementation ([facebookresearch/sam3](https://github.com/facebookresearch/sam3)). It was developed inside the [den-sq/candle](https://github.com/den-sq/candle) fork of Candle (then named `candle_sam3`) and moved here, with its history, so that the fork contains only general Candle changes.

## License

This crate is distributed under the **SAM License** (see [`LICENSE-SAM`](LICENSE-SAM)), because it is derived from Meta's SAM 3 code. If you distribute it, or anything built from it, you must:

- include a copy of the SAM License;
- acknowledge the use of SAM in publications that use it;
- follow the licence's other terms, including the trade-control restrictions.

The SAM 3 and Medical SAM3 checkpoints are separate SAM Materials, obtained under their own repositories' terms.

Candle itself, including the `den-sq/candle` fork this crate depends on, is MIT OR Apache-2.0. `examples/sam3/common.rs` contains helpers copied from Candle's `candle-examples` under those terms.

## Use

```toml
[dependencies]
candle-sam3 = { git = "https://github.com/den-sq/candle-sam3", rev = "<commit>" }
```

The crate depends on `candle-core`, `candle-nn` and `candle-transformers` from `den-sq/candle`, pinned to one commit. A consumer that also uses Candle directly must depend on the same repository and commit, so that both use the same Candle types.

Features:

| Feature | Purpose |
|---|---|
| `cuda`, `cudnn`, `metal` | Candle device back ends |
| `media` | Caller-owned media adapters (image decoding, PNG debug output) used by the examples |
| `sam3-parity-support` | Hooks used by the parity harness ([den-sq/sam_parity](https://github.com/den-sq/sam_parity)) |
| `sam3-nvtx` | NVTX profiling ranges (needs the CUDA toolkit headers) |

## Examples

See [`examples/sam3/README.md`](examples/sam3/README.md):

```sh
cargo run --release --features media --example sam3 -- --help
```
