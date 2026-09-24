# PROMPTSMITH

**Turn a rough visual idea into an editable image-generation prompt.**

PROMPTSMITH is a Computational Creativity course project at Leiden University by Yinzhi Xie, [Shaowan Liang](https://github.com/shaowanwan), and [Xiaojing Zhao](https://github.com/JOJOZhaoX). It explores how people can steer image generation through explicit creative decisions rather than repeatedly rewriting a single long prompt.

The implementation is an interactive **Google Colab notebook with text prompts**, not a deployed web application. It combines a Qwen2.5 language model, structured prompt editing, SDXL image generation, and OpenCLIP alignment scoring.

## What it does

1. Interprets a brief as a subject and scene.
2. Builds six editable fields: colors, foreground objects, background mood, style, camera, and lighting.
3. Proposes three intent-based variants for the user to choose from.
4. Supports field-level refinements, image previews, and session logging.

**Guided mode** asks questions about each field. **Free-form mode** infers fields from the user's description. Both expose the resulting structure for inspection and refinement.

## Example from an earlier prototype

The following images are saved outputs from an **earlier SD-Turbo version**, not fresh outputs from the included SDXL notebook. The original brief was “A cute cat”, followed by “long black fur and big eyes”. The user subsequently requested a cartoon style and cel-shaded composition.

| Initial preview | After style refinement | After further refinement |
| --- | --- | --- |
| ![Initial generated cat](examples/early-sd-turbo/preview_seed0_steps1.png) | ![Cat after cartoon-style request](examples/early-sd-turbo/gen_seed0_steps1.png) | ![Cat after composition refinement](examples/early-sd-turbo/gen_seed42_steps4.png) |

The first two images use seed 0 and one inference step. The third uses seed 42 and four steps, so it is **not a controlled before/after comparison**. The [session log](examples/early-sd-turbo/session.json) records prompts, edits, parameters, and scores.

## Engineering choices and lessons

- **Explicit schema:** creative choices remain visible and editable.
- **Constrained additions:** variant generation attempts to preserve user choices while adding alternatives. This is not a formal guarantee against semantic conflicts.
- **Localized refinement:** users request a change to one field and inspect the result.
- **Traceable sessions:** logs record decisions, parameters, and outputs for review.
- **Model feedback has limits:** CLIP measures broad text-image alignment; it is not a direct measure of user satisfaction or creativity. The historical run's scores did not improve monotonically after refinement.

The project report describes exploratory comparisons of guided and free-form interaction. These are qualitative prototype observations, not evidence of production adoption or a large user study. A central lesson is that a clearer prompt edit does not necessarily produce a predictable visual change.

## Run in Google Colab

1. Download [COCONUT.ipynb](COCONUT.ipynb), then upload it through Colab's **File > Upload notebook**.
2. Select a GPU runtime. Model downloads require internet access and may require Hugging Face authentication.
3. Review the cells, then execute them in order. The notebook mounts your Google Drive and saves sessions under `MyDrive/CC_final/outputs/`.
4. Choose `A` or `B`, enter a brief, answer the prompts, and select a variant.
5. Use `show`, `refine`, `img`, and `done` in the final interaction loop.

The notebook installs Transformers, Accelerate, bitsandbytes, Safetensors, Diffusers, and OpenCLIP. Its configured models are `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`, `stabilityai/stable-diffusion-xl-base-1.0`, and OpenCLIP `ViT-B-32` with `openai` weights. Model weights are downloaded separately and remain subject to their respective terms.

## Verification

The current notebook was smoke-tested on a Google Colab Tesla T4 runtime in September 2026. The test covered loading the 4-bit Qwen model, producing structured JSON in free-form mode, generating three intent variants, loading OpenCLIP, loading SDXL, and generating a 512×512 image. The image smoke test used seed 42, eight inference steps, and guidance 5.5.

For a fast test that does not download models or require a GPU:

```bash
python tests/test_core.py
```

This checks JSON extraction with braces inside quoted strings, exact preservation of locked fields, and validation of numeric and image-size inputs.

## Limitations and verification

- GPU memory requirements depend on the runtime; SDXL and the language model can exhaust available memory.
- Dependencies use minimum versions rather than a fully locked environment.
- LLM-generated JSON and prompt interpretation can fail.
- The notebook applies an 80-token prompt cap using the LLM tokenizer; CLIP has a separate text limit and tokenizer.
- Colab currently emits non-blocking deprecation and model-configuration warnings from Transformers, Diffusers, and OpenCLIP; these do not prevent the verified workflow from running.
- Notebook outputs and execution metadata are cleared in this repository; the verification above was run in a separate cloud test copy.

## Attribution

This is a team project by Yinzhi Xie, [Shaowan Liang](https://github.com/shaowanwan), and [Xiaojing Zhao](https://github.com/JOJOZhaoX), published with the team members' consent. Individual implementation ownership is not inferred from author order. No open-source license is granted unless a license file is added later.
