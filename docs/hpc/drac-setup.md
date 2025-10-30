---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Deploying on DRAC HPC Clusters

**Last Updated:** October 2025  
**Tested On:** Nibi cluster  
**Author:** [Seamus Beairsto]

:::{note}
This guide is specific to the **Nibi cluster** on DRAC. Other clusters (Trillium, Rorqual, Fir, Narval) may have different module versions and configurations. This will be updated as testing on other clusters is completed.
:::

### Issue: `torch.cuda.is_available()` Returns False

When running an interactive job on DRAC, I was able to confirm...

[Your detailed debugging steps]

:::{tip}
Always verify your PyTorch CUDA version matches the loaded module:
\```bash
python -c "import torch; print(torch.version.cuda)"
module list  # check loaded CUDA version
\```
:::

## Understanding Python Wheels on DRAC

:::{note}
DRAC uses custom-built wheels (`.whl` files) optimized for their hardware...
[Brief explanation if you want]
:::