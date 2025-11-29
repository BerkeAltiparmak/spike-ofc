# REPOS (Reference Code to Vendor into / link as submodules)

Recommended to include in the workspace for the agent:
- SCN authors’ repository: <link here>
- Bio-OFC authors’ repository: <link here>

How to add:
- As submodules:
git submodule add <SCN-repo-url> external/scn
git submodule add <BioOFC-repo-url> external/bioofc
- Or drop tarballs under `external/` and reference them here.

The agent should scan these for:
- How SCN implement spike resets, thresholding, `Ω_f`.
- Any ready helpers for delay lines / innovations from Bio-OFC code.