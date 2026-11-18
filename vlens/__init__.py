"""V-lens: draw the hidden states of a unified text-and-image model's text computation as pictures.

Modules:
    models/     one adapter per model (read text with hidden states, edit states, draw, answer)
    hidden      edits of hidden states (replace a state at a layer) and word positions
    carry       the V-lens transport (Eq. 1), simpler copies, the final-norm variant and the ablations
    transport   average-Jacobian maps (the Jacobian lens variants), layer averages and radius
    lens_fit    a generic average-Jacobian fitter (the text J-lens)
    text_lens   text lenses (logit lens, J-lens), the text readouts of Supp. D
    readout     shared helpers of the depth, held-thought and two-state experiments
    judges      the three multimodal judges and their questions
    score       CLIP ranks for the insertion-layer calibration
    lora        switchable LoRA adapters for lens tuning
    prompts, paths, results, gallery   item files, locations, output folders, picture grids
"""
