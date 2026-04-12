# Denny's Rules to Live By

*A PhD researcher's operating manual for rigorous, reproducible, and publishable ML research.*

Distilled from the advice of [Bengio](https://cifar.ca/cifarnews/2018/08/01/q-a-with-yoshua-bengio/), [Gebru](https://doi.org/10.1145/3458723), [Hamming](https://www.cs.virginia.edu/~robins/YouAndYourResearch.html), [Hinton](https://digitalhabitats.global/blogs/synthetic-minds/geoffrey-hinton-on-working-with-ilya-choosing-problems-and-the-power-of-intuition), [Karpathy](http://karpathy.github.io/2019/04/25/recipe/), [LeCun](https://www.youtube.com/watch?v=Umi6Vkv9DNQ), [Li](https://profiles.stanford.edu/fei-fei-li), [Ng](https://www.kdnuggets.com/2019/09/advice-building-machine-learning-career-research-papers-andrew-ng.html), [Pineau](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf), [Raschka](https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html), [Schulman](http://joschu.net/blog/opinionated-guide-ml-research.html), [Varoquaux](https://gael-varoquaux.info/about.html), and hard-won community wisdom.

---

## I. Choosing Problems

1. **Work on important problems.** Ask yourself regularly: *what are the most important open problems in my area, and am I working on one?* ([Hamming](https://www.cs.virginia.edu/~robins/YouAndYourResearch.html))
2. **Be goal-driven, not idea-driven.** Develop a vision of a capability you want to achieve, then solve problems that bring you closer. This gives you a differentiated perspective and reduces scoop risk. ([Schulman](http://joschu.net/blog/opinionated-guide-ml-research.html))
3. **Think several steps ahead.** Choose fertile ground where follow-up work is possible -- you need a coherent thesis, not isolated papers. ([Karpathy](http://karpathy.github.io/2016/09/07/phd/))
4. **Own a clear contribution.** Aim to be "the person who did X" -- something easy to describe and remember. ([Karpathy](http://karpathy.github.io/2016/09/07/phd/))
5. **Be ambitious about importance, not difficulty.** A 10x more important problem typically requires only 2-3x more effort. ([Karpathy](http://karpathy.github.io/2016/09/07/phd/))
6. **Protect thinking time.** Block time each week with no coding, writing, or reading -- just thinking about big questions. ([Bengio](https://cifar.ca/cifarnews/2018/08/01/q-a-with-yoshua-bengio/))
7. **Trust your intuition, then verify.** "If you don't allow yourself to say things that could be wrong, you're not going to be a researcher." ([Bengio](https://cifar.ca/cifarnews/2018/08/01/q-a-with-yoshua-bengio/))
8. **The north star is the problem, not the method.** Don't chase the latest technique -- ask what problem in the world your work serves. Sometimes the most impactful contribution is the dataset or benchmark that reframes the question. ([Fei-Fei Li](https://profiles.stanford.edu/fei-fei-li))

---

## II. Becoming One with the Data

9. **Spend hours with your data before writing any model code.** Scan thousands of examples. Identify duplicates, corrupted labels, imbalances, biases. Write code to search/filter/sort by any dimension. ([Karpathy](http://karpathy.github.io/2019/04/25/recipe/))
10. **Data quality > model architecture.** Most gains in real-world applications come from better data handling, not fancier models. Invest your time accordingly. ([Varoquaux](https://gael-varoquaux.info/about.html))
11. **Document your data.** Every dataset you use or release should have a datasheet: motivation, composition, collection process, preprocessing, intended uses, and limitations. ([Gebru](https://doi.org/10.1145/3458723))
12. **Visualize everything.** Distributions, outliers, correlations, class balance. If you can't visualize it, you don't understand it.
13. **Verify data just before it enters the model.** Visualize the exact tensor that goes into `model(x)` -- after all preprocessing, augmentation, and batching. ([Karpathy](http://karpathy.github.io/2019/04/25/recipe/))

---

## III. Training Neural Networks ([The Karpathy Recipe](http://karpathy.github.io/2019/04/25/recipe/))

14. **Fix your random seed first.** Use `set_seed(seed)` (see `src/utils/seed.py`) to seed Python, NumPy, PyTorch, and CUDA deterministically. Always know which seed produced which result.
15. **Start with a dumb baseline.** Linear classifier, tiny model, or even random predictions. Establish the floor before building up.
16. **Verify loss at initialization.** For softmax with C classes, initial loss should be `-log(1/C)`. If it's not, something is already wrong.
17. **Overfit a single batch first.** If your model can't memorize 2-8 examples to zero loss, you have a bug. Fix it before scaling up.
18. **Don't be a hero.** Find the most related paper and copy their simplest architecture. Add complexity one thing at a time. ([Karpathy](http://karpathy.github.io/2019/04/25/recipe/))
19. **Adam is safe.** Start with Adam at `3e-4`. Don't touch learning rate schedules until everything else works. ([Karpathy](http://karpathy.github.io/2019/04/25/recipe/))
20. **Add one thing at a time.** Never change two things simultaneously. Each modification gets its own run with a clear hypothesis for why it should help.
21. **Random search over grid search.** Neural nets are very sensitive to some hyperparameters and insensitive to others -- random search finds the sensitive ones faster.

---

## IV. Reproducibility

22. **Every run is reproducible.** The `set_seed()` function seeds `random`, `numpy`, `torch`, `torch.cuda`, and sets `PYTHONHASHSEED`, `cudnn.deterministic`, and `cudnn.benchmark`. Use `torch.use_deterministic_algorithms(True, warn_only=True)` during development. ([PyTorch Reproducibility Docs](https://pytorch.org/docs/stable/notes/randomness.html))
23. **Pin everything.** Python version, PyTorch version, CUDA version, every dependency. Use `pyproject.toml` with exact versions + a lockfile. Include a `Dockerfile` for full environment reproducibility.
24. **Snapshot configs with every run.** Hydra does this automatically -- every run gets a timestamped output directory with the full resolved config saved as YAML.
25. **Tag every experiment with a git commit.** W&B does this automatically. Never run experiments on uncommitted code.
26. **Treat raw data as immutable.** `data/raw/` is read-only. All transformations produce outputs in `data/processed/`. Document the pipeline.
27. **Use the DataLoader reproducibility protocol.** Set `worker_init_fn` and `generator` with seeded values for multi-worker data loading. ([PyTorch Reproducibility Docs](https://pytorch.org/docs/stable/notes/randomness.html))
28. **Use a reproducibility checklist.** Before submitting, verify: all hyperparameters stated, results include error bars over multiple runs, code is released, random seeds are documented. ([Pineau](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf))

---

## V. Statistical Significance

29. **No claim without evidence.** Every claim of "method A outperforms method B" must be backed by a statistical test or clearly stated as a trend. ([Raschka](https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html))

30. **Benchmark honestly.** Most published benchmarks are broken -- cherry-picked datasets, cherry-picked seeds, too few tasks to draw real conclusions. If your method wins by 0.2% on 5 datasets, that is noise, not signal. Use large benchmark suites with proper statistical tests. ([Varoquaux](https://arxiv.org/abs/2207.08815))

31. **Seed budget tiers:**

    | Compute Budget | Seeds | What to Report |
    |---|---|---|
    | **Low** (LLM-scale, days/run) | 3 | Mean +/- std, note limited seeds |
    | **Medium** (hours/run) | 5 | Mean +/- std, Wilcoxon signed-rank test |
    | **High** (minutes/run) | 10+ | Mean +/- std, Wilcoxon or paired t-test, bootstrap CIs |

32. **Use paired evaluation.** Run your method and the baseline on the *same* seeds. This induces positive correlation, giving tighter confidence intervals at the same compute budget. ([Paired Bootstrap Protocol](https://arxiv.org/abs/2511.19794))

33. **Default test: Wilcoxon signed-rank** (non-parametric, paired). Use `scipy.stats.wilcoxon(scores_ours, scores_baseline)`. It does not assume normality. Fall back to paired t-test (`ttest_rel`) if you have reason to assume normality.

34. **Report it right:**
    - Always: mean +/- std (or SEM), number of seeds, which seeds
    - When possible: p-value from paired test, 95% confidence interval
    - Always state: "Error bars represent standard deviation across N random seeds"
    - Use `src/utils/stats.py` for consistent reporting

35. **Ablation studies are mandatory.** Remove each component one at a time. Show what each piece contributes. Report in both directions: removal from full model AND addition to minimal baseline.

---

## VI. Code Quality and Readability

36. **Compact but readable.** Write the shortest code that a colleague can understand without asking you questions. No clever one-liners that require 5 minutes to parse. No unnecessary abstraction for code used once.

37. **Shape-annotated signatures with `jaxtyping` + `beartype`:**
    ```python
    from jaxtyping import Float, Int
    from torch import Tensor
    from beartype import beartype

    @beartype
    def forward(
        self,
        x: Float[Tensor, "batch channels height width"],
        labels: Int[Tensor, "batch"],
    ) -> Float[Tensor, "batch num_classes"]:
        ...
    ```
    Every function that touches tensors documents its shapes in the signature. This is your documentation AND your runtime shape-checker.

38. **Google-style docstrings** for all public functions:
    ```python
    def train_step(self, batch: dict, lr: float = 1e-4) -> dict:
        """Perform a single training step.

        Args:
            batch: Dictionary with keys 'input' (B, C, H, W) and
                'target' (B,).
            lr: Learning rate.

        Returns:
            Dictionary with 'loss' and 'accuracy' scalars.
        """
    ```

39. **Ruff for everything.** Ruff replaces Black, isort, flake8, pyupgrade in one tool. Run it via pre-commit hooks so nothing unformatted ever gets committed.

40. **Type hints on public interfaces.** Use `mypy` in relaxed mode to start; tighten gradually.

---

## VII. Documentation

41. **Auto-generate docs with MkDocs + mkdocstrings.** Write Google-style docstrings with shape annotations; MkDocs renders them into a browsable site. Deployed to GitHub Pages with one command.

42. **README as the entry point.** Every project README must have:
    - One-sentence summary of what this does
    - Key result (table or figure)
    - Installation (copy-pasteable commands)
    - Quick start (train + eval in 2 commands)
    - Citation block (BibTeX)

43. **Notebooks for exploration, scripts for production.** Notebooks are for EDA and visualization. Training, evaluation, and anything that runs on a cluster is a `.py` script with Hydra config.
    - Notebook naming convention: `01-dl-initial-data-exploration.ipynb` (number, initials, description)

---

## VIII. Experiment Workflow

44. **Use Hydra for all configuration.** No hardcoded hyperparameters in Python files. Every tunable value lives in a YAML config. Override from the command line:
    ```bash
    python src/train.py model.lr=1e-3 data.batch_size=64 seed=42
    ```

45. **Use W&B for experiment tracking.** Free for academics. Log hyperparameters, metrics, system stats, git commit, and artifacts. Tag runs with experiment names. Compare runs in the dashboard.

46. **Version-control experiment configs.** Each publishable experiment gets a config in `configs/experiment/`. This is your lab notebook -- a colleague should reproduce your result by running:
    ```bash
    python src/train.py experiment=paper_table1_row3
    ```

47. **Multi-seed launcher.** Use `scripts/run_seeds.sh` to launch the same experiment across N seeds and aggregate results:
    ```bash
    bash scripts/run_seeds.sh experiment=paper_table1_row3 seeds="42,123,456,789,1337"
    ```

---

## IX. Publishing at NeurIPS / ICML / ICLR

48. **One paper = one core contribution.** Identify your single key insight before writing. Everything in the paper argues for that one thing. ([Karpathy](http://karpathy.github.io/2016/09/07/phd/))

49. **Hit the NeurIPS checklist from day one.** Don't treat it as a last-minute checkbox. The [16-item checklist](https://neurips.cc/public/guides/PaperChecklist) (error bars, compute resources, limitations section, broader impact, etc.) should be baked into your workflow from the start.

50. **Internal deadlines:** Have a 5-page draft 2 weeks before the submission deadline. The last 2 weeks are for polishing, not panicking. ([Karpathy](http://karpathy.github.io/2016/09/07/phd/))

51. **Separate experiment design from results.** Describe *what* you tested and *why* before showing results. Reviewers want to see your reasoning, not just numbers. ([ICML Best Practices](https://icml.cc/Conferences/2022/BestPractices))

52. **Report everything reviewers will ask for:**
    - Hyperparameter search ranges AND final values
    - Compute resources (GPU type, memory, total hours)
    - Error bars with the source of variability stated
    - Ablation study
    - Limitations section (honesty will not cause rejection)

53. **Release code that passes the [5-item completeness checklist](https://medium.com/paperswithcode/ml-code-completeness-checklist-e9127b168501):**
    1. Dependencies pinned (`requirements.txt` or `pyproject.toml`)
    2. Training code with hyperparameters
    3. Evaluation code
    4. Pre-trained models (HuggingFace Hub or Zenodo)
    5. Results table in README with exact reproduction commands

---

## X. Demos and Project Pages

54. **Every paper gets a Gradio demo on HuggingFace Spaces.** A reviewer or reader should be able to try your method in their browser without installing anything. Even a simple demo dramatically increases impact.

55. **Every paper gets a project page.** Use the `project_page/` template (fork of [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)). Deploy to GitHub Pages. Include: abstract, key figure, results, BibTeX, links to paper/code/demo.

---

## XI. Research Habits

56. **Keep a research notebook.** Record daily ideas and experiments. Conduct a condensing review every 1-2 weeks. ([Schulman](http://joschu.net/blog/opinionated-guide-ml-research.html))
57. **Read two papers a week, every week.** Consistency beats cramming. Start with title/abstract/figures/intro/conclusions. ([Ng](https://www.kdnuggets.com/2019/09/advice-building-machine-learning-career-research-papers-andrew-ng.html))
58. **Reimplement to understand.** If you don't understand an algorithm well enough to code it from scratch, you don't understand it. ([Schulman](http://joschu.net/blog/opinionated-guide-ml-research.html), [Ng](https://www.kdnuggets.com/2019/09/advice-building-machine-learning-career-research-papers-andrew-ng.html))
59. **Mine PhD theses for literature reviews.** They map active research domains better than any survey paper. ([Schulman](http://joschu.net/blog/opinionated-guide-ml-research.html))
60. **Ship code publicly.** "Committing to releasing your code will force you to adopt better coding habits." ([Karpathy](http://karpathy.github.io/2016/09/07/phd/))
61. **Open source is a scientific contribution.** Maintaining widely-used research software is real research output, not "just engineering." Every grad student who writes a one-off implementation that nobody else can run wastes the community's time. ([Varoquaux](https://gael-varoquaux.info/about.html))
62. **Compound interest.** "Knowledge and productivity work like compound interest -- consistent daily effort compounds dramatically." ([Hamming](https://www.cs.virginia.edu/~robins/YouAndYourResearch.html))

---

## XII. Debugging Checklist ([Full Stack Deep Learning](https://fullstackdeeplearning.com/spring2021/lecture-7/))

63. **Systematic debugging protocol:**
    - [ ] Can the model overfit a single batch?
    - [ ] Is the loss correct at initialization?
    - [ ] Are gradients flowing? (check for NaN/zero gradients)
    - [ ] Is train/eval mode toggled correctly? (BatchNorm, Dropout)
    - [ ] Are tensor shapes correct? (broadcasting hides shape bugs silently)
    - [ ] Is the data correct just before entering the model?
    - [ ] Is the loss function receiving the right inputs? (logits vs probabilities)
    - [ ] Are learning rate and weight decay reasonable?

---

## Quick Reference Card

| What | Tool | Why |
|---|---|---|
| Config | Hydra + OmegaConf | CLI overrides, composition, auto-snapshots |
| Tracking | W&B (free academic) | Best UI, git integration, sweeps |
| Training | Lightning Fabric | Multi-GPU, mixed precision, readable training loop |
| Shapes | jaxtyping + beartype | Runtime shape checking + self-documenting |
| Docs | MkDocs + mkdocstrings | Auto-generated from docstrings |
| Formatting | Ruff + pre-commit | One tool, milliseconds, catches everything |
| Types | mypy (relaxed mode) | Catch bugs early |
| Stats | scipy.stats (wilcoxon, ttest_rel) | Paired significance tests |
| Demos | Gradio on HF Spaces | Free hosting, ML-native widgets |
| Project page | Academic Project Page Template | Battle-tested, GitHub Pages |
| Containers | Docker + NVIDIA base images | Full environment reproducibility |

---

*"Do good research, communicate it properly, people will notice and good things will happen."* -- [Andrej Karpathy](http://karpathy.github.io/2016/09/07/phd/)
